"""Shrinking-horizon MPC (EXPLAINER §10 and Module 10).

The decision is a residual on a retargeted path: 8 knots × 3 axes, plus two integer
shifts of the close and open times. CEM scores each candidate by rolling it out in
the world model. The first `execute_steps` actions are applied, then the search is
warm-started from the same residual shifted forward by that many steps.
"""

from __future__ import annotations

import numpy as np
import torch

from palm_prior.plan.cem import cem
from palm_prior.retarget.build_plan import build_plan, residual_path
from palm_prior.state import next_state
from palm_prior.wm.model import rollout


def shift_knots(knots: np.ndarray, steps: int, n: int) -> np.ndarray:
    """Move a knot residual `steps` along a path of length n, and pin knot 0 at 0.

    knots (K, 3). returns (K, 3).
    """
    knots = np.asarray(knots, float).copy()
    knots[0] = 0.0
    k = len(knots)
    where = np.linspace(0.0, n - 1, k)
    grid = np.arange(n)
    residual = np.column_stack([np.interp(grid, where, knots[:, d]) for d in range(3)])
    shifted = np.vstack([residual[steps:], np.repeat(residual[-1:], steps, axis=0)])
    shifted = shifted - shifted[0]
    out = np.column_stack([np.interp(where, grid, shifted[:, d]) for d in range(3)])
    out[0] = 0.0
    return out


def _clip_dp(dp: np.ndarray, limit: float) -> np.ndarray:
    out = np.array(dp, float, copy=True)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    scale = np.ones_like(norm)
    np.divide(limit, norm, out=scale, where=norm > limit)
    return out * scale


def actions_from_plan(plan, max_step: float) -> np.ndarray:
    """Deltas of the commanded path, clipped, with the gripper of the arrival step. (K-1, 4)."""
    dp = _clip_dp(np.diff(plan.p_star, axis=0), float(max_step))
    g = plan.grip[1:, None]
    return np.concatenate([dp, g], axis=1).astype(np.float32)


def _smooth(theta: np.ndarray, n_knots: int) -> np.ndarray:
    """Sum of squared steps between residual knots. theta (N, 26) → (N,)."""
    knots = theta[:, : n_knots * 3].reshape(len(theta), n_knots, 3).copy()
    knots[:, 0] = 0.0
    diff = knots[:, 1:] - knots[:, :-1]
    return (diff ** 2).sum(axis=(1, 2))


def _miss(state: torch.Tensor, block_size: float, d_xy: float, d_z: float) -> torch.Tensor:
    """D of EXPLAINER §10. state (..., 8) → (...,)."""
    d_to = state[..., 3:5]
    z = state[..., 5]
    h = state[..., 6]
    xy = (d_to ** 2).sum(dim=-1) / (float(d_xy) ** 2)
    height = (z - h - float(block_size) / 2.0) ** 2 / (float(d_z) ** 2)
    return xy + height


@torch.no_grad()
def _wm_cost(model, stats, s0: np.ndarray, actions: np.ndarray, smooth: np.ndarray, cfg) -> np.ndarray:
    """J for a batch of action sequences. actions (N, H, 4), smooth (N,) → (N,)."""
    device = next(model.parameters()).device
    n = actions.shape[0]
    m = model.n_members
    s = torch.as_tensor(s0, dtype=torch.float32, device=device).view(1, 1, 8).expand(m, n, 8).clone()
    act = torch.as_tensor(actions, dtype=torch.float32, device=device).unsqueeze(0).expand(m, -1, -1, -1)
    dp, _ = rollout(model, s, act, 1.0, stats)
    state = s
    for k in range(act.shape[2]):
        state = next_state(state, act[:, :, k], dp[:, :, k])
    d = _miss(state, float(cfg.objects.sim_block_size), float(cfg.mpc.d_xy_scale), float(cfg.mpc.d_z_scale))
    mean = d.mean(dim=0)
    std = d.std(dim=0)
    j = mean + float(cfg.mpc.beta) * std
    return j.cpu().numpy() + float(cfg.mpc.w_smooth) * smooth


def _timing_grid(limit: int) -> list[tuple[int, int]]:
    vals = range(-int(limit), int(limit) + 1)
    return [(a, b) for a in vals for b in vals]


class Planner:
    """One CEM search around a clip, then a short execution, repeated until the path ends."""

    def __init__(self, model, stats, clip, cfg):
        self.model = model
        self.stats = stats
        self.clip = clip
        self.cfg = cfg
        self.n_knots = int(cfg.retarget.n_knots)
        limit = int(cfg.mpc.timing_shift_range)
        self._timings = _timing_grid(limit)

    def _plans(self, block_xy, target_xy, h_tgt, ee) -> dict:
        out = {}
        for shift in self._timings:
            out[shift] = build_plan(
                self.clip, block_xy, target_xy, h_tgt,
                None, np.array(shift, float), self.cfg, ee_start_W=ee,
            )
        return out

    def _actions_for(self, plans, theta: np.ndarray, cursor: int) -> np.ndarray:
        """theta (N, 26) → the next actions from `cursor`, at most 60 steps. (N, H, 4)."""
        limit = int(self.cfg.mpc.timing_shift_range)
        timing = np.clip(np.rint(theta[:, -2:]), -limit, limit).astype(int)
        sequences = []
        for i, (dc, do) in enumerate(timing):
            plan = plans[(int(dc), int(do))]
            knots = theta[i, : self.n_knots * 3].reshape(self.n_knots, 3)
            path = plan.p_star + residual_path(len(plan.p_star), knots, self.n_knots)
            dp = _clip_dp(np.diff(path, axis=0), float(self.cfg.sim.max_step))
            sequences.append(np.concatenate([dp, plan.grip[1:, None]], axis=1))
        remain = min(len(s) - cursor for s in sequences)
        horizon = max(1, min(remain, 60))
        return np.stack([s[cursor : cursor + horizon] for s in sequences]).astype(np.float32)

    def replan(self, s0, block_xy, target_xy, h_tgt, ee, mu, sigma, rng, cursor: int):
        """One CEM search from path index `cursor`.

        returns mu (26,), sigma (26,), and the mean plan's actions from the cursor. (H, 4).
        """
        plans = self._plans(block_xy, target_xy, h_tgt, ee)
        cfg = self.cfg

        def score(theta: np.ndarray) -> np.ndarray:
            actions = self._actions_for(plans, theta, cursor)
            return _wm_cost(self.model, self.stats, s0, actions, _smooth(theta, self.n_knots), cfg)

        mu, sigma = cem(
            mu, sigma, score,
            n_samples=int(cfg.mpc.samples),
            n_elites=int(cfg.mpc.elites),
            n_iters=int(cfg.mpc.iters),
            sigma_min=np.concatenate([
                np.full(self.n_knots * 3, float(cfg.mpc.sigma_knot) / 4.0),
                np.ones(2),
            ]),
            rng=rng,
            integer_dims=(self.n_knots * 3, self.n_knots * 3 + 1),
        )
        actions = self._actions_for(plans, mu.reshape(1, -1), cursor)[0]
        length = min(len(p.p_star) for p in plans.values())
        return mu, sigma, actions, length


def initial_belief(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Mean residual 0, knot sigma and timing sigma from the config. ((26,), (26,))."""
    k = int(cfg.retarget.n_knots)
    mu = np.zeros(k * 3 + 2)
    sigma = np.concatenate([
        np.full(k * 3, float(cfg.mpc.sigma_knot)),
        np.full(2, float(cfg.mpc.sigma_timing)),
    ])
    return mu, sigma


def run_episode(env, planner: Planner, rng: np.random.Generator) -> bool:
    """Re-plan every execute_steps until the retargeted path is finished, then settle.

    returns whether the episode met the success test.
    """
    cfg = planner.cfg
    mu, sigma = initial_belief(cfg)
    cursor = 0
    length = 1
    while cursor < length - 1:
        mu, sigma, actions, length = planner.replan(
            env.get_state(), env.layout.block_xy, env.layout.target_xy, env.h_tgt,
            env.p_ee_W(), mu, sigma, rng, cursor,
        )
        n_exec = min(int(cfg.mpc.execute_steps), len(actions), length - 1 - cursor)
        if n_exec <= 0:
            break
        for a in actions[:n_exec]:
            env.step(a)
        knots = mu[: planner.n_knots * 3].reshape(planner.n_knots, 3)
        mu = mu.copy()
        mu[: planner.n_knots * 3] = shift_knots(knots, n_exec, length).reshape(-1)
        cursor += n_exec
    env.settle()
    return bool(env.success())
