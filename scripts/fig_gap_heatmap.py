"""Embodiment-gap figure (EXPLAINER §25).

A grid of grasp xy offsets over ±eval.heatmap_offset. From a start just above the
block the action is five steps: descend, close, lift. Each panel is P(the block
rises by more than the success height tolerance): the fraction of ensemble members
for the two world models, and 0/1 from actually running the simulator.

Usage:
    uv run python scripts/fig_gap_heatmap.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _bootstrap  # noqa: E402,F401

from palm_prior.sim.env import Env  # noqa: E402
from palm_prior.sim.scene import build_scene  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_torch_threads  # noqa: E402
from palm_prior.wm.model import Ensemble, NormStats, rollout  # noqa: E402


def _load(path, cfg) -> tuple[Ensemble, NormStats]:
    blob = torch.load(path, weights_only=False)
    hidden = tuple(int(h) for h in blob.get("cfg_hidden", cfg.wm.hidden))
    model = Ensemble(int(cfg.wm.n_members), hidden)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, NormStats.load(blob["stats"])


def _actions(cfg) -> np.ndarray:
    """Five steps: two of them descend onto the block and close, three lift."""
    step = float(cfg.sim.max_step)
    a = np.zeros((5, 4), np.float32)
    a[0] = [-0.0, 0.0, -step, 0.0]
    a[1] = [0.0, 0.0, -step, 1.0]
    a[2:] = [0.0, 0.0, step, 1.0]
    return a


def _wm_prob(model: Ensemble, stats: NormStats, offsets: np.ndarray, cfg) -> np.ndarray:
    """Fraction of members whose predicted block rise exceeds the height tolerance.

    offsets (G, 2). returns (G,).
    """
    step = float(cfg.sim.max_step)
    z_obj = float(cfg.objects.sim_block_size) / 2.0
    h_tgt = float(cfg.goals.plate)
    actions = torch.as_tensor(_actions(cfg)).view(1, 1, 5, 4).expand(model.n_members, -1, -1, -1)
    out = np.zeros(len(offsets), float)
    with torch.no_grad():
        for i, (dx, dy) in enumerate(offsets):
            s = torch.tensor(
                [[dx, dy, 2.0 * step, 0.15, 0.0, z_obj, h_tgt, 0.0]],
                dtype=torch.float32,
            )
            s = s.unsqueeze(0).expand(model.n_members, -1, -1)
            act = actions.expand(-1, 1, -1, -1)
            dp, _ = rollout(model, s, act, 1.0, stats)
            rise = dp[:, 0, :, 2].sum(dim=-1)
            out[i] = float((rise > float(cfg.sim.success.z_tol)).float().mean())
    return out


def _reach(env: Env, target: np.ndarray, grip: float, limit: float, cfg) -> None:
    cmd = env._p_ee_target.copy()
    tol = float(cfg.scripted.reach_tol)
    for _ in range(int(cfg.scripted.max_reach_steps)):
        if float(np.linalg.norm(target - cmd)) <= tol:
            return
        dp = target - cmd
        norm = float(np.linalg.norm(dp))
        if norm > limit:
            dp = dp * (limit / norm)
        env.step(np.concatenate([dp, [grip]]))
        cmd = cmd + dp


def _true_prob(env: Env, offsets: np.ndarray, cfg) -> np.ndarray:
    """1 when the simulated block actually rises, else 0."""
    saved = env.save_state()
    step = float(cfg.sim.max_step)
    fast = step
    out = np.zeros(len(offsets), float)
    for i, (dx, dy) in enumerate(offsets):
        env.restore_state(saved)
        block = env.p_obj_W()
        z0 = float(block[2])
        hover = np.array([block[0] + dx, block[1] + dy, z0 + 2.0 * step])
        _reach(env, hover, 0.0, fast, cfg)
        for a in _actions(cfg):
            env.step(a)
        out[i] = float(env.p_obj_W()[2] - z0 > float(cfg.sim.success.z_tol))
    env.restore_state(saved)
    return out


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_torch_threads()
    ckpt = resolve(cfg.paths.checkpoints)
    human_path = ckpt / "wm_pretrain_ft_N0_seed0.pt"
    adapted_path = ckpt / "wm_pretrain_ft_N25_seed0.pt"
    cache = ckpt / "_human_pretrain_seed0.pt"
    if not human_path.exists() and cache.exists():
        human_path = cache
    missing = [str(p.name) for p in (human_path, adapted_path) if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoint(s) {missing}; run scripts/exp_e2.py first")

    half = float(cfg.eval.heatmap_offset)
    axis = np.linspace(-half, half, int(round(2 * half / (half / 3))) + 1)
    dx, dy = np.meshgrid(axis, axis, indexing="xy")
    offsets = np.stack([dx.ravel(), dy.ravel()], axis=1)

    with Timer("fig_gap_heatmap"):
        panels = {
            "human only": _wm_prob(*_load(human_path, cfg), offsets, cfg),
            "adapted, 25 episodes": _wm_prob(*_load(adapted_path, cfg), offsets, cfg),
        }
        scene = build_scene(cfg)
        env = Env("plate", 0, cfg, scene=scene)
        env.reset()
        panels["true simulator"] = _true_prob(env, offsets, cfg)
        env.close()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (title, values) in zip(axes, panels.items()):
        grid = values.reshape(dx.shape)
        im = ax.imshow(
            grid, origin="lower", vmin=0, vmax=1, cmap="viridis",
            extent=[axis[0] * 100, axis[-1] * 100, axis[0] * 100, axis[-1] * 100],
        )
        ax.set_title(title)
        ax.set_xlabel("grasp offset x (cm)")
    axes[0].set_ylabel("grasp offset y (cm)")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="P(block lifts)")
    out = ensure_dir(resolve(cfg.paths.results)) / "gap_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
