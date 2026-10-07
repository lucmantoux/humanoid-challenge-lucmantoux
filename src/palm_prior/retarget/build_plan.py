"""Turn one human clip into a robot end-effector path (EXPLAINER Module 6).

The horizontal part is the two-anchor similarity of §8: the pinch at the grasp lands on
the block, the pinch at the release lands on the target. Height keeps the shape of the
human lift, scaled by kappa, and adds the target height after the release.

Deviation from the written z formula. §8 sets the grasp height to the measured pinch
height z_h(t_g). On these clips that pinch sits about 5 cm above the table even while
the hand is flat, so a robot that closed there would close in the air. The grasp height
used here is the sim block centre. The lift above that, kappa (z_h(t) - z_h(t_g)), and
the h_tgt ramp are the rest of the formula.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from palm_prior.human.transitions import sample_times
from palm_prior.perception.hand import fill_gaps
from palm_prior.retarget.two_anchor import apply_xy, kappa, release_ramp, similarity


class Plan(NamedTuple):
    """A 10 Hz path in the world frame W."""

    p_star: np.ndarray   # (K, 3) commanded tcp
    grip: np.ndarray     # (K,) 1 while closed
    k_close: int
    k_open: int
    alpha: complex
    beta: complex


def path_at_robot_rate(clip: dict, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Resample a saved clip to the robot rate. Returns p_ee (K, 3) in T and g (K,).

    clip is the dict inside data/human/clip_NNN.npz (t_frame, p_ee, g). Times follow
    Module 5: t_k = k / (rate_hz * tau), the same grid the transitions were built on.
    """
    t_frame = np.asarray(clip["t_frame"], float)
    p_ee = np.asarray(clip["p_ee"], float)
    g = np.asarray(clip["g"], float)
    assert p_ee.shape == (len(t_frame), 3) and g.shape == (len(t_frame),)
    dt = float(np.median(np.diff(t_frame))) if len(t_frame) > 1 else 1.0 / 30.0
    duration = float(t_frame[-1] + dt)
    t_q = sample_times(duration, float(cfg.human.rate_hz), float(cfg.human.tau))
    ee = np.full((len(t_q), 3), np.nan)
    finite = np.isfinite(p_ee).all(axis=1)
    if int(finite.sum()) >= 2:
        for d in range(3):
            ee[:, d] = np.interp(t_q, t_frame[finite], p_ee[finite, d])
    idx = np.searchsorted(t_frame, t_q, side="left")
    idx = np.clip(idx, 0, len(t_frame) - 1)
    prev = np.clip(idx - 1, 0, len(t_frame) - 1)
    use_prev = np.abs(t_frame[prev] - t_q) <= np.abs(t_frame[idx] - t_q)
    gg = g[np.where(use_prev, prev, idx)]
    ee = fill_gaps(ee, int(cfg.hand.max_gap_frames))
    # A still-missing sample would make the similarity NaN. Hold the nearest finite one.
    good = np.isfinite(ee).all(axis=1)
    if not good.all() and good.any():
        last = ee[int(np.argmax(good))]
        for i in range(len(ee)):
            if good[i]:
                last = ee[i]
            else:
                ee[i] = last
    assert ee.shape == (len(t_q), 3) and gg.shape == (len(t_q),)
    return ee, gg


def grasp_window(g: np.ndarray) -> tuple[int, int]:
    """First close and the open after it, as indices into g. (k_close, k_open).

    A clip that never closes is treated as open throughout: the window is empty
    (k_open == k_close) and the caller keeps the gripper open.
    """
    g = np.asarray(g, float)
    closed = g >= 0.5
    if not closed.any():
        return 0, 0
    k_close = int(np.argmax(closed))
    later = np.where(~closed[k_close + 1 :])[0]
    k_open = (k_close + 1 + int(later[0])) if len(later) else (len(g) - 1)
    return k_close, k_open


def _shift(k_close: int, k_open: int, timing_shift: np.ndarray | None, n: int) -> tuple[int, int]:
    if timing_shift is None:
        return k_close, k_open
    shift = np.asarray(timing_shift, float).reshape(2)
    k_close = int(np.clip(k_close + int(np.round(shift[0])), 0, n - 1))
    k_open = int(np.clip(k_open + int(np.round(shift[1])), 0, n - 1))
    if k_open < k_close:
        k_open = k_close
    return k_close, k_open


def residual_path(n: int, knots: np.ndarray | None, n_knots: int) -> np.ndarray:
    """Linear interpolation of the knots over the path. Knot 0 is pinned at 0. (n, 3)."""
    if knots is None:
        return np.zeros((n, 3))
    knots = np.asarray(knots, float).copy()
    assert knots.shape == (n_knots, 3), knots.shape
    knots[0] = 0.0
    where = np.linspace(0.0, n - 1, n_knots)
    out = np.column_stack([np.interp(np.arange(n), where, knots[:, d]) for d in range(3)])
    assert out.shape == (n, 3), out.shape
    return out


def build_plan(
    clip,
    block_W: np.ndarray,
    tgt_W: np.ndarray,
    h_tgt: float,
    residual_knots: np.ndarray | None,
    timing_shift: np.ndarray | None,
    cfg,
    ee_start_W: np.ndarray | None = None,
    alpha_beta: tuple[complex, complex] | None = None,
) -> Plan:
    """Retarget `clip` onto one robot layout.

    clip            npz-like with t_frame (N,), p_ee (N, 3) in T, g (N,)
    block_W, tgt_W  (2,) or (3,) block and target in W. Only xy is used.
    h_tgt           target top height, metres
    residual_knots  (8, 3) or None. Knot 0 is replaced by 0.
    timing_shift    (2,) integer steps added to the close and open indices, or None
    ee_start_W      (3,) current tcp. The approach before the grasp is blended from
                    here so the arm starts where it actually is. The grasp and the
                    release are not moved.
    alpha_beta      if given, this fixed similarity is used instead of the two anchors.
                    That is the naive baseline: the same transform for every clip, with
                    no dependence on this layout's block and target.

    returns p_star (K, 3), grip (K,), k_close, k_open.
    """
    p_h, g = path_at_robot_rate(clip, cfg)
    n = len(p_h)
    assert n >= 2, n
    k_close, k_open = _shift(*grasp_window(g), timing_shift, n)
    block_xy = np.asarray(block_W, float).reshape(-1)[:2]
    target_xy = np.asarray(tgt_W, float).reshape(-1)[:2]
    assert block_xy.shape == target_xy.shape == (2,)

    if alpha_beta is None:
        alpha, beta = similarity(
            p_h[k_close, :2], p_h[k_open, :2], block_xy, target_xy,
            float(cfg.retarget.degenerate_dist),
        )
    else:
        alpha, beta = alpha_beta
    xy = apply_xy(p_h[:, :2], alpha, beta)
    kap = kappa(alpha, float(cfg.retarget.kappa_min), float(cfg.retarget.kappa_max))
    ramp = release_ramp(np.arange(n), k_close, k_open)
    # Grasp height is the block centre, not the measured pinch. See the module docstring.
    z_grasp = float(cfg.objects.sim_block_size) / 2.0
    # A noisy dip in the measured pinch would drive the block through the table.
    lift = np.maximum(kap * (p_h[:, 2] - float(p_h[k_close, 2])), 0.0)
    z = z_grasp + lift + float(h_tgt) * ramp
    p_star = np.column_stack([xy, z])
    p_star = p_star + residual_path(n, residual_knots, int(cfg.retarget.n_knots))

    if ee_start_W is not None and k_close > 0:
        start = np.asarray(ee_start_W, float).reshape(3)
        for k in range(k_close):
            w = k / float(k_close)
            p_star[k] = (1.0 - w) * start + w * p_star[k]

    grip = np.zeros(n)
    if k_open > k_close:
        grip[k_close:k_open] = 1.0
    assert p_star.shape == (n, 3) and grip.shape == (n,)
    return Plan(p_star=p_star, grip=grip, k_close=k_close, k_open=k_open, alpha=alpha, beta=beta)
