"""Sample a tracked clip into the shared state and action (EXPLAINER Module 5)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from palm_prior.state import attached, build_state


def sample_times(duration_s: float, rate_hz: float, tau: float) -> np.ndarray:
    """Video times t_k = k / (rate_hz * tau), strictly inside [0, duration_s).

    tau stretches a fast human clip onto the slower robot rate (EXPLAINER Module 5).
    returns (K,) seconds.
    """
    assert duration_s > 0 and rate_hz > 0 and tau > 0, (duration_s, rate_hz, tau)
    step = 1.0 / (float(rate_hz) * float(tau))
    k_max = int(np.floor((float(duration_s) - 1e-9) / step))
    return step * np.arange(k_max + 1, dtype=float)


def _interp_rows(t_frame: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (N,) or (N, D) series. NaN where an endpoint is missing."""
    values = np.asarray(values, float)
    squeeze = values.ndim == 1
    cols = values.reshape(len(values), -1) if squeeze else values
    out = np.full((len(t_query), cols.shape[1]), np.nan)
    finite = np.isfinite(cols).all(axis=1)
    if int(finite.sum()) < 2:
        return out[:, 0] if squeeze else out
    tf = t_frame[finite]
    src = cols[finite]
    for d in range(cols.shape[1]):
        out[:, d] = np.interp(t_query, tf, src[:, d], left=np.nan, right=np.nan)
    # np.interp extrapolates with the edge value. Kill queries outside the finite span
    # and queries whose neighbouring frames were not both finite.
    inside = (t_query >= tf[0]) & (t_query <= tf[-1])
    out[~inside] = np.nan
    idx = np.searchsorted(t_frame, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_frame) - 2)
    both = finite[idx] & finite[np.minimum(idx + 1, len(finite) - 1)]
    out[~both] = np.nan
    return out[:, 0] if squeeze else out


def _nearest(t_frame: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Nearest-frame sample of a (N,) series. NaN when that frame is missing."""
    values = np.asarray(values, float)
    out = np.full(len(t_query), np.nan)
    if len(t_frame) == 0:
        return out
    idx = np.searchsorted(t_frame, t_query, side="left")
    idx = np.clip(idx, 0, len(t_frame) - 1)
    prev = np.clip(idx - 1, 0, len(t_frame) - 1)
    use_prev = np.abs(t_frame[prev] - t_query) < np.abs(t_frame[idx] - t_query)
    idx = np.where(use_prev, prev, idx)
    out = values[idx]
    return out


class ClipTransitions(NamedTuple):
    """One clip, already at the robot rate. Length K is the number of transitions, not samples."""

    s: np.ndarray          # (K, 8)
    a: np.ndarray          # (K, 4)
    dp_obj: np.ndarray     # (K, 3)
    attach: np.ndarray     # (K,)
    p_ee: np.ndarray       # (K+1, 3) sampled pinch, kept so a failure can be inspected
    p_obj: np.ndarray      # (K+1, 3)


def clip_transitions(
    t_frame: np.ndarray,
    p_ee: np.ndarray,
    p_obj: np.ndarray,
    g: np.ndarray,
    p_tgt_xy: np.ndarray,
    h_tgt: float,
    block_size: float,
    attach_margin: float,
    duration_s: float,
    rate_hz: float,
    tau: float,
) -> ClipTransitions:
    """Resample one clip and build s, a, dp_obj, attach. Rows with a missing endpoint are dropped.

    t_frame (N,) seconds. p_ee, p_obj (N, 3) in T. g (N,) 0/1. p_tgt_xy (2,) metres.
    """
    t_frame = np.asarray(t_frame, float)
    p_ee = np.asarray(p_ee, float)
    p_obj = np.asarray(p_obj, float)
    g = np.asarray(g, float)
    p_tgt_xy = np.asarray(p_tgt_xy, float).reshape(2)
    assert p_ee.shape == p_obj.shape == (len(t_frame), 3)
    assert g.shape == (len(t_frame),)

    t_q = sample_times(duration_s, rate_hz, tau)
    ee = _interp_rows(t_frame, p_ee, t_q)
    obj = _interp_rows(t_frame, p_obj, t_q)
    gg = _nearest(t_frame, g, t_q)
    good = np.isfinite(ee).all(axis=1) & np.isfinite(obj).all(axis=1) & np.isfinite(gg)
    # A transition needs the sample and the next sample.
    pair = good[:-1] & good[1:]
    ee_k, ee_n = ee[:-1][pair], ee[1:][pair]
    obj_k, obj_n = obj[:-1][pair], obj[1:][pair]
    g_n = gg[1:][pair]
    g_k = gg[:-1][pair]
    h = np.full(len(ee_k), float(h_tgt))
    tgt = np.repeat(p_tgt_xy.reshape(1, 2), len(ee_k), axis=0)
    s = build_state(ee_k, obj_k, tgt, h, g_k)
    a = np.concatenate([ee_n - ee_k, g_n.reshape(-1, 1)], axis=1)
    dp = obj_n - obj_k
    att = attached(g_k, obj_k[:, 2], float(block_size), float(attach_margin))
    assert s.shape == (len(ee_k), 8) and a.shape == (len(ee_k), 4) and dp.shape == (len(ee_k), 3)
    return ClipTransitions(s=s, a=a, dp_obj=dp, attach=att, p_ee=ee_k, p_obj=obj_k)


def grasp_count(g: np.ndarray) -> int:
    """How many times the gripper goes from open to closed. NaN is not an edge."""
    g = np.asarray(g, float)
    n = 0
    for i in range(1, len(g)):
        if np.isfinite(g[i]) and np.isfinite(g[i - 1]) and g[i - 1] < 0.5 <= g[i]:
            n += 1
    return n


def expected_grasps(clip_type: str) -> int:
    """Success, F1, F2 and F3 have one grasp. F4 has none (EXPLAINER Module 5)."""
    if clip_type == "F4":
        return 0
    if clip_type in ("success", "F1", "F2", "F3"):
        return 1
    raise ValueError(f"unknown clip type {clip_type!r}")
