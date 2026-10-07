"""Palm-size depth, lid anchors, Kalman smoothing, One Euro filter (FIX.md §2).

Landmark z from MediaPipe is not used. world_landmarks are used only as a tilt
factor. Lengths are Luc's, in metres inside these functions.
"""

from __future__ import annotations

import numpy as np

THUMB_TIP = 4
INDEX_TIP = 8


def segment_lengths_m(cfg) -> list[tuple[int, int, float]]:
    """(a, b, length_m) from hand_depth.segments. length_mm is millimetres."""
    out = []
    for seg in cfg.hand_depth.segments:
        length_mm = seg.get("length_mm")
        if length_mm is None:
            raise ValueError("a hand segment length is null")
        out.append((int(seg.a), int(seg.b), float(length_mm) / 1000.0))
    if len(out) != 8:
        raise ValueError(f"expected 8 hand segments, got {len(out)}")
    return out


def fit_scale(ell: np.ndarray, length: np.ndarray) -> tuple[float, np.ndarray]:
    """Least squares s in L = s * ell, through the origin, and relative residuals.

    ell, length (n,). s is metres per pixel when ell is in pixels and length in metres.
    """
    ell = np.asarray(ell, float)
    length = np.asarray(length, float)
    assert ell.shape == length.shape and ell.ndim == 1
    denom = float(np.dot(ell, ell))
    if denom < 1e-18:
        return float("nan"), np.full(len(ell), np.nan)
    s = float(np.dot(length, ell) / denom)
    residual = np.abs(s * ell - length) / np.maximum(length, 1e-12)
    return s, residual


def fit_frame(
    pixels: np.ndarray,
    world: np.ndarray,
    segments: list[tuple[int, int, float]],
    f_px: float,
    min_foreshorten: float,
    max_residual: float,
    min_segments: int,
    tilt_correction: bool = True,
) -> dict:
    """One frame. pixels (21, 2), world (21, 3). Returns Z_ruler and the fit."""
    ell_list = []
    length_list = []
    ids = []
    for a, b, length in segments:
        if not np.isfinite(pixels[a]).all() or not np.isfinite(pixels[b]).all():
            continue
        ell = float(np.linalg.norm(pixels[b] - pixels[a]))
        if ell < 1e-6:
            continue
        if tilt_correction and np.isfinite(world[a]).all() and np.isfinite(world[b]).all():
            delta = world[b] - world[a]
            norm = float(np.linalg.norm(delta))
            c = float(np.linalg.norm(delta[:2]) / norm) if norm > 1e-9 else 0.0
        else:
            c = 1.0
        if c < float(min_foreshorten):
            continue
        ell_list.append(ell / c)
        length_list.append(length)
        ids.append((a, b))
    ell = np.asarray(ell_list, float)
    length = np.asarray(length_list, float)
    kept = np.ones(len(ell), dtype=bool)
    s = float("nan")
    residual = np.full(len(ell), np.nan)
    for _ in range(2):
        if int(kept.sum()) < 1:
            break
        s, residual_kept = fit_scale(ell[kept], length[kept])
        residual = np.full(len(ell), np.nan)
        residual[kept] = residual_kept
        drop = kept & (residual > float(max_residual))
        if not drop.any():
            break
        kept[drop] = False
    n_kept = int(kept.sum())
    if n_kept < int(min_segments) or not np.isfinite(s):
        return {
            "Z": np.nan, "s": s, "Z_std": np.nan, "Z_se": np.nan,
            "n_kept": n_kept, "depth_ok": False, "dropped": [ids[i] for i in range(len(ids)) if not kept[i]],
        }
    z_i = float(f_px) * length[kept] / ell[kept]
    z_std = float(np.std(z_i)) if n_kept > 1 else 0.0
    z_se = z_std / np.sqrt(n_kept)
    return {
        "Z": float(f_px) * s,
        "s": s,
        "Z_std": z_std,
        "Z_se": z_se,
        "n_kept": n_kept,
        "depth_ok": True,
        "dropped": [ids[i] for i in range(len(ids)) if i < len(kept) and not kept[i]],
        "kept_ids": [ids[i] for i in range(len(ids)) if kept[i]],
    }


def filter_log_z(
    y: np.ndarray,
    sigma_y: np.ndarray,
    fps: float,
    q: float,
    gate_sigma: float,
    v_max: float,
    max_consecutive_reject: int,
    rts: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Constant-velocity Kalman on log Z, then an RTS backward pass.

    y (n,) log of the ruler depth. sigma_y (n,) is Z_se / Z.
    returns smoothed Z (n,), rejected (n,) bool, n_reinit.
    """
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)
    n = len(y)
    te = 1.0 / float(fps)
    f = np.array([[1.0, te], [0.0, 1.0]])
    q_mat = float(q) * np.array([[te ** 4 / 4, te ** 3 / 2], [te ** 3 / 2, te ** 2]])
    rejected = np.zeros(n, dtype=bool)
    log_out = np.full(n, np.nan)
    x_filt = np.zeros((n, 2))
    p_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    p_pred = np.zeros((n, 2, 2))
    finite = np.isfinite(y)
    if not finite.any():
        return np.full(n, np.nan), rejected, 0
    k0 = int(np.argmax(finite))
    x = np.array([y[k0], 0.0])
    p = np.diag([max(float(sigma_y[k0]) ** 2, 1e-4), float(q)])
    streak = 0
    n_reinit = 0
    for k in range(n):
        x_pred[k] = f @ x
        p_pred[k] = f @ p @ f.T + q_mat
        if not np.isfinite(y[k]):
            rejected[k] = True
            x, p = x_pred[k], p_pred[k]
        else:
            nu = float(y[k] - x_pred[k, 0])
            sy = float(sigma_y[k]) if np.isfinite(sigma_y[k]) else 0.05
            s_var = float(p_pred[k, 0, 0] + sy ** 2)
            z_prev = float(np.exp(x[0]))
            speed_lim = float(v_max) / (float(fps) * max(z_prev, 1e-3))
            jump = abs(float(y[k] - x[0]))
            if abs(nu) > float(gate_sigma) * np.sqrt(max(s_var, 1e-12)) or jump > speed_lim:
                rejected[k] = True
                streak += 1
                x, p = x_pred[k], p_pred[k]
                if streak >= int(max_consecutive_reject):
                    x = np.array([y[k], 0.0])
                    p = np.diag([max(sy ** 2, 1e-4), float(q)])
                    streak = 0
                    n_reinit += 1
                    rejected[k] = False
            else:
                streak = 0
                gain = p_pred[k, :, 0] / s_var
                x = x_pred[k] + gain * nu
                p = (np.eye(2) - np.outer(gain, np.array([1.0, 0.0]))) @ p_pred[k]
        x_filt[k] = x
        p_filt[k] = p
        log_out[k] = x[0]
    if rts and n > 1:
        xs = x_filt.copy()
        for k in range(n - 2, -1, -1):
            try:
                c = p_filt[k] @ f.T @ np.linalg.inv(p_pred[k + 1])
            except np.linalg.LinAlgError:
                continue
            xs[k] = x_filt[k] + c @ (xs[k + 1] - x_pred[k + 1])
        log_out = xs[:, 0]
    return np.exp(log_out), rejected, n_reinit


def one_euro(x: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    """One Euro filter (Casiez et al. 2012). x (n,) or (n, d) in pixels."""
    x = np.asarray(x, float)
    te = 1.0 / float(fps)

    def alpha(fc: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(fc, 1e-6))
        return 1.0 / (1.0 + tau / te)

    flat = x.reshape(len(x), -1)
    out = np.full_like(flat, np.nan)
    start = int(np.argmax(np.isfinite(flat).all(axis=1))) if np.isfinite(flat).any() else 0
    hat = flat[start].copy()
    speed_hat = np.zeros_like(hat)
    out[start] = hat
    a_d = alpha(float(d_cutoff))
    for k in range(start + 1, len(flat)):
        if not np.isfinite(flat[k]).all():
            out[k] = hat
            continue
        speed = (flat[k] - hat) / te
        speed_hat = a_d * speed + (1.0 - a_d) * speed_hat
        fc = float(min_cutoff) + float(beta) * float(np.linalg.norm(speed_hat))
        a = alpha(fc)
        hat = a * flat[k] + (1.0 - a) * hat
        out[k] = hat
    return out.reshape(x.shape)


def kappa_at(index: np.ndarray, anchors: list[tuple[int, float]]) -> np.ndarray:
    """Piecewise-linear kappa. anchors are (frame, kappa), sorted."""
    index = np.asarray(index, float)
    if not anchors:
        return np.ones(len(index))
    frames = np.array([a[0] for a in anchors], float)
    values = np.array([a[1] for a in anchors], float)
    order = np.argsort(frames)
    frames, values = frames[order], values[order]
    out = np.interp(index, frames, values)
    out[index <= frames[0]] = values[0]
    out[index >= frames[-1]] = values[-1]
    return out


def backproject(u: float, v: float, depth: float, k_inv: np.ndarray, r_ct: np.ndarray, t_ct: np.ndarray) -> np.ndarray:
    """Pinch pixel at a camera depth, into the table frame. Returns (3,)."""
    x_cam = float(depth) * (k_inv @ np.array([u, v, 1.0]))
    return r_ct.T @ (x_cam - t_ct)


def camera_depth(x_table: np.ndarray, r_ct: np.ndarray, t_ct: np.ndarray) -> float:
    """Z component of R X + t."""
    return float((r_ct @ np.asarray(x_table, float) + t_ct)[2])


def grasp_intervals(g: np.ndarray) -> list[tuple[int, int]]:
    """Each rising edge to the next falling edge, as (k_close, k_open)."""
    g = np.asarray(g, float)
    closed = np.isfinite(g) & (g >= 0.5)
    intervals = []
    k = 0
    while k < len(closed):
        if not closed[k]:
            k += 1
            continue
        k_open = k
        while k_open < len(closed) and closed[k_open]:
            k_open += 1
        intervals.append((k, max(k, k_open - 1)))
        k = k_open
    return intervals
