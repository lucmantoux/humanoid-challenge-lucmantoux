"""Block centre in the table frame T (EXPLAINER Module 4).

On the table the pixel is a ray meeting z_T = b/2. In the hand, closed and not hovering,
the same pixel is placed at the pinch point's depth. A blob much smaller than the
rest-time blob is treated as occluded.
"""

from __future__ import annotations

import numpy as np

from palm_prior.perception.aruco import intersect_plane_z, ray
from palm_prior.perception.hand import fill_gaps, smooth_series


def on_table(g: float, pinch_z: float, last_z: float, lift_margin: float) -> bool:
    """True when the block is still on the table, not carried.

    EXPLAINER Module 4: the gripper is open, or the pinch point is more than
    lift_margin above the block's last centre. Both mean the hand is not holding it.
    """
    if not np.isfinite(g) or g < 0.5:
        return True
    if not np.isfinite(pinch_z):
        return True
    return float(pinch_z) > float(last_z) + float(lift_margin)


def track_block(
    centroid_px: np.ndarray,
    area: np.ndarray,
    g: np.ndarray,
    p_ee: np.ndarray,
    p_ee_C: np.ndarray,
    R_CT: np.ndarray,
    t_CT: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    block_size: float,
    rest_frames: int,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame block centre in T, then gap-fill and smooth.

    centroid_px (N, 2) pixels, NaN if no blob. area (N,). g (N,) 0/1.
    p_ee (N, 3) pinch in T. p_ee_C (N, 3) the same pinch in the camera frame.
    R_CT (3, 3), t_CT (3,) the static table pose. K (3, 3).
    rest_frames is how many frames at each end count as the still rest, for the
    occlusion area reference.

    returns p_obj (N, 3) metres and valid (N,) bool.
    """
    centroid_px = np.asarray(centroid_px, float)
    area = np.asarray(area, float)
    g = np.asarray(g, float)
    p_ee = np.asarray(p_ee, float)
    p_ee_C = np.asarray(p_ee_C, float)
    n = len(area)
    assert centroid_px.shape == (n, 2), centroid_px.shape
    assert g.shape == (n,) and p_ee.shape == (n, 3) and p_ee_C.shape == (n, 3), (
        g.shape, p_ee.shape, p_ee_C.shape,
    )
    assert R_CT.shape == (3, 3) and t_CT.shape == (3,) and K.shape == (3, 3)

    c = cfg.block_track
    rest_frames = max(1, min(int(rest_frames), n))
    rest_samples = np.concatenate([area[:rest_frames], area[-rest_frames:]])
    rest_samples = rest_samples[np.isfinite(rest_samples) & (rest_samples > 0)]
    if len(rest_samples) == 0:
        rest_samples = area[np.isfinite(area) & (area > 0)]
    rest_area = float(np.median(rest_samples)) if len(rest_samples) else 0.0
    floor = float(c.occlusion_area_frac) * rest_area

    p_obj = np.full((n, 3), np.nan)
    half = float(block_size) / 2.0
    last_z = half
    for i in range(n):
        uv = centroid_px[i]
        if not np.isfinite(uv).all() or not np.isfinite(area[i]) or area[i] < floor:
            continue
        pinch_z = float(p_ee[i, 2]) if np.isfinite(p_ee[i]).all() else np.nan
        if on_table(float(g[i]), pinch_z, last_z, float(c.lift_margin)):
            direction = ray(float(uv[0]), float(uv[1]), K, dist)
            p = intersect_plane_z(direction, R_CT, t_CT, half)
        else:
            depth = float(p_ee_C[i, 2])
            if not np.isfinite(depth) or depth <= 1e-6:
                continue
            direction = ray(float(uv[0]), float(uv[1]), K, dist)
            p = R_CT.T @ (float(depth) * direction - t_CT)
        p_obj[i] = p
        last_z = float(p[2])

    p_obj = fill_gaps(p_obj, int(c.max_gap_frames))
    p_obj = smooth_series(p_obj, int(c.savgol_window), int(c.savgol_order))
    valid = np.isfinite(p_obj).all(axis=1)
    return p_obj, valid
