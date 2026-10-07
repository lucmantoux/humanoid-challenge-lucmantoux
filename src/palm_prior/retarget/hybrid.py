"""Hybrid retargeting (FIX.md §5).

xy comes from the hand, pinned so the grasp lands on the block and the release
lands on the target. Height near those objects is a minimum-jerk profile.
naive and two_anchor are not in this file.
"""

from __future__ import annotations

import numpy as np

from palm_prior.retarget.two_anchor import apply_xy, similarity


def min_jerk(z0: float, z1: float, n: int) -> np.ndarray:
    """z(tau) = z0 + (z1-z0)(10 tau^3 - 15 tau^4 + 6 tau^5), n samples including both ends."""
    if n <= 1:
        return np.array([z1], float)
    tau = np.linspace(0.0, 1.0, n)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return z0 + (z1 - z0) * s


def hybrid_path(
    p_h: np.ndarray,
    k_g: int,
    k_r: int,
    block_xyz: np.ndarray,
    target_xyz: np.ndarray,
    z_top: float,
    block_half: float,
    fps: float,
    hover_z: float,
    carry_z: float,
    grasp_z_offset: float,
    release_clearance: float,
    descend_s: float,
    close_hold_s: float,
    lift_s: float,
    lower_s: float,
    open_hold_s: float,
    degenerate_dist: float,
    carry_from_hand: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Robot tcp path and gripper command.

    p_h (n, 3) hand pinch in metres. k_g, k_r indices into that path.
    returns p_r (n, 3), grip (n,).
    """
    p_h = np.asarray(p_h, float)
    n = len(p_h)
    assert p_h.shape == (n, 3)
    k_g = int(np.clip(k_g, 0, n - 1))
    k_r = int(np.clip(k_r, k_g, n - 1))
    block = np.asarray(block_xyz, float).reshape(3)
    target = np.asarray(target_xyz, float).reshape(3)
    alpha, beta = similarity(p_h[k_g, :2], p_h[k_r, :2], block[:2], target[:2], float(degenerate_dist))
    q = apply_xy(p_h[:, :2], alpha, beta)
    e_g = block[:2] - q[k_g]
    e_r = target[:2] - q[k_r]
    span = max(k_r - k_g, 1)
    e = np.zeros((n, 2))
    for k in range(n):
        if k <= k_g:
            e[k] = e_g
        elif k >= k_r:
            e[k] = e_r
        else:
            w = (k - k_g) / span
            e[k] = (1.0 - w) * e_g + w * e_r
    xy = q + e

    z_grasp = float(block[2] + grasp_z_offset)
    z_release = float(z_top) + float(block_half) + float(release_clearance)
    z_carry = max(float(carry_z), z_release + float(hover_z))
    n_desc = max(1, int(round(descend_s * fps)))
    n_close = max(1, int(round(close_hold_s * fps)))
    n_lift = max(1, int(round(lift_s * fps)))
    n_lower = max(1, int(round(lower_s * fps)))
    n_open = max(1, int(round(open_hold_s * fps)))
    # Shrink windows that do not fit. The grasp and release indices stay put.
    n_desc = min(n_desc, k_g)
    room_after_grasp = max(k_r - k_g, 0)
    if n_close + n_lift + n_lower > room_after_grasp and room_after_grasp > 0:
        scale = room_after_grasp / (n_close + n_lift + n_lower)
        n_close = max(1, int(n_close * scale))
        n_lift = max(1, int(n_lift * scale))
        n_lower = max(1, room_after_grasp - n_close - n_lift)
    k_desc = k_g - n_desc
    k_lift0 = min(n - 1, k_g + n_close)
    k_lift1 = min(n - 1, k_lift0 + n_lift)
    k_low0 = max(k_lift1, k_r - n_lower)
    k_open1 = min(n - 1, k_r + n_open)

    z = np.full(n, z_grasp + hover_z)
    z[:k_desc] = z_grasp + hover_z
    if k_g > k_desc:
        z[k_desc:k_g + 1] = min_jerk(z_grasp + hover_z, z_grasp, k_g - k_desc + 1)
    z[k_g:k_lift0 + 1] = z_grasp
    if k_lift1 > k_lift0:
        z[k_lift0:k_lift1 + 1] = min_jerk(z_grasp, z_carry, k_lift1 - k_lift0 + 1)
    if carry_from_hand:
        hand_z = np.clip(p_h[:, 2] + z_grasp, z_carry, 0.25)
        z[k_lift1:k_low0 + 1] = hand_z[k_lift1:k_low0 + 1]
    else:
        z[k_lift1:k_low0 + 1] = z_carry
    if k_r > k_low0:
        z[k_low0:k_r + 1] = min_jerk(float(z[k_low0]), z_release, k_r - k_low0 + 1)
    z[k_r:k_open1 + 1] = z_release
    if n - 1 > k_open1:
        z[k_open1:] = min_jerk(z_release, z_release + hover_z, n - k_open1)[- (n - k_open1):]

    xy_out = xy.copy()
    xy_out[k_desc:k_lift1 + 1] = block[:2]
    xy_out[k_low0:] = target[:2]
    grip = np.zeros(n)
    if k_r > k_g:
        grip[k_g:k_r] = 1.0
    p = np.column_stack([xy_out, z])
    assert p.shape == (n, 3) and grip.shape == (n,)
    return p, grip
