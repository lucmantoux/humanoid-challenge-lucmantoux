"""Hybrid retargeting (FIX.md §5). two_anchor is not modified here."""

import numpy as np

from palm_prior.retarget.hybrid import hybrid_path, min_jerk
from palm_prior.retarget.two_anchor import similarity


def test_min_jerk_is_still_at_both_ends():
    z = min_jerk(0.12, 0.02, 19)
    assert abs(z[0] - 0.12) < 1e-12 and abs(z[-1] - 0.02) < 1e-12
    dt = 1.0 / 18.0
    assert abs(z[1] - z[0]) / dt < 1e-3 or abs(z[1] - z[0]) < 1e-3
    assert abs(z[-1] - z[-2]) < 1e-3


def test_endpoints_land_on_the_block_and_the_target():
    n = 40
    t = np.linspace(0, 1, n)
    p = np.column_stack([0.1 + 0.2 * t, 0.05 + 0.05 * np.sin(np.pi * t), 0.02 + 0.08 * np.sin(np.pi * t)])
    k_g, k_r = 8, 30
    block = np.array([0.50, 0.00, 0.02])
    target = np.array([0.65, 0.10, 0.02])
    path, grip = hybrid_path(
        p, k_g, k_r, block, target, z_top=0.015, block_half=0.02, fps=10.0,
        hover_z=0.10, carry_z=0.12, grasp_z_offset=0.0, release_clearance=0.005,
        descend_s=0.4, close_hold_s=0.2, lift_s=0.3, lower_s=0.4, open_hold_s=0.2,
        degenerate_dist=0.05,
    )
    assert np.linalg.norm(path[k_g, :2] - block[:2]) < 1e-3
    assert abs(path[k_g, 2] - 0.02) < 1e-3
    z_release = 0.015 + 0.02 + 0.005
    assert np.linalg.norm(path[k_r, :2] - target[:2]) < 1e-3
    assert abs(path[k_r, 2] - z_release) < 1e-3
    assert grip[k_g] == 1.0 and grip[k_r] == 0.0
    assert np.isfinite(path).all()


def test_two_anchor_formula_is_unchanged():
    alpha, beta = similarity(
        np.array([0.10, 0.05]), np.array([0.30, 0.15]),
        np.array([0.50, 0.10]), np.array([0.60, -0.12]), 0.05,
    )
    assert abs(alpha - (-0.04 - 1.08j)) < 1e-9
    assert abs(beta - (0.45 + 0.21j)) < 1e-9
