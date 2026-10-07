"""Two-anchor retargeting (EXPLAINER §8). The worked example is copied from there."""

import inspect

import numpy as np
import pytest

from palm_prior.retarget.naive import naive_global
from palm_prior.retarget.two_anchor import (
    apply_xy,
    kappa,
    release_ramp,
    similarity,
    vertical,
)

# EXPLAINER §8 worked example, metres.
X_G = np.array([0.10, 0.05])
X_R = np.array([0.30, 0.15])
BLOCK = np.array([0.50, 0.10])
TARGET = np.array([0.60, -0.12])
DEGENERATE = 0.05


def test_worked_example_alpha_and_beta():
    alpha, beta = similarity(X_G, X_R, BLOCK, TARGET, DEGENERATE)
    assert alpha == pytest.approx(-0.04 - 1.08j, abs=1e-3)
    assert beta == pytest.approx(0.45 + 0.21j, abs=1e-3)
    landed = apply_xy(X_R, alpha, beta)
    np.testing.assert_allclose(landed, TARGET, atol=1e-3)


def test_anchors_are_exact():
    """Both correspondences hold to 1e-9, not merely the release one in the write-up."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        x_g = rng.uniform(-0.2, 0.2, size=2)
        x_r = x_g + rng.uniform(0.1, 0.4, size=2)  # comfortably above the 5 cm threshold
        block = rng.uniform(0.4, 0.7, size=2)
        target = rng.uniform(0.4, 0.7, size=2)
        alpha, beta = similarity(x_g, x_r, block, target, DEGENERATE)
        np.testing.assert_allclose(apply_xy(x_g, alpha, beta), block, atol=1e-9)
        np.testing.assert_allclose(apply_xy(x_r, alpha, beta), target, atol=1e-9)


def test_degenerate_anchors_give_alpha_one():
    """|x_r - x_g| = 2.2 cm, under the 5 cm threshold, so the anchors are one point."""
    x_g = np.array([0.10, 0.10])
    x_r = np.array([0.12, 0.11])
    block = np.array([0.50, 0.20])
    alpha, beta = similarity(x_g, x_r, block, np.array([0.90, -0.20]), DEGENERATE)
    assert alpha == 1 + 0j
    np.testing.assert_allclose(apply_xy(x_g, alpha, beta), block, atol=1e-12)


def test_kappa_is_clipped_to_the_configured_range():
    lo, hi = 0.5, 1.5
    assert kappa(0.1 + 0j, lo, hi) == pytest.approx(lo)
    assert kappa(3.0 + 0j, lo, hi) == pytest.approx(hi)
    # the worked example sits inside the window, so clipping must leave it alone
    alpha, _ = similarity(X_G, X_R, BLOCK, TARGET, DEGENERATE)
    assert kappa(alpha, lo, hi) == pytest.approx(abs(alpha))


def test_target_height_is_added_only_after_release():
    """s goes from 0 at the grasp to 1 at the release, and stays 1 afterwards."""
    z_h = np.array([0.03, 0.03, 0.12, 0.12, 0.05])
    k_close, k_open = 1, 3
    s = release_ramp(np.arange(len(z_h)), k_close, k_open)
    assert s[0] == 0.0 and s[k_close] == 0.0
    assert s[k_open] == 1.0 and s[-1] == 1.0

    h_tgt = 0.015
    with_target = vertical(z_h, z_h[k_close], 1.0, h_tgt, s)
    without = vertical(z_h, z_h[k_close], 1.0, 0.0, s)
    np.testing.assert_allclose(with_target[: k_close + 1], without[: k_close + 1])
    assert with_target[-1] - without[-1] == pytest.approx(h_tgt)


def test_plan_grasp_lands_on_the_block_and_release_on_the_target():
    from palm_prior.retarget.build_plan import build_plan
    from palm_prior.utils import load_config

    cfg = load_config()
    k = np.arange(10)
    t = k * 0.05
    p = np.zeros((10, 3))
    p[:, 0] = np.linspace(X_G[0], X_R[0], 10)
    p[:, 1] = np.linspace(X_G[1], X_R[1], 10)
    p[:, 2] = 0.06
    p[2, :2] = X_G
    p[7, :2] = X_R
    g = np.zeros(10)
    g[2:7] = 1.0
    plan = build_plan(
        {"t_frame": t, "p_ee": p, "g": g},
        BLOCK, TARGET, 0.015, None, None, cfg, ee_start_W=None,
    )
    assert plan.k_close == 2 and plan.k_open == 7
    np.testing.assert_allclose(plan.p_star[2, :2], BLOCK, atol=1e-9)
    np.testing.assert_allclose(plan.p_star[7, :2], TARGET, atol=1e-9)
    assert plan.p_star[2, 2] == pytest.approx(float(cfg.objects.sim_block_size) / 2)
    assert plan.grip[2] == 1.0 and plan.grip[7] == 0.0 and plan.grip[0] == 0.0


def test_naive_global_does_not_take_object_positions():
    """The baseline is one fixed transform. Block and target xy are not arguments."""
    names = set(inspect.signature(naive_global).parameters)
    assert names == {"xy", "alpha", "beta"}
    xy = np.array([[0.10, 0.05], [0.30, 0.15]])
    out = naive_global(xy, 1 + 0j, 0.2 + 0.1j)
    np.testing.assert_allclose(out, xy + np.array([0.2, 0.1]))
