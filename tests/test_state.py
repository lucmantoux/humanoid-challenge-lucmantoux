"""Tests for the shared state of EXPLAINER §2, including its worked example."""

import numpy as np
import pytest
import torch

from palm_prior.state import (
    ACTION_DIM,
    STATE_DIM,
    attached,
    build_action,
    build_state,
    decode_state,
    next_state,
)

BLOCK = 0.04  # the worked example uses a 4 cm block

# EXPLAINER §2: hand 1 cm above a 4 cm block, closed, target 15 cm +x / 5 cm +y away, plate goal.
S0 = np.array([0.0, 0.0, 0.01, 0.15, 0.05, 0.02, 0.015, 1.0])
A0 = np.array([0.0, 0.0, 0.02, 1.0])
DP_LIFT = np.array([0.0, 0.0, 0.019])
S1 = np.array([0.0, 0.0, 0.011, 0.15, 0.05, 0.039, 0.015, 1.0])


def test_build_state_matches_worked_example():
    p_obj = np.array([0.30, 0.15, 0.02])
    p_ee = p_obj + np.array([0.0, 0.0, 0.01])
    p_tgt_xy = p_obj[:2] + np.array([0.15, 0.05])
    s = build_state(p_ee, p_obj, p_tgt_xy, h_tgt=0.015, g=1.0)
    assert s.shape == (STATE_DIM,)
    np.testing.assert_allclose(s, S0, atol=1e-12)


def test_next_state_worked_example_lift():
    np.testing.assert_allclose(next_state(S0, A0, DP_LIFT), S1, atol=1e-12)


def test_next_state_worked_example_missed_grasp():
    """With no block motion the hand/block gap grows from 1 cm to 3 cm."""
    s1 = next_state(S0, A0, np.zeros(3))
    np.testing.assert_allclose(s1[2], 0.03, atol=1e-12)
    np.testing.assert_allclose(s1[3:5], S0[3:5], atol=1e-12)
    np.testing.assert_allclose(s1[5], S0[5], atol=1e-12)


def test_translation_invariance_in_xy():
    rng = np.random.default_rng(0)
    p_ee = rng.normal(size=3)
    p_obj = rng.normal(size=3)
    p_tgt_xy = rng.normal(size=2)
    s = build_state(p_ee, p_obj, p_tgt_xy, 0.015, 0.0)
    c = rng.normal(size=2)
    s_shift = build_state(
        p_ee + np.r_[c, 0.0], p_obj + np.r_[c, 0.0], p_tgt_xy + c, 0.015, 0.0
    )
    np.testing.assert_allclose(s, s_shift, atol=1e-12)


def test_build_action_and_decode():
    a = build_action(np.array([0.0, 0.0, 0.02]), 1.0)
    assert a.shape == (ACTION_DIM,)
    np.testing.assert_allclose(a, A0, atol=1e-12)
    parts = decode_state(S0)
    np.testing.assert_allclose(parts.d_eo, [0.0, 0.0, 0.01], atol=1e-12)
    np.testing.assert_allclose(parts.d_to, [0.15, 0.05], atol=1e-12)
    assert parts.z_obj == pytest.approx(0.02)
    assert parts.h_tgt == pytest.approx(0.015)
    assert parts.g == pytest.approx(1.0)


def test_batched_shapes_and_consistency():
    rng = np.random.default_rng(1)
    shape = (4, 3)
    p_ee = rng.normal(size=shape + (3,))
    p_obj = rng.normal(size=shape + (3,))
    p_tgt_xy = rng.normal(size=shape + (2,))
    s = build_state(p_ee, p_obj, p_tgt_xy, 0.08, rng.integers(0, 2, shape).astype(float))
    assert s.shape == shape + (STATE_DIM,)
    a = build_action(rng.normal(size=shape + (3,)), 1.0)
    dp = rng.normal(size=shape + (3,))
    s_next = next_state(s, a, dp)
    assert s_next.shape == s.shape
    # a batched update equals the per-element update
    np.testing.assert_allclose(s_next[2, 1], next_state(s[2, 1], a[2, 1], dp[2, 1]), atol=1e-12)


def test_torch_matches_numpy():
    s = torch.tensor(S0, dtype=torch.float64)
    a = torch.tensor(A0, dtype=torch.float64)
    dp = torch.tensor(DP_LIFT, dtype=torch.float64)
    s_next = next_state(s, a, dp)
    assert isinstance(s_next, torch.Tensor)
    np.testing.assert_allclose(s_next.numpy(), S1, atol=1e-12)

    p_obj = torch.tensor([0.30, 0.15, 0.02], dtype=torch.float64)
    p_ee = p_obj + torch.tensor([0.0, 0.0, 0.01], dtype=torch.float64)
    p_tgt_xy = p_obj[:2] + torch.tensor([0.15, 0.05], dtype=torch.float64)
    s_built = build_state(p_ee, p_obj, p_tgt_xy, 0.015, 1.0)
    assert isinstance(s_built, torch.Tensor)
    np.testing.assert_allclose(s_built.numpy(), S0, atol=1e-12)


def test_attach_rule():
    """attached = closed and the block lifted more than 1 cm above its resting height."""
    z = np.array([BLOCK / 2, BLOCK / 2 + 0.005, BLOCK / 2 + 0.05, BLOCK / 2 + 0.05])
    g = np.array([1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(attached(g, z, BLOCK, 0.01), [0.0, 0.0, 1.0, 0.0])
    t = attached(torch.tensor(g), torch.tensor(z), BLOCK, 0.01)
    np.testing.assert_allclose(t.numpy(), [0.0, 0.0, 1.0, 0.0])
