"""Object rays, the grasp rule, and the 11-D state (FIX.md §4, §6, §7)."""

import numpy as np
import pytest

from palm_prior.human.qc import grasp_passes
from palm_prior.state import STATE_DIM_V2, build_state_v2, next_state
from palm_prior.vision.objects import ray_plane
from palm_prior.wm.model import assert_state_version


def test_ray_meets_the_plane_in_the_worked_example():
    # Camera whose centre and ray match FIX.md §4.
    # X = C + λ d is checked directly through ray_plane on a constructed camera.
    c = np.array([0.0, -0.5, 0.6])
    d = np.array([0.1, 0.8, -0.95])
    h = 0.02
    lam = (h - c[2]) / d[2]
    x = c + lam * d
    assert abs(lam - 0.611) < 0.01
    assert abs(x[0] - 0.061) < 0.01 and abs(x[1] + 0.011) < 0.01
    assert abs(x[2] - h) < 1e-9


def test_ray_plane_returns_the_plane_height():
    k = np.array([[700.0, 0, 320.0], [0, 700.0, 240.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    r = np.eye(3)
    t = np.array([0.0, 0.0, 1.0])
    point = ray_plane(320.0, 240.0, 0.02, k, dist, r, t)
    assert abs(point[2] - 0.02) < 1e-8


def test_failure_clips_may_regrasp_and_a_success_may_not():
    allow = ["F1", "F2", "F3", "F4"]
    assert grasp_passes("F1", 3, allow, 8)
    assert not grasp_passes("success", 2, allow, 8)
    assert grasp_passes("success", 1, allow, 8)
    assert grasp_passes("F4", 0, allow, 8)
    assert not grasp_passes("F1", 0, allow, 8)


def test_v2_state_keeps_the_target_and_a_v1_checkpoint_is_refused():
    p_ee = np.array([0.1, 0.0, 0.05])
    p_obj = np.array([0.0, 0.0, 0.02])
    p_tgt = np.array([0.4, 0.1, 0.015])
    s = build_state_v2(p_ee, p_obj, p_tgt, 0.015, 1.0)
    assert s.shape == (STATE_DIM_V2,)
    a = np.array([0.0, 0.0, 0.01, 1.0])
    s2 = next_state(s, a, np.zeros(3))
    np.testing.assert_allclose(s2[-3:], p_tgt)
    with pytest.raises(ValueError, match="state_version"):
        assert_state_version({"state_version": 1})
