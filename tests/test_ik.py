"""IK tests (EXPLAINER §9, Module 7)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from palm_prior.sim.env import Env, sample_layout
from palm_prior.sim.ik import R_DOWN, dls_step, orientation_error
from palm_prior.sim.scene import build_scene
from palm_prior.utils import load_config

CFG = load_config()
DAMPING = float(CFG.sim.ik.damping)


@pytest.fixture(scope="module")
def scene():
    return build_scene(CFG)


def test_dls_reproduces_the_worked_example(): 
    """EXPLAINER §9: with J = 0.5 and e = 0.02 the damping changes almost nothing."""
    dq = dls_step(
        np.array([[0.5]]), np.array([0.02]), np.zeros(1), np.zeros(1), DAMPING, 0.0
    )
    assert dq[0] == pytest.approx(0.0396, abs=1e-4)
    assert dq[0] < 0.040  # the undamped answer, which it should sit just under


def test_damping_tames_a_near_singular_jacobian():
    """Same EXPLAINER §9 example at J = 0.01: 2.0 rad undamped, 0.077 rad damped."""
    dq = dls_step(
        np.array([[0.01]]), np.array([0.02]), np.zeros(1), np.zeros(1), DAMPING, 0.0
    )
    assert dq[0] == pytest.approx(0.077, abs=1e-3)
    assert dq[0] < 0.1  # the undamped answer is 2.0 rad, a violent motion


def test_nullspace_term_does_not_move_the_end_effector():
    """The posture term must live in the null space of J, or it fights the task."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 7))
    q = rng.normal(size=7)
    q_home = rng.normal(size=7)
    task = dls_step(J, np.zeros(6), q, q_home, DAMPING, 0.0)
    both = dls_step(J, np.zeros(6), q, q_home, DAMPING, 0.1)
    np.testing.assert_allclose(task, 0.0, atol=1e-12)
    assert np.linalg.norm(both) > 1e-3, "the posture term did nothing at all"
    # (I - J^+ J) with a *damped* inverse is only an approximate projector, so a little
    # task motion leaks through. It must be a tiny fraction of what the same posture pull
    # would do if it were applied unprojected.
    unprojected = 0.1 * (q_home - q)
    assert np.linalg.norm(J @ both) < 0.01 * np.linalg.norm(J @ unprojected)


def test_nullspace_term_pulls_towards_the_home_posture():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(6, 7))
    q = rng.normal(size=7)
    q_home = rng.normal(size=7)
    dq = dls_step(J, np.zeros(6), q, q_home, DAMPING, 0.1)
    assert np.linalg.norm(q + dq - q_home) < np.linalg.norm(q - q_home)


def test_orientation_error_is_a_rotation_vector():
    assert np.allclose(orientation_error(R_DOWN, R_DOWN), 0.0)
    turn = Rotation.from_rotvec([0.0, 0.0, 0.2])
    np.testing.assert_allclose(
        orientation_error(R_DOWN, turn.as_matrix() @ R_DOWN), [0.0, 0.0, 0.2], atol=1e-12
    )


def test_r_down_is_a_rotation_with_the_gripper_pointing_down():
    assert np.linalg.det(R_DOWN) == pytest.approx(1.0)
    np.testing.assert_allclose(R_DOWN @ R_DOWN.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R_DOWN[:, 2], [0.0, 0.0, -1.0])


def test_tracker_reaches_five_random_targets(scene):
    """EXPLAINER Module 7 acceptance: under 5 mm within 30 control steps."""
    env = Env("plate", 0, CFG, scene=scene)
    rng = np.random.default_rng(7)
    lay = CFG.sim.layout
    for trial in range(5):
        env.reset()
        target = np.array(
            [
                rng.uniform(*lay.x_range),
                rng.uniform(*lay.y_range),
                rng.uniform(0.05, 0.30),
            ]
        )
        for _ in range(30):
            env.step(np.concatenate([target - env.p_ee_W(), [0.0]]))
        error = np.linalg.norm(env.p_ee_W() - target)
        assert error < 0.005, (trial, target, error)
    env.close()


def test_tracker_keeps_the_gripper_pointing_down(scene):
    """Orientation is not a decision variable: it must not drift while reaching."""
    env = Env("plate", 1, CFG, scene=scene)
    env.reset()
    target = np.array([float(CFG.sim.layout.x_range[1]), float(CFG.sim.layout.y_range[0]), 0.25])
    for _ in range(30):
        env.step(np.concatenate([target - env.p_ee_W(), [0.0]]))
    R_cur = env.data.site_xmat[scene.tcp].reshape(3, 3)
    tilt = np.degrees(np.linalg.norm(orientation_error(R_cur, R_DOWN)))
    assert tilt < 2.0, tilt
    env.close()


def test_action_displacement_is_clipped(scene):
    """sim.max_step bounds how far one 10 Hz action may ask the tcp to travel."""
    env = Env("plate", 2, CFG, scene=scene)
    env.reset()
    start = env.p_ee_W()
    env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    moved = np.linalg.norm(env.p_ee_W() - start)
    assert moved <= float(CFG.sim.max_step) + 1e-3, moved
    env.close()


def test_layouts_are_separated_and_inside_the_workspace():
    rng = np.random.default_rng(3)
    lay = CFG.sim.layout
    for _ in range(200):
        layout = sample_layout("pad", rng, CFG)
        for xy in (layout.block_xy, layout.target_xy):
            assert float(lay.x_range[0]) <= xy[0] <= float(lay.x_range[1])
            assert float(lay.y_range[0]) <= xy[1] <= float(lay.y_range[1])
        gap = np.linalg.norm(layout.block_xy - layout.target_xy)
        assert gap >= float(lay.min_separation), gap


def test_layout_sampling_is_reproducible_from_the_seed():
    a = sample_layout("box", np.random.default_rng(11), CFG)
    b = sample_layout("box", np.random.default_rng(11), CFG)
    np.testing.assert_array_equal(a.block_xy, b.block_xy)
    np.testing.assert_array_equal(a.target_xy, b.target_xy)
