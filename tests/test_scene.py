"""Scene tests (EXPLAINER Module 7).

These pin down the two things that are easy to get silently wrong: where the `tcp` site
actually sits, and whether configs/default.yaml really reaches the compiled model.
"""

import mujoco
import numpy as np
import pytest

from palm_prior.sim.scene import ARM_ACTUATORS, GRIPPER_ACTUATOR, build_scene, goal_body_id
from palm_prior.utils import load_config

CFG = load_config()


@pytest.fixture(scope="module")
def scene():
    return build_scene(CFG)


@pytest.fixture(scope="module")
def at_home(scene):
    data = mujoco.MjData(scene.model)
    key = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(scene.model, data, key)
    mujoco.mj_forward(scene.model, data)
    return data


def _fingertip_geoms(model):
    """The geom on each finger that reaches furthest towards the workpiece."""
    tips = []
    for name in ("left_finger", "right_finger"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        assert geoms, f"no geoms on {name}"
        tips.append(max(geoms, key=lambda g: model.geom_pos[g][2]))
    return tips


def test_tcp_sits_between_the_fingertips(scene, at_home):
    """The whole 8-D state is measured at this point, so it must be the grasp point."""
    tcp = at_home.site_xpos[scene.tcp]
    left, right = (at_home.geom_xpos[g] for g in _fingertip_geoms(scene.model))

    midpoint = 0.5 * (left + right)
    assert np.linalg.norm(tcp - midpoint) < 0.005, (tcp, midpoint)
    # equidistant from both pads, i.e. on the axis the fingers close along
    assert abs(np.linalg.norm(tcp - left) - np.linalg.norm(tcp - right)) < 1e-6
    # and below the hand body, not inside the wrist
    hand = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    assert tcp[2] < at_home.xpos[hand][2] - 0.09


def test_the_gripper_opens_wide_enough_for_the_block(scene, at_home):
    """An open gripper must clear the block, or no grasp is ever possible."""
    left, right = (at_home.geom_xpos[g] for g in _fingertip_geoms(scene.model))
    opening = np.linalg.norm(left - right)
    assert opening > float(CFG.objects.block_size) + 0.005, opening


def test_menagerie_actuator_names_and_gripper_range(scene):
    model = scene.model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == [*ARM_ACTUATORS, GRIPPER_ACTUATOR]
    lo, hi = model.actuator_ctrlrange[scene.gripper]
    assert (lo, hi) == (float(CFG.sim.gripper_close_ctrl), float(CFG.sim.gripper_open_ctrl))


def test_object_sizes_come_from_the_config(scene):
    """configs/default.yaml is the authority; assets/scene.xml only sketches the stage."""
    model = scene.model
    half = float(CFG.objects.sim_block_size) / 2
    np.testing.assert_allclose(model.geom_size[scene.block_geom], [half, half, half])
    assert model.body_mass[scene.block_body] == pytest.approx(float(CFG.objects.block_mass))
    assert model.geom_friction[scene.block_geom][0] == pytest.approx(float(CFG.objects.block_friction))

    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate")
    assert model.geom_size[plate][0] == pytest.approx(float(CFG.objects.plate_radius))
    assert 2 * model.geom_size[plate][1] == pytest.approx(float(CFG.objects.plate_height))

    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    assert 2 * model.geom_size[box][0] == pytest.approx(float(CFG.objects.box_size))


def test_goal_objects_stand_at_their_target_heights(scene, at_home):
    """h_tgt in the state is the *top* of the goal object, so the geometry must agree."""
    model = scene.model
    for goal, geom_name in (("plate", "plate"), ("pad", "pad"), ("box", "box")):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        body_z = at_home.xpos[goal_body_id(model, goal)][2]
        axis = 1 if geom_name == "plate" else 2  # a cylinder's half-length is size[1]
        top = at_home.geom_xpos[g][2] + model.geom_size[g][axis] - body_z
        assert top == pytest.approx(float(CFG.goals[goal]), abs=1e-9), (goal, top)


def test_physics_options_match_the_config(scene):
    opt = scene.model.opt
    assert opt.timestep == pytest.approx(float(CFG.sim.timestep))
    assert opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert opt.impratio == pytest.approx(float(CFG.sim.impratio))


def test_a_control_step_is_a_tenth_of_a_second(scene):
    """The human clips, the robot episodes and the world model all assume 10 Hz."""
    dt = scene.model.opt.timestep * int(CFG.sim.substeps_per_control)
    assert dt == pytest.approx(1.0 / float(CFG.human.rate_hz))
