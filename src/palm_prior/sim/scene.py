"""Assembling the MuJoCo model (EXPLAINER Module 7).

assets/scene.xml holds the stage. The Panda comes from MuJoCo Menagerie, whose path is
only known at run time, and which defines no sites at all — so the arm is attached and
the `tcp` site is added here, through MjSpec, rather than by editing anyone's XML.
"""

from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np
from robot_descriptions import panda_mj_description

from palm_prior.utils import REPO_ROOT

SCENE_XML = REPO_ROOT / "assets" / "scene.xml"

# Menagerie exposes actuator1..actuator8; 1-7 are joint position servos, 8 drives both
# fingers through the `split` tendon with ctrlrange 0..255, 255 open.
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
GOAL_BODIES = ("plate", "pad", "box")


class Scene(NamedTuple):
    """A compiled model plus the ids everything else needs.

    tcp         site id of the point between the fingertips
    block_body  body id of the free-floating block
    block_qpos  index of the block free joint in qpos (7 numbers: xyz + quat)
    arm         (7,) actuator ids for the joints, in order
    arm_qpos    (7,) qpos indices of the same joints
    arm_dof     (7,) velocity-space indices, i.e. the Jacobian columns that the arm owns
    gripper     actuator id of the fingers
    q_home      (7,) the Menagerie "home" posture, used as the IK null-space target
    """

    model: mujoco.MjModel
    tcp: int
    block_body: int
    block_geom: int
    block_qpos: int
    arm: np.ndarray
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    gripper: int
    q_home: np.ndarray


def build_model(cfg) -> mujoco.MjModel:
    """Compile the stage + Panda + tcp site, with every size taken from cfg."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    panda = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)

    hand = panda.body("hand")
    hand.add_site(
        name="tcp",
        pos=[0.0, 0.0, float(cfg.sim.tcp_offset)],
        size=[0.006, 0.006, 0.006],
        group=4,
        rgba=[1.0, 0.2, 0.2, 1.0],
    )
    # prefix="" keeps the Menagerie names (actuator1.., hand, home) exactly as documented.
    spec.attach(panda, prefix="", frame=spec.worldbody.add_frame(name="panda_mount"))

    b = float(cfg.objects.block_size)
    spec.geom("block").size = [b / 2, b / 2, b / 2]
    spec.geom("block").mass = float(cfg.objects.block_mass)
    spec.geom("block").friction[0] = float(cfg.objects.block_friction)
    spec.body("block").pos = [0.5, 0.0, b / 2]

    plate_h = float(cfg.objects.plate_height)
    spec.geom("plate").size = [float(cfg.objects.plate_radius), plate_h / 2, 0.0]
    spec.geom("plate").pos = [0.0, 0.0, plate_h / 2]

    pad = np.array(cfg.objects.pad_size, float)
    spec.geom("pad").size = list(pad / 2)
    spec.geom("pad").pos = [0.0, 0.0, pad[2] / 2]

    box = float(cfg.objects.box_size)
    spec.geom("box").size = [box / 2, box / 2, box / 2]
    spec.geom("box").pos = [0.0, 0.0, box / 2]

    return spec.compile()


def build_scene(cfg) -> Scene:
    """Compile the model and look up every id and index used downstream."""
    model = build_model(cfg)
    name2id = mujoco.mj_name2id

    tcp = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert tcp >= 0, "the tcp site was not attached"
    block_body = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_joint = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
    arm = np.array([name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
    gripper = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    assert (arm >= 0).all() and gripper >= 0, "Menagerie actuator names have changed"

    # The block's free joint sits in front of the arm in qpos, so the arm indices have to
    # be looked up rather than assumed to start at 0.
    arm_joints = np.array(
        [name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    )
    assert (arm_joints >= 0).all(), "Menagerie joint names have changed"
    arm_qpos = model.jnt_qposadr[arm_joints]
    arm_dof = model.jnt_dofadr[arm_joints]

    key = name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key >= 0, "the Menagerie home keyframe is missing"

    return Scene(
        model=model,
        tcp=tcp,
        block_body=block_body,
        block_geom=name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block"),
        block_qpos=int(model.jnt_qposadr[block_joint]),
        arm=arm,
        arm_qpos=arm_qpos,
        arm_dof=arm_dof,
        gripper=gripper,
        q_home=model.key_qpos[key][arm_qpos].copy(),
    )


def goal_body_id(model: mujoco.MjModel, goal: str) -> int:
    """Body id of the object a goal refers to. goal is one of plate, pad, box."""
    assert goal in GOAL_BODIES, f"unknown goal {goal!r}, expected one of {GOAL_BODIES}"
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, goal)
