"""Damped least-squares inverse kinematics for the Panda (EXPLAINER §9, Module 7).

The planner only ever asks for a tcp position. Orientation is not a decision variable:
the gripper is held pointing straight down at a fixed yaw, which is what the human clips
look like and what keeps the 8-D state honest about being position-only.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# The hand frame when the gripper points straight down and the fingers slide along world
# x. This is exactly the Menagerie "home" orientation, so the arm starts with zero
# orientation error and never has to unwind a wrist.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Rotation vector taking R_cur to R_des, in world axes. (3,3), (3,3) -> (3,)

    Its direction is the axis to turn about and its length the angle in radians, so it
    drops into the same 6-vector as the position error without any scaling trick.
    """
    assert R_cur.shape == (3, 3) and R_des.shape == (3, 3), (R_cur.shape, R_des.shape)
    return Rotation.from_matrix(R_des @ R_cur.T).as_rotvec()


def dls_step(
    J: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    q_home: np.ndarray,
    damping: float,
    nullspace_gain: float,
) -> np.ndarray:
    """One damped least-squares step with a null-space posture term (EXPLAINER §9).

    dq = J^T (J J^T + lam^2 I)^-1 e  +  (I - J^+ J) k_n (q_home - q)

    J (m, n) task Jacobian, e (m,) task error, q and q_home (n,). Returns dq (n,).

    The damping is what makes this safe near a singularity: with a well-conditioned J it
    changes the answer by a fraction of a percent, and with a nearly singular one it
    turns a violent joint motion into a small one.
    """
    m, n = J.shape
    assert e.shape == (m,), (e.shape, J.shape)
    assert q.shape == (n,) and q_home.shape == (n,), (q.shape, q_home.shape, J.shape)

    damped = J @ J.T + damping**2 * np.eye(m)
    pinv = J.T @ np.linalg.inv(damped)  # the damped pseudo-inverse, (n, m)
    dq = pinv @ e
    if nullspace_gain != 0.0:
        dq = dq + (np.eye(n) - pinv @ J) @ (nullspace_gain * (q_home - q))
    return dq


def tcp_pose(model, data, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Current tcp position (3,) and rotation (3,3) in the world frame W."""
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def track(model, data, scene, p_des_W: np.ndarray, cfg) -> None:
    """Drive the 7 arm actuators one DLS step towards p_des_W, gripper pointing down.

    p_des_W (3,) in the world frame. Writes the new joint targets into data.ctrl; the
    caller runs the physics. Mutating ctrl rather than returning it keeps the control
    loop in env.step readable.
    """
    assert p_des_W.shape == (3,), p_des_W.shape
    p_cur, R_cur = tcp_pose(model, data, scene.tcp)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, scene.tcp)
    J = np.vstack([jacp[:, scene.arm_dof], float(cfg.sim.ik.rot_weight) * jacr[:, scene.arm_dof]])
    e = np.concatenate(
        [p_des_W - p_cur, float(cfg.sim.ik.rot_weight) * orientation_error(R_cur, R_DOWN)]
    )

    q = data.qpos[scene.arm_qpos]
    dq = dls_step(J, e, q, scene.q_home, float(cfg.sim.ik.damping), float(cfg.sim.ik.nullspace_gain))

    lo = model.jnt_range[model.actuator_trnid[scene.arm, 0], 0]
    hi = model.jnt_range[model.actuator_trnid[scene.arm, 0], 1]
    data.ctrl[scene.arm] = np.clip(q + dq, lo, hi)
