"""The pick-and-place environment (EXPLAINER Module 7).

Everything outside this file sees only the 8-D state of EXPLAINER §2 and the 4-D action,
which is what lets a human clip and a robot episode be the same kind of thing.
"""

from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np

from palm_prior.sim import ik
from palm_prior.sim.scene import GOAL_BODIES, Scene, build_scene, goal_body_id
from palm_prior.state import build_state

# Where the two goal objects an episode does not use are parked: far enough that the arm
# can never touch them, spread out so they do not stack on each other.
PARKED_X = 3.0

# The tcp site is authored in group 4, which MuJoCo hides by default.
TCP_SITE_GROUP = 4


class Layout(NamedTuple):
    """One sampled episode: block start and target, both in the world frame W."""

    block_xy: np.ndarray  # (2,)
    target_xy: np.ndarray  # (2,)
    goal: str


def sample_layout(goal: str, rng: np.random.Generator, cfg) -> Layout:
    """Block and target inside the arm's reach, at least min_separation apart.

    Rejection sampling: the box is small enough that this lands in a handful of tries,
    and rejection keeps the distribution uniform, which a nudge-until-valid loop would not.
    """
    lay = cfg.sim.layout
    x_lo, x_hi = float(lay.x_range[0]), float(lay.x_range[1])
    y_lo, y_hi = float(lay.y_range[0]), float(lay.y_range[1])
    sep = float(lay.min_separation)
    for _ in range(1000):
        block_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
        target_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
        if np.linalg.norm(block_xy - target_xy) >= sep:
            return Layout(block_xy=block_xy, target_xy=target_xy, goal=goal)
    raise RuntimeError(
        f"no layout with {sep} m separation fits in x{list(lay.x_range)} y{list(lay.y_range)}"
    )


class Env:
    """A Panda, a block and one goal object, stepped at 10 Hz.

    goal is one of plate, pad, box. The same Env instance is reset many times; the model
    is compiled once because compiling costs far more than a reset.
    """

    def __init__(self, goal: str, seed: int, cfg, scene: Scene | None = None):
        assert goal in GOAL_BODIES, f"unknown goal {goal!r}"
        self.cfg = cfg
        self.goal = goal
        self.rng = np.random.default_rng(seed)
        self.scene = build_scene(cfg) if scene is None else scene
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        self.h_tgt = float(cfg.goals[goal])
        self.block_size = float(cfg.objects.sim_block_size)
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.layout: Layout | None = None
        self._renderer: mujoco.Renderer | None = None
        self._p_ee_target = np.zeros(3)
        self._g = 0.0

    # ---------------------------------------------------------------- episode lifecycle

    def reset(self, layout: Layout | None = None) -> np.ndarray:
        """Place the objects and the arm, settle one step, return the 8-D state."""
        self.layout = sample_layout(self.goal, self.rng, self.cfg) if layout is None else layout
        assert self.layout.goal == self.goal, (self.layout.goal, self.goal)

        # Static bodies move by editing the model, which only takes effect at mj_forward.
        for i, name in enumerate(GOAL_BODIES):
            body = goal_body_id(self.model, name)
            if name == self.goal:
                self.model.body_pos[body] = [*self.layout.target_xy, 0.0]
            else:
                self.model.body_pos[body] = [PARKED_X, 0.5 * i, 0.0]

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.set_block_pose(np.array([*self.layout.block_xy, self.block_size / 2]))
        self.data.ctrl[self.scene.arm] = self.data.qpos[self.scene.arm_qpos]
        self.data.ctrl[self.scene.gripper] = float(self.cfg.sim.gripper_open_ctrl)
        self._g = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._p_ee_target = self.p_ee_W()
        return self.get_state()

    def step(self, a: np.ndarray) -> np.ndarray:
        """One 10 Hz control step. a = [dx, dy, dz, g_next] per EXPLAINER §2.

        The displacement is clipped to sim.max_step and added to the *commanded* tcp
        position, not the measured one, so tracking lag cannot silently shrink the action
        the world model was trained on.
        """
        a = np.asarray(a, float)
        assert a.shape == (4,), a.shape
        dp = a[:3]
        norm = np.linalg.norm(dp)
        max_step = float(self.cfg.sim.max_step)
        if norm > max_step:
            dp = dp * (max_step / norm)
        self._p_ee_target = self._p_ee_target + dp
        self._g = float(a[3])

        grip = self.cfg.sim.gripper_close_ctrl if self._g >= 0.5 else self.cfg.sim.gripper_open_ctrl
        self.data.ctrl[self.scene.gripper] = float(grip)

        substeps = int(self.cfg.sim.substeps_per_control)
        ik_every = int(self.cfg.sim.ik_every)
        for i in range(substeps):
            if i % ik_every == 0:
                ik.track(self.model, self.data, self.scene, self._p_ee_target, self.cfg)
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    # ------------------------------------------------------------------------ observing

    def p_ee_W(self) -> np.ndarray:
        """Measured tcp position in W. (3,)"""
        return self.data.site_xpos[self.scene.tcp].copy()

    def p_obj_W(self) -> np.ndarray:
        """Block centre in W. (3,)"""
        return self.data.xpos[self.scene.block_body].copy()

    def get_state(self) -> np.ndarray:
        """The shared 8-D state of EXPLAINER §2. (8,)"""
        assert self.layout is not None, "call reset() first"
        return build_state(
            p_ee=self.p_ee_W(),
            p_obj=self.p_obj_W(),
            p_tgt_xy=self.layout.target_xy,
            h_tgt=self.h_tgt,
            g=self._g,
        )

    def block_speed(self) -> float:
        """Magnitude of the block's linear velocity, m/s."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.scene.block_body, vel, False
        )
        return float(np.linalg.norm(vel[3:]))

    def success(self) -> bool:
        """EXPLAINER Module 7: on target in xy, at the right height, released, and still."""
        assert self.layout is not None, "call reset() first"
        s = self.cfg.sim.success
        p_obj = self.p_obj_W()
        on_target = np.linalg.norm(p_obj[:2] - self.layout.target_xy) < float(s.xy_radius)
        at_height = abs(p_obj[2] - (self.h_tgt + self.block_size / 2)) < float(s.z_tol)
        return bool(on_target and at_height and self._g < 0.5 and self.block_speed() < float(s.max_speed))

    def settle(self) -> None:
        """Hold still for success.settle_seconds so the speed check means something."""
        n = int(round(float(self.cfg.sim.success.settle_seconds) / self.model.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # -------------------------------------------------------------------------- editing

    def set_block_pose(self, p_W: np.ndarray, quat: np.ndarray | None = None) -> None:
        """Teleport the block (used by reset and by the E5 disturbance test).

        p_W (3,) centre position; quat (4,) w-first, identity if omitted. Velocity is
        zeroed, otherwise the block keeps whatever it was doing before the teleport.
        """
        p_W = np.asarray(p_W, float)
        assert p_W.shape == (3,), p_W.shape
        i = self.scene.block_qpos
        self.data.qpos[i : i + 3] = p_W
        self.data.qpos[i + 3 : i + 7] = [1.0, 0.0, 0.0, 0.0] if quat is None else quat
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self.scene.block_body]]
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def save_state(self) -> tuple:
        """Everything needed to rewind, including the IK's own commanded tcp position."""
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.act.copy(),
            self.data.ctrl.copy(),
            self._p_ee_target.copy(),
            self._g,
        )

    def restore_state(self, saved) -> None:
        qpos, qvel, act, ctrl, p_target, g = saved
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act[:] = act
        self.data.ctrl[:] = ctrl
        self._p_ee_target = p_target.copy()
        self._g = g
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------------ rendering

    def render(
        self, width: int, height: int, camera: str = "agentview", show_tcp: bool = False
    ) -> np.ndarray:
        """One RGB frame, (height, width, 3) uint8. The renderer is reused across calls.

        show_tcp turns on site group 4, which is where the tcp marker lives. It is off by
        default so result videos are not littered with a red dot.
        """
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        options = mujoco.MjvOption()
        options.sitegroup[TCP_SITE_GROUP] = 1 if show_tcp else 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=options)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
