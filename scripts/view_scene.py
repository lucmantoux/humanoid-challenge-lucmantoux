"""Phase 4a: open the scene in the interactive MuJoCo viewer.

macOS needs mjpython for the viewer's main-thread event loop; Windows and Linux use the
ordinary interpreter. Nothing else in this project needs the viewer, so nothing else
cares which interpreter you used.

    macOS:            uv run mjpython scripts/view_scene.py
    Windows / Linux:  uv run python scripts/view_scene.py

It also writes results/tcp_check.png, a close-up of the gripper with the tcp marker on,
so the check survives without a desktop session.

Site group 4 is enabled on start so the red tcp sphere is visible. Press Tab for the
menus; the block can be dragged with ctrl + right-drag.
"""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import mujoco
import mujoco.viewer

import _bootstrap  # noqa: F401  puts src/ on sys.path

from palm_prior.sim.env import TCP_SITE_GROUP, Env
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip


def tcp_closeup(env, width: int = 800, height: int = 600):
    """Render the gripper from close range with the tcp marker on. (H, W, 3) uint8.

    A free camera rather than `agentview`: the whole point is to see a 6 mm sphere, and
    from the agentview distance it is three pixels across.
    """
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        options = mujoco.MjvOption()
        options.sitegroup[TCP_SITE_GROUP] = 1
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, camera)
        camera.lookat[:] = env.p_ee_W() + [0.0, 0.0, -0.02]
        camera.distance, camera.azimuth, camera.elevation = 0.28, 110.0, -12.0
        renderer.update_scene(env.data, camera=camera, scene_option=options)
        return renderer.render()
    finally:
        renderer.close()


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    goal = str(cfg.get("goal", "plate"))
    out_png = resolve(cfg.paths.results) / "tcp_check.png"

    with Timer("view_scene"):
        env = Env(goal, int(cfg.seed), cfg)
        env.reset()
        print(f"goal={goal} block={env.layout.block_xy.round(3)} target={env.layout.target_xy.round(3)}")
        print(f"tcp at {env.p_ee_W().round(4)} — the red sphere must sit between the fingertips")
        if not should_skip(out_png, bool(cfg.get("overwrite", False))):
            ensure_dir(out_png.parent)
            imageio.imwrite(out_png, tcp_closeup(env))
            print(f"wrote {out_png}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[TCP_SITE_GROUP] = 1
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == "__main__":
    main(sys.argv[1:])
