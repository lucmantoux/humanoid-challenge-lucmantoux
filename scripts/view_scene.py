"""Phase 4a: open the scene in the interactive MuJoCo viewer.

macOS needs mjpython for the viewer's main-thread event loop; Windows and Linux use the
ordinary interpreter. Nothing else in this project needs the viewer, so nothing else
cares which interpreter you used.

    macOS:            uv run mjpython scripts/view_scene.py
    Windows / Linux:  uv run python scripts/view_scene.py

Site group 4 is enabled on start so the red tcp sphere is visible. Press Tab for the
menus; the block can be dragged with ctrl + right-drag.
"""

from __future__ import annotations

import sys

import mujoco
import mujoco.viewer

import _bootstrap  # noqa: F401  puts src/ on sys.path

from palm_prior.sim.env import TCP_SITE_GROUP, Env
from palm_prior.utils import load_config, set_seed


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    goal = str(cfg.get("goal", "plate"))
    env = Env(goal, int(cfg.seed), cfg)
    env.reset()
    print(f"goal={goal} block={env.layout.block_xy.round(3)} target={env.layout.target_xy.round(3)}")
    print(f"tcp at {env.p_ee_W().round(4)} — the red sphere must sit between the fingertips")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[TCP_SITE_GROUP] = 1
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == "__main__":
    main(sys.argv[1:])
