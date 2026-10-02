"""Phase 0 smoke test: load the Menagerie Panda, step it, render one frame.

The Panda comes from the `robot_descriptions` package, which git-clones
MuJoCo Menagerie into ~/.cache/robot_descriptions on first use (needs network once).

Usage:
    uv run python scripts/smoke_mujoco.py
    uv run python scripts/smoke_mujoco.py seed=1 overwrite=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from robot_descriptions import panda_mj_description

from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_png = resolve(cfg.paths.results) / "smoke.png"
    ensure_dir(out_png.parent)
    if should_skip(out_png, bool(cfg.get("overwrite", False))):
        return

    scene_xml = Path(panda_mj_description.MJCF_PATH).parent / "scene.xml"
    print(f"menagerie scene: {scene_xml}")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print(f"actuators ({model.nu}): {actuators}")
    print(f"gripper actuator ctrlrange: {model.actuator_ctrlrange[-1]}")

    # Hold the "home" keyframe: arm actuators track their home angles, gripper open.
    key_home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_home)
    data.ctrl[:7] = model.key_qpos[key_home, :7]
    data.ctrl[7] = model.actuator_ctrlrange[7, 1]  # 255 = open
    mujoco.mj_forward(model, data)

    n_steps = int(cfg.smoke.n_steps)
    with Timer("step") as t:
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
    print(f"{n_steps / t.seconds:.0f} physics steps/s at timestep {model.opt.timestep} s")
    print(f"simulated {n_steps * model.opt.timestep:.2f} s; hand height {data.body('hand').xpos[2]:.3f} m")

    with mujoco.Renderer(model, height=int(cfg.smoke.render_h), width=int(cfg.smoke.render_w)) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
    imageio.imwrite(out_png, np.asarray(pixels))
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main(sys.argv[1:])
