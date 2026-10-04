"""Phase 4a: render a grid of sampled layouts, to check them by eye.

What you are looking for: the block and the goal object well apart, both in front of the
arm, nothing intersecting anything, and the red tcp dot between the fingertips.

Usage:
    uv run python scripts/render_layouts.py
    uv run python scripts/render_layouts.py seed=3 overwrite=true
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _bootstrap  # noqa: E402,F401  puts src/ on sys.path

from palm_prior.sim.env import Env  # noqa: E402
from palm_prior.sim.scene import GOAL_BODIES, build_scene  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip  # noqa: E402

N_LAYOUTS = 6


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_png = resolve(cfg.paths.results) / "layouts.png"
    if should_skip(out_png, bool(cfg.get("overwrite", False))):
        return

    with Timer("render_layouts"):
        scene = build_scene(cfg)
        frames = []
        for i in range(N_LAYOUTS):
            goal = GOAL_BODIES[i % len(GOAL_BODIES)]
            env = Env(goal, int(cfg.seed) + i, cfg, scene=scene)
            env.reset()
            frames.append((goal, env.layout, env.render(640, 480, show_tcp=True)))
            env.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (goal, layout, frame) in zip(axes.flat, frames):
        ax.imshow(frame)
        gap = float(((layout.block_xy - layout.target_xy) ** 2).sum() ** 0.5)
        ax.set_title(
            f"{goal}  block ({layout.block_xy[0]:.2f}, {layout.block_xy[1]:.2f})  "
            f"target ({layout.target_xy[0]:.2f}, {layout.target_xy[1]:.2f})  gap {gap * 100:.0f} cm",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main(sys.argv[1:])
