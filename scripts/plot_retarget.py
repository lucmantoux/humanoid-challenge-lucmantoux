"""Top view of one human path and the robot path it retargets to.

The robot path is drawn in the world frame. Its grasp sits on the block and its
release sits on the target; the human path is drawn in the table frame, same shape.

Usage:
    uv run python scripts/plot_retarget.py
    uv run python scripts/plot_retarget.py clip=1 seed=0 goal=plate
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402,F401

from palm_prior.retarget.build_plan import build_plan  # noqa: E402
from palm_prior.sim.env import sample_layout  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed  # noqa: E402


def _passed_successes(cfg) -> list[int]:
    report = resolve(cfg.paths.results) / "extraction_report.csv"
    ids = []
    with report.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            cols = dict(zip(header, line.strip().split(",")))
            if cols["type"] == "success" and cols["pass"] == "1":
                ids.append(int(cols["clip"]))
    return ids


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    clip_id = int(cfg.get("clip", 0) or _passed_successes(cfg)[0])
    goal = str(cfg.get("goal", "plate"))
    path = resolve(cfg.paths.human_dir) / f"clip_{clip_id:03d}.npz"
    clip = np.load(path)
    layout = sample_layout(goal, np.random.default_rng(int(cfg.seed)), cfg)
    plan = build_plan(
        clip, layout.block_xy, layout.target_xy, float(cfg.goals[goal]),
        None, None, cfg, ee_start_W=None,
    )
    human = clip["p_ee"]
    with Timer("plot_retarget"):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(human[:, 0], human[:, 1], color="0.75", lw=1, label="hand (table frame)")
        ax.plot(plan.p_star[:, 0], plan.p_star[:, 1], color="C0", lw=1, alpha=0.35, label="robot, whole clip")
        sl = slice(plan.k_close, plan.k_open + 1)
        ax.plot(plan.p_star[sl, 0], plan.p_star[sl, 1], color="C0", lw=2, label="robot, grasp to release")
        ax.scatter(*plan.p_star[plan.k_close, :2], c="C2", s=70, zorder=3, label="block = grasp")
        ax.scatter(*plan.p_star[plan.k_open, :2], c="C3", s=70, marker="s", zorder=3, label="target = release")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"clip {clip_id} → {goal}, seed {int(cfg.seed)}")
        ax.legend(loc="best")
        out = ensure_dir(resolve(cfg.paths.results)) / "retarget_top.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
    err_g = np.linalg.norm(plan.p_star[plan.k_close, :2] - layout.block_xy)
    err_r = np.linalg.norm(plan.p_star[plan.k_open, :2] - layout.target_xy)
    print(f"grasp error {err_g * 100:.3f} cm, release error {err_r * 100:.3f} cm")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
