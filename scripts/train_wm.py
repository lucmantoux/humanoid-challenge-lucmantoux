"""Train one world model and save a checkpoint.

Usage:
    uv run python scripts/train_wm.py variant=pretrain_ft N=25 seed=0
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import _bootstrap  # noqa: F401

from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, set_torch_threads  # noqa: E402
from exp_e2 import _heldout_clips, _take, _train_variant  # noqa: E402


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    set_torch_threads()
    variant = str(cfg.get("variant", "pretrain_ft"))
    n = int(cfg.get("N", 25))
    seed = int(cfg.seed)
    human_z = np.load(resolve(cfg.paths.human_dir) / "transitions.npz")
    hold = _heldout_clips(human_z["clip_id"], human_z["is_failure"], int(cfg.human.n_heldout_clips), seed)
    train_mask = ~np.isin(human_z["clip_id"], list(hold))
    human = _take(human_z["s"], human_z["a"], human_z["dp_obj"], human_z["attach"], human_z["clip_id"], train_mask)
    robot_z = np.load(resolve(cfg.paths.robot_dir) / "episodes.npz")
    robot = (robot_z["s"], robot_z["a"], robot_z["dp_obj"], robot_z["attach"], robot_z["episode_id"])
    held_mask = np.isin(
        robot_z["episode_id"],
        np.arange(int(cfg.robot_data.heldout_seed_start), int(cfg.robot_data.heldout_seed_start) + int(cfg.robot_data.n_heldout_episodes)),
    )
    held = _take(*robot, held_mask)
    with Timer("train_wm"):
        model, stats, pos, auroc = _train_variant(cfg, variant, n, seed, human, robot, held)
    path = ensure_dir(resolve(cfg.paths.checkpoints)) / f"wm_{variant}_N{n}_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "stats": stats.save()}, path)
    print(f"held-out 10-step error {pos:.2f} cm, attach AUROC {auroc:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
