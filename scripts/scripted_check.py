"""Phase 4b: a waypoint pick-and-place, 10 seeds per goal.

The waypoints are approach, descend, close, lift, carry, descend, open, retreat.
Actions are deltas of the *commanded* tcp position, so the command stops on the waypoint
instead of integrating past it and driving the block into the target.

Usage:
    uv run python scripts/scripted_check.py
    uv run python scripts/scripted_check.py seed=1 scripted.n_seeds=3
    uv run python scripts/scripted_check.py sweep=true
"""

from __future__ import annotations

import sys

import numpy as np
from omegaconf import OmegaConf

import _bootstrap  # noqa: F401  puts src/ on sys.path

from palm_prior.sim.env import Env  # noqa: E402
from palm_prior.sim.scene import GOAL_BODIES, build_scene  # noqa: E402
from palm_prior.utils import (  # noqa: E402
    Timer,
    ensure_dir,
    load_config,
    resolve,
    set_seed,
    should_skip,
    write_video,
)


def _toward(cmd: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    """A delta that moves cmd toward target by at most `limit` metres. (3,)"""
    err = target - cmd
    norm = float(np.linalg.norm(err))
    if norm <= limit:
        return err
    return err * (limit / norm)


def _snap(env, frames: list[np.ndarray] | None) -> None:
    if frames is not None:
        frames.append(env.render(480, 360))


def move(env, cmd: np.ndarray, target: np.ndarray, grip: float, cfg, frames, creep: bool) -> np.ndarray:
    """Step the commanded tcp to `target`. Returns the updated command."""
    tol = float(cfg.scripted.reach_tol)
    limit = float(cfg.scripted.creep_step) if creep else float(cfg.sim.max_step)
    for _ in range(int(cfg.scripted.max_reach_steps)):
        if float(np.linalg.norm(target - cmd)) <= tol:
            break
        dp = _toward(cmd, target, limit)
        env.step(np.concatenate([dp, [grip]]))
        cmd = cmd + dp
        _snap(env, frames)
    return cmd


def dwell(env, cmd: np.ndarray, grip: float, n: int, frames) -> None:
    for _ in range(n):
        env.step(np.array([0.0, 0.0, 0.0, grip]))
        _snap(env, frames)


def pick_and_place(env, cfg, record: bool) -> tuple[bool, list[np.ndarray]]:
    """One episode from the env's current reset. Returns (success, frames)."""
    frames: list[np.ndarray] | None = [] if record else None
    sc = cfg.scripted
    b = env.block_size
    grasp_z = b / 2 + float(sc.grasp_offset)
    approach_z = b + float(sc.approach_height)
    lift_z = float(sc.carry_height)
    dwell_n = int(sc.dwell_steps)
    settle_n = int(sc.settle_steps)
    cmd = env.p_ee_W().copy()
    _snap(env, frames)

    block_xy = env.p_obj_W()[:2].copy()
    cmd = move(env, cmd, np.array([*block_xy, approach_z]), 0.0, cfg, frames, creep=False)
    dwell(env, cmd, 0.0, settle_n, frames)
    cmd = move(env, cmd, np.array([*env.p_obj_W()[:2], grasp_z]), 0.0, cfg, frames, creep=True)
    dwell(env, cmd, 1.0, dwell_n, frames)

    cmd = move(env, cmd, np.array([cmd[0], cmd[1], lift_z]), 1.0, cfg, frames, creep=False)
    target_xy = env.layout.target_xy
    cmd = move(env, cmd, np.array([*target_xy, lift_z]), 1.0, cfg, frames, creep=False)
    dwell(env, cmd, 1.0, settle_n, frames)
    # The block hangs a few millimetres below the tcp. Stopping the tcp at the resting
    # centre drives the block into the plate and the opening fingers shove it off.
    hang = float(cmd[2] - env.p_obj_W()[2])
    place_z = env.h_tgt + b / 2 + max(hang, 0.0) + float(sc.release_clearance)
    cmd = move(env, cmd, np.array([*target_xy, place_z]), 1.0, cfg, frames, creep=True)
    dwell(env, cmd, 1.0, settle_n, frames)
    # Open without moving the hand. Lifting or stepping aside while the fingers are
    # in contact shoves the block off the plate; once they have finished opening it is
    # already sitting there.
    dwell(env, cmd, 0.0, int(sc.open_steps), frames)
    env.settle()
    _snap(env, frames)
    return env.success(), [] if frames is None else frames


def run_goal(goal: str, cfg, scene, record_seed: int | None) -> list[tuple[int, bool, list]]:
    rows = []
    for i in range(int(cfg.scripted.n_seeds)):
        seed = int(cfg.seed) + i
        env = Env(goal, seed, cfg, scene=scene)
        env.reset()
        ok, frames = pick_and_place(env, cfg, record=seed == record_seed)
        rows.append((seed, ok, frames))
        env.close()
    return rows


def summarise(goal: str, rows: list[tuple[int, bool, list]]) -> None:
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"{goal:5s}  {n_ok}/{len(rows)} succeeded")
    for seed, ok, _ in rows:
        if not ok:
            print(f"       seed {seed} failed")


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_csv = resolve(cfg.paths.results) / "scripted_check.csv"
    if should_skip(out_csv, bool(cfg.get("overwrite", False))):
        return

    sweep = bool(cfg.get("sweep", False))
    with Timer("scripted_check"):
        scene = build_scene(cfg)
        if sweep:
            print("grasp_offset  max_step   " + "  ".join(f"{g:>5s}" for g in GOAL_BODIES))
            for grasp_offset in cfg.scripted.sweep_grasp_offset:
                for max_step in cfg.scripted.sweep_max_step:
                    trial = OmegaConf.merge(
                        cfg,
                        OmegaConf.create(
                            {
                                "scripted": {"grasp_offset": float(grasp_offset)},
                                "sim": {"max_step": float(max_step)},
                            }
                        ),
                    )
                    counts = []
                    for goal in GOAL_BODIES:
                        rows = run_goal(goal, trial, scene, record_seed=None)
                        counts.append(f"{sum(ok for _, ok, _ in rows):2d}/{len(rows)}")
                    print(
                        f"{float(grasp_offset):12.3f}  {float(max_step):8.3f}   "
                        + "  ".join(f"{c:>5s}" for c in counts)
                    )
            return

        lines = ["goal,seed,success"]
        ensure_dir(out_csv.parent)
        for goal in GOAL_BODIES:
            rows = run_goal(goal, cfg, scene, record_seed=int(cfg.seed))
            summarise(goal, rows)
            for seed, ok, frames in rows:
                lines.append(f"{goal},{seed},{int(ok)}")
                if frames:
                    path = resolve(cfg.paths.results) / f"scripted_check_{goal}.mp4"
                    write_video(path, frames, fps=float(cfg.human.rate_hz))
                    print(f"wrote {path}")
        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
