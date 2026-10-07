"""Robot episodes for the world model (EXPLAINER Module 8).

Each episode: a layout from its seed, a random goal, a random QC-passed success clip,
the two-anchor plan plus smooth knot noise, and with probability 0.2 a failure
(early release, or a close 2–3 cm off the block). No rendering.

Seeds 1000–1299 are training, 900–999 are held out. Seeds 0–49 are not used here.

Usage:
    uv run python scripts/generate_robot.py
    uv run python scripts/generate_robot.py overwrite=true
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401

from palm_prior.retarget.build_plan import build_plan  # noqa: E402
from palm_prior.sim.env import Env  # noqa: E402
from palm_prior.sim.scene import GOAL_BODIES, build_scene  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, should_skip  # noqa: E402
from replay import _passed_successes, _load_clip, _track  # noqa: E402


def _one(env: Env, seed: int, clips: list[int], loaded: dict, cfg, split: str) -> dict:
    rng = np.random.default_rng(seed)
    goal = str(rng.choice(list(GOAL_BODIES)))
    env.goal = goal
    env.h_tgt = float(cfg.goals[goal])
    env.rng = np.random.default_rng(seed)
    env.reset()
    clip_id = int(rng.choice(clips))
    knots = rng.normal(0.0, float(cfg.robot_data.knot_noise), size=(int(cfg.retarget.n_knots), 3))
    knots[0] = 0.0
    mode = "none"
    block_xy = env.layout.block_xy.copy()
    shift = None
    if rng.random() < float(cfg.robot_data.failure_prob):
        if rng.random() < 0.5:
            mode = "early"
            # Rebuilt below once the grasp window is known.
        else:
            mode = "offset"
            lo, hi = (float(x) for x in cfg.robot_data.failure_offset)
            mag = float(rng.uniform(lo, hi))
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            block_xy = block_xy + mag * direction
    plan = build_plan(
        loaded[clip_id], block_xy, env.layout.target_xy, env.h_tgt,
        knots, None, cfg, ee_start_W=env.p_ee_W(),
    )
    if mode == "early" and plan.k_open - plan.k_close > 4:
        delta = int(rng.integers(2, max(3, (plan.k_open - plan.k_close) // 2)))
        plan = build_plan(
            loaded[clip_id], block_xy, env.layout.target_xy, env.h_tgt,
            knots, np.array([0.0, -delta]), cfg, ee_start_W=env.p_ee_W(),
        )
    _frames, rows = _track(env, plan, cfg, record=False)
    ok = bool(env.success())
    if not rows:
        raise RuntimeError(f"seed {seed} produced no steps")
    s = np.stack([r[0] for r in rows])
    a = np.stack([r[1] for r in rows])
    dp = np.stack([r[2] for r in rows])
    att = np.array([r[3] for r in rows])
    return {
        "s": s, "a": a, "dp_obj": dp, "attach": att,
        "seed": seed, "goal": goal, "clip": clip_id,
        "success": int(ok), "failure_mode": mode, "split": split,
    }


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    out_dir = ensure_dir(resolve(cfg.paths.robot_dir))
    out = out_dir / "episodes.npz"
    if should_skip(out, bool(cfg.get("overwrite", False))):
        return
    clips = _passed_successes(cfg)
    loaded = {i: _load_clip(cfg, i) for i in clips}
    rd = cfg.robot_data
    jobs = [(int(rd.heldout_seed_start) + i, "heldout") for i in range(int(rd.n_heldout_episodes))]
    jobs += [(int(rd.train_seed_start) + i, "train") for i in range(int(rd.n_train_episodes))]

    scene = build_scene(cfg)
    env = Env("plate", 0, cfg, scene=scene)
    episodes = []
    with Timer("generate_robot"):
        for n, (seed, split) in enumerate(jobs):
            episodes.append(_one(env, seed, clips, loaded, cfg, split))
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(jobs)}")
        env.close()

    def cat(key):
        return np.concatenate([ep[key] for ep in episodes])

    ids = np.concatenate([np.full(len(ep["s"]), ep["seed"]) for ep in episodes])
    success = np.array([ep["success"] for ep in episodes])
    seeds = np.array([ep["seed"] for ep in episodes])
    split = np.array([ep["split"] for ep in episodes])
    mode = np.array([ep["failure_mode"] for ep in episodes])
    train = split == "train"
    print(
        f"train success {100 * success[train].mean():.1f}%  "
        f"held-out success {100 * success[~train].mean():.1f}%  "
        f"injected {(mode != 'none').mean() * 100:.1f}%"
    )
    np.savez_compressed(
        out,
        s=cat("s"), a=cat("a"), dp_obj=cat("dp_obj"), attach=cat("attach"),
        episode_id=ids, episode_seed=seeds, episode_success=success,
        episode_split=split, episode_mode=mode,
    )
    print(f"wrote {out}  ({len(cat('s'))} transitions, {len(episodes)} episodes)")


if __name__ == "__main__":
    main(sys.argv[1:])
