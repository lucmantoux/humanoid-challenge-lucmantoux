"""E1: open-loop replay of retargeted human clips (EXPLAINER §25).

methods: naive (one fixed similarity for every clip) and two_anchor (anchors per layout).
Layouts are seeds 0–49, the range reserved for evaluation, crossed with every goal.
QC-passed success clips are used round-robin. The arm tracks p_star by adding (next waypoint − current command) each step.
env.step adds that delta to the commanded tcp and clips it to sim.max_step.

Usage:
    uv run python scripts/replay.py
    uv run python scripts/replay.py n_seeds=1
    uv run python scripts/replay.py overwrite=true
"""

from __future__ import annotations

import csv
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from palm_prior.retarget.build_plan import build_plan  # noqa: E402
from palm_prior.sim.env import Env  # noqa: E402
from palm_prior.state import attached  # noqa: E402
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


def _passed_successes(cfg) -> list[int]:
    report = resolve(cfg.paths.results) / "extraction_report.csv"
    ids = []
    with report.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            cols = dict(zip(header, line.strip().split(",")))
            if cols["type"] == "success" and cols["pass"] == "1":
                ids.append(int(cols["clip"]))
    if not ids:
        raise SystemExit("no QC-passed success clips in extraction_report.csv")
    return ids


def _file_of(cfg) -> dict[int, str]:
    path = resolve(cfg.paths.raw_dir) / "clip_files.csv"
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["clip"])] = row["file"]
    return out


def _load_clip(cfg, clip_id: int) -> dict:
    path = resolve(cfg.paths.human_dir) / f"clip_{clip_id:03d}.npz"
    data = np.load(path)
    return {key: data[key] for key in ("t_frame", "p_ee", "g")}


def _track(env: Env, plan, cfg, record: bool) -> tuple[list[np.ndarray], list[tuple]]:
    """Approach the grasp from above, then follow the retargeted path.

    The command is what env.step integrates, so each delta is (waypoint − command),
    clipped to sim.max_step. A straight dive from the home pose stalls on the table,
    so the approach is the same one the scripted check uses: hover, creep down,
    dwell closed, then the path from the grasp onward.
    """
    frames = []
    rows: list[tuple] = []
    p = plan.p_star
    cmd = env.p_ee_W().copy()
    fast = float(cfg.sim.max_step)
    creep = float(cfg.scripted.creep_step)
    tol = float(cfg.scripted.reach_tol)

    def go(target: np.ndarray, grip: float, limit: float) -> None:
        nonlocal cmd
        dp = target - cmd
        norm = float(np.linalg.norm(dp))
        if norm > limit:
            dp = dp * (limit / norm)
        action = np.concatenate([dp, [grip]])
        before = env.get_state().copy()
        p_obj = env.p_obj_W().copy()
        env.step(action)
        dp_obj = env.p_obj_W() - p_obj
        att = float(attached(before[7], before[5], env.block_size, float(cfg.human.attach_margin)))
        rows.append((before, action.copy(), dp_obj, att))
        cmd = cmd + dp
        if record:
            frames.append(env.render(480, 360))

    def reach(target: np.ndarray, grip: float, limit: float) -> None:
        for _ in range(int(cfg.scripted.max_reach_steps)):
            if float(np.linalg.norm(target - cmd)) <= tol:
                break
            go(target, grip, limit)

    grasp = p[plan.k_close]
    hover_z = env.block_size + float(cfg.scripted.approach_height)
    reach(np.array([grasp[0], grasp[1], hover_z]), 0.0, fast)
    for _ in range(int(cfg.scripted.settle_steps)):
        go(cmd, 0.0, fast)
    reach(grasp, 0.0, creep)
    for _ in range(int(cfg.scripted.dwell_steps)):
        go(grasp, 1.0, creep)
    for k in range(plan.k_close, len(p) - 1):
        grip = float(plan.grip[min(k + 1, len(p) - 1)])
        go(p[k + 1], grip, fast)
        if k + 1 == plan.k_open:
            for _ in range(int(cfg.scripted.open_steps)):
                go(p[k + 1], 0.0, fast)
    env.settle()
    if record and frames:
        frames.append(env.render(480, 360))
    return frames, rows


def _side_by_side(phone: str, sim_frames: list[np.ndarray], out, cfg) -> None:
    """Phone frames at the stretched robot times, beside the sim frames."""
    cap = cv2.VideoCapture(phone)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1.0 / (float(cfg.human.rate_hz) * float(cfg.human.tau))
    h, w = sim_frames[0].shape[:2]
    paired = []
    for k, sim in enumerate(sim_frames):
        idx = min(n - 1, int(round(k * step * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((h, w, 3), np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (w, h))
        paired.append(np.concatenate([fr, sim], axis=1))
    cap.release()
    write_video(out, paired, float(cfg.human.rate_hz))


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    results = ensure_dir(resolve(cfg.paths.results))
    out_csv = results / "e1.csv"
    if should_skip(out_csv, bool(cfg.get("overwrite", False))):
        return

    clips = _passed_successes(cfg)
    loaded = {i: _load_clip(cfg, i) for i in clips}
    files = _file_of(cfg)
    lo, hi = (int(x) for x in cfg.robot_data.eval_seeds)
    seeds = list(range(lo, hi + 1))
    n_seeds = cfg.get("n_seeds", None)
    if n_seeds is not None:
        seeds = seeds[: int(n_seeds)]
    goals = list(GOAL_BODIES)

    # The naive transform is fixed from the first passed clip and seed 0's plate
    # layout, then applied to every other clip. It never sees the layout it is replayed on.
    scene = build_scene(cfg)
    env = Env("plate", seeds[0], cfg, scene=scene)
    env.reset()
    ref = build_plan(
        loaded[clips[0]], env.layout.block_xy, env.layout.target_xy,
        float(cfg.goals.plate), None, None, cfg, ee_start_W=None,
    )
    naive = (ref.alpha, ref.beta)
    print(f"naive alpha={naive[0]:.4f} beta={naive[1]:.4f} from clip {clips[0]}, seed {seeds[0]} plate")

    rows = []
    videos_done = 0
    with Timer("replay"):
        for goal in goals:
            env.goal = goal
            env.h_tgt = float(cfg.goals[goal])
            for seed in seeds:
                env.rng = np.random.default_rng(seed)
                env.reset()
                clip_id = clips[(seed + goals.index(goal)) % len(clips)]
                clip = loaded[clip_id]
                for method in ("naive", "two_anchor"):
                    saved = env.save_state()
                    plan = build_plan(
                        clip, env.layout.block_xy, env.layout.target_xy, env.h_tgt,
                        None, None, cfg, ee_start_W=env.p_ee_W(),
                        alpha_beta=naive if method == "naive" else None,
                    )
                    record = method == "two_anchor" and seed == seeds[0] and videos_done < 3
                    frames, _rows = _track(env, plan, cfg, record)
                    ok = env.success()
                    err = float(np.linalg.norm(env.p_obj_W()[:2] - env.layout.target_xy))
                    rows.append({
                        "method": method, "goal": goal, "seed": seed, "clip": clip_id,
                        "success": int(ok), "xy_err_m": round(err, 4),
                    })
                    if record:
                        phone = resolve(cfg.paths.raw_dir) / files.get(clip_id, f"demo_{clip_id:03d}.mp4")
                        video = results / f"e1_{goal}.mp4"
                        if phone.exists():
                            _side_by_side(str(phone), frames, video, cfg)
                            print(f"wrote {video}")
                        videos_done += 1
                    env.restore_state(saved)
                    print(f"{method:11s} {goal:5s} seed {seed:2d} clip {clip_id:2d}  {'ok' if ok else 'miss'}  {err*100:.1f} cm")
        env.close()

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "goal", "seed", "clip", "success", "xy_err_m"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")
    for method in ("naive", "two_anchor"):
        part = [r for r in rows if r["method"] == method]
        rate = 100.0 * np.mean([r["success"] for r in part])
        print(f"{method:11s} {rate:.1f}%  ({sum(r['success'] for r in part)}/{len(part)})")


if __name__ == "__main__":
    main(sys.argv[1:])
