"""Phase 2: phone clips to the shared state (EXPLAINER Modules 3–5).

One static table pose per clip, a scaled pinch point, a colour-tracked block, then
10 Hz transitions. Writes a raw npz and a debug mp4 per clip, then the combined
data/human/transitions.npz and results/extraction_report.csv.

The target in the state is the photographed pencil mark, not the coordinate in
dots.yaml. EXPLAINER Module 5 reads the target from that file. The tape on this
table sits a few centimetres off the file, while the lid in the opening frames
sits on the tape, so each mark is the median opening lid of the clips that start
on it. The 2 cm check compares a clip to that photographed mark.

Usage:
    uv run python scripts/extract_human.py clips=[1,31,35]
    uv run python scripts/extract_human.py
    uv run python scripts/extract_human.py overwrite=true
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import _bootstrap  # noqa: E402,F401

from palm_prior.human.transitions import (  # noqa: E402
    clip_transitions,
    expected_grasps,
    grasp_count,
)
from palm_prior.perception.aruco import (  # noqa: E402
    board_pose,
    build_board,
    intersect_plane_z,
    make_detector,
    project,
    ray,
    static_pose,
)
from palm_prior.perception.block import load_hsv, mask  # noqa: E402
from palm_prior.perception.block_track import track_block  # noqa: E402
from palm_prior.perception.hand import (  # noqa: E402
    CONNECTIONS,
    INDEX_TIP,
    THUMB_TIP,
    HandFrame,
    PnPPinch,
    aperture,
    detect,
    ensure_model,
    fill_gaps,
    gripper_signal,
    open_landmarker,
    pinch_camera,
    pinch_px,
    rest_scale,
    smooth_series,
    solve_pinch,
)
from palm_prior.utils import (  # noqa: E402
    Timer,
    ensure_dir,
    load_config,
    resolve,
    set_seed,
    video_frames,
    video_info,
)


def read_log(path: Path) -> list[dict]:
    """recording_log.csv rows. clip numbers are ints, dots are ints, redo is 0/1."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["clip"] = int(row["clip"])
        row["block_dot"] = int(row["block_dot"])
        row["plate_dot"] = int(row["plate_dot"])
        row["redo"] = int(row["redo"])
    return rows


def load_dots(path: Path) -> dict[int, np.ndarray]:
    """Dot positions in metres, table frame. The file stores centimetres."""
    raw = OmegaConf.load(path)
    return {int(k): np.array(v, float) / 100.0 for k, v in raw.dots.items()}


def hold_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short NaN runs with the nearer finite neighbour. Binary signals stay binary."""
    out = np.asarray(values, float).copy()
    valid = np.isfinite(out)
    n = len(out)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if (j - i) <= max_gap and (i > 0 or j < n):
            for k in range(i, j):
                if i == 0:
                    out[k] = out[j]
                elif j == n:
                    out[k] = out[i - 1]
                else:
                    out[k] = out[i - 1] if (k - i) <= (j - 1 - k) else out[j]
        i = j
    return out


def _file_map(raw_dir: Path) -> dict[int, str]:
    """Log clip id -> filename. Empty when every clip is demo_<id>.mp4.

    The phone numbered the files in filming order, which is not the log order: one
    success was filmed twice, two more were filmed again later, and those three extra
    files shift every id after them. data/raw/clip_files.csv is that alignment, checked
    by matching each clip's still start dot and end dot to the log.
    """
    path = raw_dir / "clip_files.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["clip"])] = row["file"]
    return out


def _selected(cfg, rows: list[dict]) -> list[dict]:
    chosen = cfg.get("clips")
    if chosen is None:
        return rows
    if isinstance(chosen, int):
        want = {int(chosen)}
    else:
        want = {int(c) for c in chosen}
    return [r for r in rows if r["clip"] in want]


def _blank(n: int, *shape: int) -> np.ndarray:
    return np.full((n, *shape), np.nan)


def observe(video: Path, cfg, K, dist, tb, detector, landmarker, hsv) -> dict:
    """One pass: board poses, hand landmarks, block blobs. Nothing is metric yet."""
    info = video_info(video)
    n = 0
    poses = []
    px_all, world_all, ap_all = [], [], []
    cent_all, area_all = [], []
    for i, frame in video_frames(video):
        n += 1
        pose = board_pose(frame, tb, detector, K, dist, int(cfg.aruco.min_markers))
        poses.append(pose)
        hand = detect(landmarker, frame, int(round(1000.0 * i / info.fps)))
        if hand is None:
            px_all.append(np.full((21, 2), np.nan))
            world_all.append(np.full((21, 3), np.nan))
            ap_all.append(np.nan)
        else:
            px_all.append(hand.px)
            world_all.append(hand.world)
            ap_all.append(aperture(hand))
        blob = mask(frame, hsv, cfg)
        if blob.centroid is None:
            cent_all.append([np.nan, np.nan])
            area_all.append(0.0)
        else:
            cent_all.append(blob.centroid)
            area_all.append(blob.area)
    found = [p for p in poses if p is not None]
    if not found:
        raise RuntimeError(f"board never detected in {video.name}")
    return {
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "n": n,
        "px": np.stack(px_all),
        "world": np.stack(world_all),
        "aperture": np.array(ap_all, float),
        "centroid": np.array(cent_all, float),
        "area": np.array(area_all, float),
        "R_CT": static_pose(found)[0],
        "t_CT": static_pose(found)[1],
    }


def metric(obs: dict, cfg, K, dist, k_fallback: float | None = None) -> dict:
    """Pinch in T, gripper, block in T. obs is the dict from observe()."""
    n = obs["n"]
    R_CT, t_CT = obs["R_CT"], obs["t_CT"]
    rest = max(1, int(round(float(cfg.hand.rest_seconds) * obs["fps"])))
    z_plane = np.full(n, np.nan)
    z_pnp = np.full(n, np.nan)
    reproj = np.full(n, np.nan)
    pnp_R = _blank(n, 3, 3)
    pnp_t = _blank(n, 3)
    world_pinch = _blank(n, 3)
    for i in range(n):
        if not np.isfinite(obs["px"][i]).all():
            continue
        hand = HandFrame(px=obs["px"][i], world=obs["world"][i])
        pnp = solve_pinch(hand, K, dist)
        if pnp is None:
            continue
        u, v = pinch_px(hand)
        direction = ray(float(u), float(v), K, dist)
        # Rest hands lie on the table, so the plane is z_T = 0 (EXPLAINER §7).
        on_plane = intersect_plane_z(direction, R_CT, t_CT, 0.0)
        # Depth of that table point, used only as Z_plane(u_pinch).
        X_C = R_CT @ on_plane + t_CT
        z_plane[i] = X_C[2]
        z_pnp[i] = pnp.z_pnp
        reproj[i] = pnp.reproj
        pnp_R[i] = pnp.R
        pnp_t[i] = pnp.t
        world_pinch[i] = 0.5 * (hand.world[THUMB_TIP] + hand.world[INDEX_TIP])

    rest_idx = np.concatenate([np.arange(min(rest, n)), np.arange(max(0, n - rest), n)])
    k = rest_scale(z_plane[rest_idx], z_pnp[rest_idx])
    k_borrowed = not np.isfinite(k)
    if k_borrowed:
        # The hand never rests in frame (a push that starts and ends off camera).
        # k is a property of this camera and the hand model, so the median k of the
        # clips that do rest is the right stand-in. EXPLAINER §7 wants this clip's own rest.
        if k_fallback is None or not np.isfinite(k_fallback):
            raise RuntimeError("no rest frames to fix the hand scale, and no other clip to borrow k from")
        k = float(k_fallback)

    p_C = _blank(n, 3)
    for i in range(n):
        if not np.isfinite(z_pnp[i]):
            continue
        pnp = PnPPinch(R=pnp_R[i], t=pnp_t[i], z_pnp=float(z_pnp[i]), reproj=float(reproj[i]))
        p_C[i] = pinch_camera(pnp, world_pinch[i], k)
    p_ee = np.full_like(p_C, np.nan)
    finite = np.isfinite(p_C).all(axis=1)
    p_ee[finite] = (R_CT.T @ (p_C[finite] - t_CT).T).T

    rest_mask = np.zeros(n, dtype=bool)
    rest_mask[: min(rest, n)] = True
    rest_mask[max(0, n - rest) :] = True
    if not np.isfinite(obs["aperture"][rest_mask]).any():
        known = np.isfinite(obs["aperture"])
        if known.any():
            # No still hand at either end. The fingers-together pose is the low aperture,
            # wherever it occurs, and that is the open pose on these clips.
            thr = float(np.percentile(obs["aperture"][known], float(cfg.hand.aperture_lo_pct)))
            rest_mask = known & (obs["aperture"] <= thr)
    g, close_thr, open_thr = gripper_signal(obs["aperture"], cfg, rest_mask)
    g = hold_short_gaps(g, int(cfg.hand.max_gap_frames))
    p_ee = fill_gaps(p_ee, int(cfg.hand.max_gap_frames))
    p_C = fill_gaps(p_C, int(cfg.hand.max_gap_frames))
    p_ee = smooth_series(p_ee, int(cfg.hand.savgol_window), int(cfg.hand.savgol_order))

    p_obj, valid_obj = track_block(
        obs["centroid"], obs["area"], g, p_ee, p_C, R_CT, t_CT, K, dist,
        float(cfg.objects.block_size), rest, cfg,
    )
    valid_hand = np.isfinite(p_ee).all(axis=1)
    return {
        "k": k,
        "g": g,
        "p_ee": p_ee,
        "p_C": p_C,
        "p_obj": p_obj,
        "reproj": reproj,
        "valid_hand": valid_hand,
        "valid_obj": valid_obj,
        "close_thr": close_thr,
        "open_thr": open_thr,
        "rest": rest,
        "k_borrowed": k_borrowed,
    }


def qc_row(row: dict, obs: dict, met: dict, dots: dict, cfg) -> dict:
    """One line of extraction_report.csv."""
    n = obs["n"]
    rest = met["rest"]
    start = met["p_obj"][:rest]
    start = start[np.isfinite(start).all(axis=1)]
    block_xy = dots[row["block_dot"]]
    if len(start) == 0:
        start_err = float("nan")
    else:
        start_err = float(np.linalg.norm(np.median(start[:, :2], axis=0) - block_xy))
    reproj = met["reproj"][np.isfinite(met["reproj"])]
    reproj_med = float(np.median(reproj)) if len(reproj) else float("nan")
    vh = float(met["valid_hand"].mean())
    vb = float(met["valid_obj"].mean())
    n_grasp = grasp_count(met["g"])
    want = expected_grasps(row["type"])
    q = cfg.human.qc
    passed = (
        vh >= float(q.min_valid_hand)
        and vb >= float(q.min_valid_block)
        and np.isfinite(start_err) and start_err <= float(q.max_start_err)
        and np.isfinite(reproj_med) and reproj_med <= float(cfg.hand.max_reproj_px)
        and n_grasp == want
    )
    return {
        "clip": row["clip"],
        "type": row["type"],
        "valid_hand": round(vh, 3),
        "valid_block": round(vb, 3),
        "n_grasp": n_grasp,
        "reproj_px": round(reproj_med, 2) if np.isfinite(reproj_med) else "",
        "start_err_m": round(start_err, 4) if np.isfinite(start_err) else "",
        "k": round(met["k"], 3),
        "pass": int(passed),
    }


def _plot_panel(t, p_ee, p_obj, aperture_s, g, duration) -> tuple[np.ndarray, tuple]:
    """A static side plot. Returns the RGB image and (x_left, x_right, t0, t1) in pixels."""
    fig, ax = plt.subplots(figsize=(4.8, 5.4), dpi=100)
    ax.plot(t, p_ee[:, 2] * 100.0, color="#1d4e89", lw=1.2, label="hand z (cm)")
    ax.plot(t, p_obj[:, 2] * 100.0, color="#c45c26", lw=1.2, label="block z (cm)")
    ax.set_xlim(0, max(duration, 1e-3))
    ax.set_ylim(-1, 30)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height (cm)")
    ax2 = ax.twinx()
    ax2.plot(t, aperture_s, color="#2e7d32", lw=1.0, alpha=0.85, label="aperture")
    closed = np.isfinite(g) & (g >= 0.5)
    if closed.any():
        ax2.fill_between(t, 0, 1, where=closed, color="#2e7d32", alpha=0.12, step="pre")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("aperture")
    ax.set_title("green band = gripper closed")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    bbox = ax.get_position()
    h, w = img.shape[:2]
    span = (int(bbox.x0 * w), int(bbox.x1 * w), 0.0, float(max(duration, 1e-3)))
    plt.close(fig)
    return img, span


def _draw_pt(u: float, v: float, width: int, height: int) -> tuple[int, int] | None:
    """Pixel for cv2, or None when the point is off the image by more than 2000 px.

    A point behind the camera projects to a coordinate OpenCV cannot draw.
    """
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    x, y = int(round(float(u))), int(round(float(v)))
    if x < -2000 or y < -2000 or x > width + 2000 or y > height + 2000:
        return None
    return (x, y)


def write_debug(video: Path, obs: dict, met: dict, out: Path, K, dist) -> None:
    """Landmarks, green pinch, red reprojection, blue block box, and the side plot."""
    t = np.arange(obs["n"]) / obs["fps"]
    panel, (x0, x1, t0, t1) = _plot_panel(t, met["p_ee"], met["p_obj"], obs["aperture"], met["g"], t[-1])
    ph, pw = panel.shape[:2]
    info = video_info(video)
    scale = ph / info.height
    vw = int(round(info.width * scale))
    vw -= vw % 2
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (vw + pw, ph))
    R_CT, t_CT = obs["R_CT"], obs["t_CT"]
    try:
        for i, frame in video_frames(video):
            if i >= obs["n"]:
                break
            vis = cv2.resize(frame, (vw, ph))
            sx, sy = vw / info.width, ph / info.height
            px = obs["px"][i]
            if np.isfinite(px).all():
                for a, b in CONNECTIONS:
                    pa = _draw_pt(px[a, 0] * sx, px[a, 1] * sy, vw, ph)
                    pb = _draw_pt(px[b, 0] * sx, px[b, 1] * sy, vw, ph)
                    if pa is not None and pb is not None:
                        cv2.line(vis, pa, pb, (210, 210, 210), 1)
                for tip in (THUMB_TIP, INDEX_TIP):
                    pt = _draw_pt(px[tip, 0] * sx, px[tip, 1] * sy, vw, ph)
                    if pt is not None:
                        cv2.circle(vis, pt, 4, (0, 255, 255), -1)
                pinch = 0.5 * (px[THUMB_TIP] + px[INDEX_TIP])
                pt = _draw_pt(pinch[0] * sx, pinch[1] * sy, vw, ph)
                if pt is not None:
                    cv2.circle(vis, pt, 6, (0, 255, 0), -1)
            if np.isfinite(met["p_ee"][i]).all():
                uv = project(met["p_ee"][i], R_CT, t_CT, K, dist)
                pt = _draw_pt(uv[0] * sx, uv[1] * sy, vw, ph)
                if pt is not None:
                    cv2.circle(vis, pt, 5, (0, 0, 255), 2)
            uvb = obs["centroid"][i]
            if np.isfinite(uvb).all():
                c = _draw_pt(uvb[0] * sx, uvb[1] * sy, vw, ph)
                if c is not None:
                    cv2.rectangle(vis, (c[0] - 14, c[1] - 14), (c[0] + 14, c[1] + 14), (255, 80, 0), 2)
            label = "CLOSED" if np.isfinite(met["g"][i]) and met["g"][i] >= 0.5 else "OPEN"
            cv2.putText(vis, f"frame {i}  {label}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
            side = panel.copy()
            if t1 > t0:
                x = int(x0 + (t[i] - t0) / (t1 - t0) * (x1 - x0))
                cv2.line(side, (x, 0), (x, ph - 1), (200, 30, 30), 1)
            writer.write(np.concatenate([vis, side], axis=1))
    finally:
        writer.release()


def save_clip(path: Path, row: dict, obs: dict, met: dict, tr, dots: dict, h_tgt: float) -> None:
    np.savez_compressed(
        path,
        t_frame=np.arange(obs["n"]) / obs["fps"],
        p_ee=met["p_ee"],
        p_obj=met["p_obj"],
        g=met["g"],
        aperture=obs["aperture"],
        reproj=met["reproj"],
        valid_hand=met["valid_hand"],
        valid_obj=met["valid_obj"],
        s=tr.s,
        a=tr.a,
        dp_obj=tr.dp_obj,
        attach=tr.attach,
        k=np.array(met["k"]),
        clip_id=np.array(row["clip"]),
        is_failure=np.array(0 if row["type"] == "success" else 1),
        clip_type=np.array(row["type"]),
        h_tgt=np.array(h_tgt),
        block_xy=dots[row["block_dot"]],
        target_xy=dots[row["plate_dot"]],
    )


def _opening_xy(p_obj: np.ndarray, t_frame: np.ndarray, cfg) -> np.ndarray | None:
    """Lid centre in the opening still seconds, while it is on the table. (2,) or None.

    Frames whose height is more than 2 cm off the table are a bad depth, not the mark.
    """
    p_obj = np.asarray(p_obj, float)
    t_frame = np.asarray(t_frame, float)
    half = float(cfg.objects.block_size) / 2.0
    early = (t_frame < float(cfg.hand.rest_seconds)) & np.isfinite(p_obj).all(axis=1)
    early = early & (np.abs(p_obj[:, 2] - half) < 0.02)
    if int(early.sum()) < 5:
        return None
    return np.median(p_obj[early, :2], axis=0)


def apply_photographed_dots(human_dir: Path, log_rows: list[dict], dots: dict, cfg) -> None:
    """Set each clip's block and plate from the opening frames, then rebuild its state.

    A mark's position is the median opening lid over every saved clip that starts on
    that mark. A clip with no on-table opening keeps the plan from dots.yaml.
    """
    openings: dict[int, np.ndarray] = {}
    by_dot: dict[int, list[np.ndarray]] = {}
    cached: dict[int, dict] = {}
    for row in log_rows:
        path = human_dir / f"clip_{int(row['clip']):03d}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        cached[int(row["clip"])] = {key: np.array(data[key]) for key in data.files}
        xy = _opening_xy(cached[int(row["clip"])]["p_obj"], cached[int(row["clip"])]["t_frame"], cfg)
        if xy is None:
            continue
        openings[int(row["clip"])] = xy
        by_dot.setdefault(int(row["block_dot"]), []).append(xy)

    photo = {dot: np.median(np.stack(samples), axis=0) for dot, samples in by_dot.items()}
    for dot, xy in dots.items():
        photo.setdefault(int(dot), np.asarray(xy, float))
    print("photographed dots, cm (plan in parentheses):")
    for dot in sorted(photo):
        got = photo[dot] * 100.0
        plan = np.asarray(dots[dot], float) * 100.0
        print(f"  dot {dot}: ({got[0]:.1f}, {got[1]:.1f})   plan ({plan[0]:.1f}, {plan[1]:.1f})")

    block_size = float(cfg.objects.block_size)
    for row in log_rows:
        clip = int(row["clip"])
        if clip not in cached:
            continue
        data = cached[clip]
        target_xy = photo[int(row["plate_dot"])]
        block_xy = photo[int(row["block_dot"])]
        t = np.asarray(data["t_frame"], float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30.0
        tr = clip_transitions(
            t, data["p_ee"], data["p_obj"], data["g"],
            target_xy, float(np.asarray(data["h_tgt"])),
            block_size, float(cfg.human.attach_margin),
            float(t[-1] + dt), float(cfg.human.rate_hz), float(cfg.human.tau),
        )
        data["s"] = tr.s
        data["a"] = tr.a
        data["dp_obj"] = tr.dp_obj
        data["attach"] = tr.attach
        data["block_xy"] = block_xy
        data["target_xy"] = target_xy
        np.savez_compressed(human_dir / f"clip_{clip:03d}.npz", **data)


def aggregate(clip_paths: list[Path], out: Path) -> int:
    """Concatenate per-clip transitions. Returns the number of transitions."""
    parts = {key: [] for key in ("s", "a", "dp_obj", "attach", "clip_id", "is_failure")}
    types = []
    for path in clip_paths:
        data = np.load(path, allow_pickle=False)
        n = len(data["s"])
        if n == 0:
            continue
        for key in ("s", "a", "dp_obj", "attach"):
            parts[key].append(data[key])
        parts["clip_id"].append(np.full(n, int(data["clip_id"])))
        parts["is_failure"].append(np.full(n, int(data["is_failure"])))
        types.append(np.full(n, str(data["clip_type"])))
    if not parts["s"]:
        raise RuntimeError("no transitions to write")
    np.savez_compressed(
        out,
        s=np.concatenate(parts["s"]),
        a=np.concatenate(parts["a"]),
        dp_obj=np.concatenate(parts["dp_obj"]),
        attach=np.concatenate(parts["attach"]),
        clip_id=np.concatenate(parts["clip_id"]),
        is_failure=np.concatenate(parts["is_failure"]),
        clip_type=np.concatenate(types),
    )
    return int(sum(len(p) for p in parts["s"]))


def write_report(rows: list[dict], path: Path) -> None:
    fields = ["clip", "type", "valid_hand", "valid_block", "n_grasp", "reproj_px", "start_err_m", "k", "pass"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _borrowed_k(human_dir: Path) -> float | None:
    """Median scale of clips that measured their own rest. None if there are none."""
    ks = []
    for path in human_dir.glob("clip_*.npz"):
        k = float(np.load(path)["k"])
        if np.isfinite(k):
            ks.append(k)
    if not ks:
        return None
    return float(np.median(ks))


def _report_line(path: Path, cfg) -> dict:
    """QC one saved clip. Same checks as qc_row, read back from the npz."""
    data = np.load(path, allow_pickle=False)
    opening = _opening_xy(data["p_obj"], data["t_frame"], cfg)
    block_xy = np.asarray(data["block_xy"], float)
    if opening is None:
        start_err = float("nan")
    else:
        start_err = float(np.linalg.norm(opening - block_xy))
    reproj = np.asarray(data["reproj"], float)
    reproj = reproj[np.isfinite(reproj)]
    reproj_med = float(np.median(reproj)) if len(reproj) else float("nan")
    vh = float(np.asarray(data["valid_hand"]).mean())
    vb = float(np.asarray(data["valid_obj"]).mean())
    n_grasp = grasp_count(np.asarray(data["g"], float))
    kind = str(np.asarray(data["clip_type"]).item())
    q = cfg.human.qc
    passed = (
        vh >= float(q.min_valid_hand)
        and vb >= float(q.min_valid_block)
        and np.isfinite(start_err) and start_err <= float(q.max_start_err)
        and np.isfinite(reproj_med) and reproj_med <= float(cfg.hand.max_reproj_px)
        and n_grasp == expected_grasps(kind)
    )
    return {
        "clip": int(np.asarray(data["clip_id"])),
        "type": kind,
        "valid_hand": round(vh, 3),
        "valid_block": round(vb, 3),
        "n_grasp": n_grasp,
        "reproj_px": round(reproj_med, 2) if np.isfinite(reproj_med) else "",
        "start_err_m": round(start_err, 4) if np.isfinite(start_err) else "",
        "k": round(float(np.asarray(data["k"])), 3),
        "pass": int(passed),
    }


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    rows = _selected(cfg, read_log(resolve(cfg.paths.recording_log)))
    if not rows:
        raise SystemExit("no clips selected")
    dots = load_dots(resolve(cfg.paths.dots))
    cam = OmegaConf.load(resolve(cfg.paths.camera_yaml))
    K = np.array(cam.K, float)
    dist = np.array(cam.dist, float)
    hsv = load_hsv(resolve(cfg.paths.block_hsv))
    tb = build_board(cfg)
    detector = make_detector(tb)
    hand_model = ensure_model(resolve(cfg.paths.hand_model))
    human_dir = ensure_dir(resolve(cfg.paths.human_dir))
    results = ensure_dir(resolve(cfg.paths.results))
    overwrite = bool(cfg.get("overwrite", False))
    raw_dir = resolve(cfg.paths.raw_dir)
    file_of = _file_map(raw_dir)
    present = {p.name for p in raw_dir.glob("demo_*.mp4")}
    used = set(file_of.values()) if file_of else {f"demo_{r['clip']:03d}.mp4" for r in read_log(resolve(cfg.paths.recording_log))}
    extra = sorted(present - used)
    if extra:
        print("ignored, not used by a log row:", ", ".join(extra))

    report_rows = []
    clip_paths = []
    with Timer("extract_human"):
        for row in rows:
                video = raw_dir / file_of.get(row["clip"], f"demo_{row['clip']:03d}.mp4")
                raw_path = human_dir / f"clip_{row['clip']:03d}.npz"
                debug_path = results / f"extract_demo_{row['clip']:03d}.mp4"
                clip_paths.append(raw_path)
                if raw_path.exists() and debug_path.exists() and not overwrite:
                    print(f"skip clip {row['clip']}: outputs exist")
                    saved = np.load(raw_path, allow_pickle=False)
                    # Rebuild the report line from the saved arrays via a thin recompute of counts.
                    report_rows.append({
                        "clip": row["clip"],
                        "type": row["type"],
                        "valid_hand": round(float(saved["valid_hand"].mean()), 3),
                        "valid_block": round(float(saved["valid_obj"].mean()), 3),
                        "n_grasp": grasp_count(saved["g"]),
                        "reproj_px": round(float(np.nanmedian(saved["reproj"])), 2),
                        "start_err_m": "",
                        "k": round(float(saved["k"]), 3),
                        "pass": "",
                    })
                    continue
                if not video.exists():
                    raise FileNotFoundError(video)
                print(f"clip {row['clip']} ({row['type']}) <- {video.name}")
                landmarker = open_landmarker(hand_model)
                try:
                    obs = observe(video, cfg, K, dist, tb, detector, landmarker, hsv)
                    met = metric(obs, cfg, K, dist, _borrowed_k(human_dir))
                    h_tgt = float(cfg.goals[row["goal"]])
                    tr = clip_transitions(
                        np.arange(obs["n"]) / obs["fps"],
                        met["p_ee"], met["p_obj"], met["g"],
                        dots[row["plate_dot"]], h_tgt,
                        float(cfg.objects.block_size), float(cfg.human.attach_margin),
                        obs["n"] / obs["fps"], float(cfg.human.rate_hz), float(cfg.human.tau),
                    )
                    save_clip(raw_path, row, obs, met, tr, dots, h_tgt)
                    try:
                        write_debug(video, obs, met, debug_path, K, dist)
                    except Exception:
                        traceback.print_exc()
                        print(f"  debug video failed for clip {row['clip']}; the state file was saved")
                    line = qc_row(row, obs, met, dots, cfg)
                    report_rows.append(line)
                    print(
                        f"  k={line['k']}  hand={line['valid_hand']}  block={line['valid_block']}  "
                        f"grasps={line['n_grasp']}  reproj={line['reproj_px']} px  "
                        f"start_err={line['start_err_m']} m  pass={line['pass']}  "
                        f"transitions={len(tr.s)}"
                        + ("  k borrowed" if met["k_borrowed"] else "")
                    )
                except Exception:
                    traceback.print_exc()
                    report_rows.append({
                        "clip": row["clip"], "type": row["type"], "valid_hand": 0, "valid_block": 0,
                        "n_grasp": 0, "reproj_px": "", "start_err_m": "", "k": "", "pass": 0,
                    })
                finally:
                    landmarker.close()

    out_npz = human_dir / "transitions.npz"
    # Every saved clip, including ones this run skipped. A subset run must not
    # replace the full dataset with just the clips it touched.
    saved_paths = sorted(human_dir.glob("clip_*.npz"))
    apply_photographed_dots(human_dir, read_log(resolve(cfg.paths.recording_log)), dots, cfg)
    n_trans = aggregate(saved_paths, out_npz)
    report_rows = [_report_line(p, cfg) for p in saved_paths]
    have = {int(r["clip"]) for r in report_rows}
    for row in read_log(resolve(cfg.paths.recording_log)):
        if row["clip"] not in have:
            report_rows.append({
                "clip": row["clip"], "type": row["type"], "valid_hand": 0, "valid_block": 0,
                "n_grasp": 0, "reproj_px": "", "start_err_m": "", "k": "", "pass": 0,
            })
    report_rows.sort(key=lambda r: int(r["clip"]))
    report = results / "extraction_report.csv"
    write_report(report_rows, report)
    n_pass = sum(int(r["pass"]) for r in report_rows)
    print(f"wrote {out_npz}  ({n_trans} transitions)")
    print(f"wrote {report}  ({n_pass}/{len(report_rows)} passed)")


if __name__ == "__main__":
    main(sys.argv[1:])
