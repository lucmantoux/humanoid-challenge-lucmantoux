"""Re-extract the phone clips with palm-size depth and regenerate the results.

    uv run python -m palm_prior.fix_all
    uv run python -m palm_prior.fix_all --only e1
    uv run python -m palm_prior.fix_all --from e2

Old numbers stay in results/v1/. New numbers go to results/v2/.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from palm_prior.human.qc import grasp_passes  # noqa: E402
from palm_prior.human.transitions import clip_transitions, grasp_count  # noqa: E402
from palm_prior.perception.aruco import build_board, make_detector  # noqa: E402
from palm_prior.perception.block import load_hsv  # noqa: E402
from palm_prior.perception.hand import ensure_model, open_landmarker  # noqa: E402
from palm_prior.retarget.build_plan import Plan, grasp_window, path_at_robot_rate  # noqa: E402
from palm_prior.retarget.hybrid import hybrid_path  # noqa: E402
from palm_prior.retarget.two_anchor import similarity  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed  # noqa: E402
from palm_prior.vision.hand_depth import (  # noqa: E402
    INDEX_TIP,
    THUMB_TIP,
    backproject,
    camera_depth,
    filter_log_z,
    fit_frame,
    grasp_intervals,
    kappa_at,
    one_euro,
    segment_lengths_m,
)
from palm_prior.vision.objects import median_point, ray_plane  # noqa: E402

from extract_human import metric, observe, read_log  # noqa: E402


STAGES = [
    "check_inputs", "discover", "extract", "objects", "anchor", "validate_depth",
    "qc", "e1", "robot_data", "e2", "figures", "readme",
]


def _out(cfg) -> Path:
    return ensure_dir(resolve(cfg.pipeline.out_dir))


def _need(cfg) -> None:
    for seg in cfg.hand_depth.segments:
        if seg.get("length_mm") is None:
            raise SystemExit("hand segment length_mm is null. Fill configs/default.yaml from the ruler.")
    for key in ("lid_top_z_m", "plate_rim_z_m", "plate_surface_z_m"):
        if cfg.objects_v2.get(key) is None:
            raise SystemExit(f"objects_v2.{key} is null. Measure it and write it into configs/default.yaml.")
    if not resolve(cfg.paths.camera_yaml).exists():
        raise SystemExit(f"no camera intrinsics at {cfg.paths.camera_yaml}")


def _log(cfg) -> list[dict]:
    return read_log(resolve(cfg.paths.recording_log))


def _video(cfg, clip: int) -> Path:
    return resolve(cfg.paths.raw_dir) / f"demo_{clip:03d}.mp4"


def _nominal(cfg, row: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = yaml.safe_load(resolve(cfg.paths.dots).read_text(encoding="utf-8"))
    dots = {int(k): np.array([float(v[0]), float(v[1])]) / 100.0 for k, v in raw["dots"].items()}
    h_lid = float(cfg.objects_v2.lid_top_z_m) / 2.0
    b = np.array([dots[row["block_dot"]][0], dots[row["block_dot"]][1], h_lid])
    z_plate = float(cfg.objects_v2.plate_rim_z_m)
    t = np.array([dots[row["plate_dot"]][0], dots[row["plate_dot"]][1], z_plate])
    return b, t


def _lid_height(cfg, goal: str, where: str) -> float:
    half = float(cfg.objects_v2.lid_top_z_m) / 2.0
    if where == "start":
        return half
    if goal == "plate":
        return float(cfg.objects_v2.plate_surface_z_m) + half
    return float(cfg.goals[goal]) + half


def _points_on_plane(centroids, index, h, k, dist, r, t) -> np.ndarray:
    pts = []
    for i in index:
        uv = centroids[i]
        if not np.isfinite(uv).all():
            continue
        try:
            pts.append(ray_plane(float(uv[0]), float(uv[1]), h, k, dist, r, t))
        except ValueError:
            continue
    if not pts:
        return np.full(3, np.nan)
    return median_point(np.stack(pts))


def _accept(video_xyz: np.ndarray, nominal: np.ndarray, limit: float) -> np.ndarray:
    if not np.isfinite(video_xyz).all():
        return nominal
    if float(np.linalg.norm(video_xyz[:2] - nominal[:2])) > limit:
        return nominal
    return video_xyz


def _find_plate(frame: np.ndarray) -> tuple[float, float] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        if len(contour) < 30 or cv2.contourArea(contour) < 800:
            continue
        (cx, cy), (w, h), _ = cv2.fitEllipse(contour)
        short, long = min(w, h), max(w, h)
        if long < 80 or long > 500 or short / long < 0.3:
            continue
        area = float(cv2.contourArea(contour))
        if best is None or area > best[0]:
            best = (area, float(cx), float(cy))
    if best is None:
        return None
    return best[1], best[2]


def _pinch_path(obs, g, b_h, b_end, k, dist, cfg) -> dict:
    segments = segment_lengths_m(cfg)
    flat = obs["px"].reshape(-1, 1, 2).astype(np.float64)
    good = np.isfinite(flat[:, 0, 0])
    und = flat.copy()
    if good.any():
        und[good] = cv2.undistortPoints(flat[good], k, dist, P=k)
    pixels = und.reshape(obs["px"].shape)
    obs = dict(obs)
    obs["px"] = pixels
    f = float((k[0, 0] + k[1, 1]) / 2.0)
    k_inv = np.linalg.inv(k)
    r, t = obs["R_CT"], obs["t_CT"]
    n = obs["n"]
    z_ruler = np.full(n, np.nan)
    z_se = np.full(n, np.nan)
    n_kept = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    drop_counts = {(a, b): 0 for a, b, _ in segments}
    seen = 0
    hd = cfg.hand_depth
    for i in range(n):
        if not np.isfinite(obs["px"][i]).all():
            continue
        seen += 1
        fit = fit_frame(
            obs["px"][i], obs["world"][i], segments, f,
            float(hd.min_foreshorten), float(hd.max_residual), int(hd.min_segments),
            bool(hd.tilt_correction),
        )
        z_ruler[i] = fit["Z"]
        z_se[i] = fit["Z_se"]
        n_kept[i] = fit["n_kept"]
        ok[i] = fit["depth_ok"]
        for pair in fit["dropped"]:
            drop_counts[pair] = drop_counts.get(pair, 0) + 1
    sigma = np.where(np.isfinite(z_se) & (z_ruler > 1e-3), z_se / z_ruler, 0.05)
    y = np.log(np.where(z_ruler > 1e-3, z_ruler, np.nan))
    kal = cfg.hand_depth.kalman
    z_smooth, rejected, n_reinit = filter_log_z(
        y, sigma, float(obs["fps"]), float(kal.q), float(kal.gate_sigma),
        float(kal.v_max_mps), int(kal.max_consecutive_reject), bool(kal.rts_smooth),
    )
    intervals = grasp_intervals(g)
    if not intervals:
        anchors_idx = [(0, n - 1)]
    else:
        anchors_idx = intervals

    def med(series, k0):
        w = int(hd.anchor_window)
        sl = series[max(0, k0 - w): k0 + w + 1]
        sl = sl[np.isfinite(sl)]
        return float(np.median(sl)) if len(sl) else float("nan")

    anchor_list = []
    kappa_g = kappa_r = float("nan")
    for j, (kg, kr) in enumerate(anchors_idx):
        zg = med(z_smooth, kg)
        zr = med(z_smooth, kr)
        lid_g = b_h if j == 0 else b_end
        lid_r = b_end
        if np.isfinite(zg) and zg > 1e-3 and np.isfinite(lid_g).all():
            kg_k = camera_depth(lid_g, r, t) / zg
        else:
            kg_k = 1.0
        if np.isfinite(zr) and zr > 1e-3 and np.isfinite(lid_r).all():
            kr_k = camera_depth(lid_r, r, t) / zr
        else:
            kr_k = kg_k
        anchor_list.append((kg, kg_k))
        anchor_list.append((kr, kr_k))
        if j == 0:
            kappa_g = kg_k
        kappa_r = kr_k
    kappa = kappa_at(np.arange(n), anchor_list)
    depth = kappa * z_smooth

    pinch = np.full((n, 2), np.nan)
    for i in range(n):
        px = obs["px"][i]
        if np.isfinite(px[THUMB_TIP]).all() and np.isfinite(px[INDEX_TIP]).all():
            pinch[i] = 0.5 * (px[THUMB_TIP] + px[INDEX_TIP])
    euro = cfg.hand_depth.one_euro
    pinch_f = one_euro(pinch, float(obs["fps"]), float(euro.min_cutoff), float(euro.beta), float(euro.d_cutoff))

    closed = np.isfinite(g) & (g >= 0.5)
    if closed.any() and np.isfinite(obs["centroid"][closed]).all(axis=1).any():
        both = closed & np.isfinite(pinch_f).all(axis=1) & np.isfinite(obs["centroid"]).all(axis=1)
        first = np.where(both)[0][:5]
        if len(first):
            offset = np.median(pinch_f[first] - obs["centroid"][first], axis=0)
            norm = float(np.linalg.norm(offset))
            if norm > 40:
                offset = offset * (40.0 / norm)
        else:
            offset = np.zeros(2)
    else:
        offset = np.zeros(2)

    u_final = pinch_f.copy()
    if bool(hd.lid_lock):
        for i in range(n):
            if not closed[i] or not np.isfinite(obs["centroid"][i]).all():
                continue
            if np.isfinite(pinch_f[i]).all() and np.linalg.norm(obs["centroid"][i] - pinch_f[i]) > float(hd.lid_search_px):
                continue
            u_final[i] = obs["centroid"][i] + offset
        blend = int(hd.blend_frames)
        for kg, kr in anchors_idx:
            for edge, sign in ((kg, 1), (kr, -1)):
                for d in range(blend):
                    i = edge + sign * d
                    if i < 0 or i >= n or not np.isfinite(pinch_f[i]).all() or not np.isfinite(u_final[i]).all():
                        continue
                    w = d / max(blend, 1)
                    u_final[i] = (1 - w) * pinch_f[i] + w * u_final[i]

    p = np.full((n, 3), np.nan)
    for i in range(n):
        if not np.isfinite(u_final[i]).all() or not np.isfinite(depth[i]):
            continue
        p[i] = backproject(float(u_final[i, 0]), float(u_final[i, 1]), float(depth[i]), k_inv, r, t)
    # Release-anchor holdout: kappa fixed at the grasp, pinch at the release.
    p_hold = np.full(3, np.nan)
    kr = anchors_idx[-1][1]
    if np.isfinite(pinch_f[kr]).all() and np.isfinite(z_smooth[kr]) and np.isfinite(kappa_g):
        p_hold = backproject(float(pinch_f[kr, 0]), float(pinch_f[kr, 1]), float(kappa_g * z_smooth[kr]), k_inv, r, t)
    return {
        "p_ee": p,
        "z_ruler": z_ruler,
        "z_smooth": z_smooth,
        "rejected": rejected.astype(np.uint8),
        "n_reinit": n_reinit,
        "depth_ok": ok.astype(np.uint8),
        "n_kept": n_kept,
        "kappa_g": kappa_g,
        "kappa_r": kappa_r,
        "kappa": kappa,
        "p_holdout": p_hold,
        "k_g": int(anchors_idx[0][0]),
        "k_r": int(anchors_idx[-1][1]),
        "drop_counts": drop_counts,
        "seen": seen,
        "reject_rate": float(rejected.mean()) if n else 1.0,
    }


def _extract_one(cfg, row, k, dist, tb, detector, hsv, hand_model) -> dict:
    video = _video(cfg, row["clip"])
    landmarker = open_landmarker(hand_model)
    try:
        obs = observe(video, cfg, k, dist, tb, detector, landmarker, hsv)
        met = metric(obs, cfg, k, dist, None)
    finally:
        landmarker.close()
    b_nom, t_nom = _nominal(cfg, row)
    rest = max(1, int(round(float(cfg.hand.rest_seconds) * obs["fps"])))
    start_idx = np.arange(min(rest, obs["n"]))
    end_idx = np.arange(max(0, obs["n"] - rest), obs["n"])
    b_video = _points_on_plane(obs["centroid"], start_idx, _lid_height(cfg, row["goal"], "start"), k, dist, obs["R_CT"], obs["t_CT"])
    b_end_video = _points_on_plane(obs["centroid"], end_idx, _lid_height(cfg, row["goal"], "end"), k, dist, obs["R_CT"], obs["t_CT"])
    limit = float(cfg.objects_v2.max_dev_from_nominal_m)
    b_h = _accept(b_video, b_nom, limit)
    # The plate target keeps its nominal xy unless the end lid is a real placement nearby.
    b_end = _accept(b_end_video, np.array([t_nom[0], t_nom[1], b_end_video[2] if np.isfinite(b_end_video).all() else t_nom[2]]), limit)
    if not np.isfinite(b_end).all():
        b_end = t_nom.copy()
        b_end[2] = _lid_height(cfg, row["goal"], "end")
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    plate_uv = _find_plate(frame) if ok else None
    if plate_uv is not None:
        try:
            plate = ray_plane(plate_uv[0], plate_uv[1], float(cfg.objects_v2.plate_rim_z_m), k, dist, obs["R_CT"], obs["t_CT"])
        except ValueError:
            plate = t_nom
    else:
        plate = t_nom.copy()
    t_h = _accept(plate, t_nom, limit)
    src = "video" if np.isfinite(b_video).all() and np.allclose(b_h, b_video) else "nominal"
    depth = _pinch_path(obs, met["g"], b_h, b_end, k, dist, cfg)
    return {
        "obs_fps": obs["fps"], "n": obs["n"], "g": met["g"], "p_ee_old": met["p_ee"],
        "p_obj": met["p_obj"], "valid_obj": met["valid_obj"], "reproj": met["reproj"],
        "p_ee": depth["p_ee"], "b_h": b_h, "b_end": b_end, "t_h": t_h,
        "t_frame": np.arange(obs["n"]) / obs["fps"],
        "type": row["type"], "goal": row["goal"], "clip": row["clip"],
        "obj_source": src, "kappa_g": depth["kappa_g"], "kappa_r": depth["kappa_r"],
        "depth_ok": depth["depth_ok"], "z_ruler": depth["z_ruler"], "z_smooth": depth["z_smooth"],
        "rejected": depth["rejected"], "p_holdout": depth["p_holdout"],
        "k_g": depth["k_g"], "k_r": depth["k_r"], "n_kept": depth["n_kept"],
        "reject_rate": depth["reject_rate"], "n_reinit": depth["n_reinit"],
        "valid_hand_old": met["valid_hand"],
    }


def _save_clip(path: Path, blob: dict) -> None:
    np.savez_compressed(path, **{k: v for k, v in blob.items() if k != "type" and k != "goal" and k != "obj_source"},
                        type=np.array(blob["type"]), goal=np.array(blob["goal"]), obj_source=np.array(blob["obj_source"]))


def _load_clip(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    for key in ("type", "goal", "obj_source"):
        if key in out:
            out[key] = str(out[key])
    return out


def stage_extract(cfg, force: bool) -> None:
    from omegaconf import OmegaConf
    _need(cfg)
    out = _out(cfg) / "clips"
    out.mkdir(parents=True, exist_ok=True)
    cam = OmegaConf.load(resolve(cfg.paths.camera_yaml))
    k = np.array(cam.K, float)
    dist = np.array(cam.dist, float)
    tb = build_board(cfg)
    detector = make_detector(tb)
    hsv = load_hsv(resolve(cfg.paths.block_hsv))
    hand_model = ensure_model(resolve(cfg.paths.hand_model))
    rows = []
    for row in _log(cfg):
        dest = out / f"clip_{row['clip']:03d}.npz"
        if dest.exists() and not force:
            print(f"skip clip {row['clip']}")
            continue
        video = _video(cfg, row["clip"])
        if not video.exists():
            raise SystemExit(f"missing {video}")
        print(f"clip {row['clip']} ({row['type']})")
        try:
            blob = _extract_one(cfg, row, k, dist, tb, detector, hsv, hand_model)
        except Exception:
            traceback.print_exc()
            continue
        _save_clip(dest, blob)
        print(f"  kappa_g={blob['kappa_g']:.3f}  depth_ok={blob['depth_ok'].mean():.2f}  source={blob['obj_source']}")
        rows.append(row["clip"])
    print(f"extracted {len(list(out.glob('clip_*.npz')))} clips")


def _clip_dir(cfg) -> Path:
    return _out(cfg) / "clips"


def stage_validate(cfg) -> None:
    depth_dir = ensure_dir(_out(cfg) / "depth")
    rows = []
    for path in sorted(_clip_dir(cfg).glob("clip_*.npz")):
        d = _load_clip(path)
        b_end = np.asarray(d["b_end"], float)
        hold = np.asarray(d["p_holdout"], float)
        err = hold - b_end
        # Camera forward in the table frame is R's third row. Use vertical and full 3D.
        height = abs(float(err[2])) * 100.0 if np.isfinite(err).all() else float("nan")
        full = float(np.linalg.norm(err)) * 100.0 if np.isfinite(err).all() else float("nan")
        rows.append({
            "clip": int(d["clip"]),
            "type": d["type"],
            "kappa_g": float(d["kappa_g"]),
            "kappa_r": float(d["kappa_r"]),
            "depth_ok_frac": float(np.mean(d["depth_ok"])),
            "reject_rate": float(d["reject_rate"]),
            "n_reinit": int(d["n_reinit"]),
            "release_err_cm": full,
            "release_err_height_cm": height,
            "mean_n_kept": float(np.mean(d["n_kept"])),
        })
    dest = depth_dir / "depth_check.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clip"])
        writer.writeheader()
        writer.writerows(rows)
    succ = [r for r in rows if r["type"] == "success" and np.isfinite(r["release_err_height_cm"])]
    if succ:
        med_h = float(np.median([r["release_err_height_cm"] for r in succ]))
        med_3 = float(np.median([r["release_err_cm"] for r in succ]))
        kappas = [r["kappa_g"] for r in rows if np.isfinite(r["kappa_g"])]
        lo, hi = (float(x) for x in cfg.hand_depth.kappa_range)
        inside = sum(lo <= k <= hi for k in kappas)
        print(f"release holdout median height {med_h:.2f} cm, 3D {med_3:.2f} cm")
        print(f"kappa_g inside [{lo}, {hi}]: {inside}/{len(kappas)}")
        if med_h > 1.5 or med_3 > 4.0:
            print("WARNING: depth check 3a did not pass. Results below use this depth anyway.")
        if len(kappas) and inside / len(kappas) < 0.9:
            print("WARNING: depth check 3b did not pass (kappa_g outside the band).")
    print(f"wrote {dest}")


def stage_qc(cfg) -> None:
    qc_dir = ensure_dir(_out(cfg) / "qc")
    allow = [str(x) for x in cfg.qc_v2.allow_multi_grasp_labels]
    rows = []
    for path in sorted(_clip_dir(cfg).glob("clip_*.npz")):
        d = _load_clip(path)
        g = np.asarray(d["g"], float)
        p = np.asarray(d["p_ee"], float)
        valid_hand = float(np.isfinite(p).all(axis=1).mean())
        valid_block = float(np.mean(d["valid_obj"]))
        n_g = grasp_count(g)
        grasp_ok = grasp_passes(d["type"], n_g, allow, int(cfg.qc_v2.max_grasps))
        passed = int(
            valid_hand >= float(cfg.human.qc.min_valid_hand)
            and valid_block >= float(cfg.human.qc.min_valid_block)
            and grasp_ok
        )
        rows.append({
            "clip": int(d["clip"]), "type": d["type"], "valid_hand": round(valid_hand, 3),
            "valid_block": round(valid_block, 3), "n_grasp": n_g,
            "depth_ok_frac": round(float(np.mean(d["depth_ok"])), 3),
            "pass": passed,
        })
    dest = qc_dir / "extraction_report.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_pass = sum(r["pass"] for r in rows)
    print(f"QC {n_pass}/{len(rows)}")
    print(f"wrote {dest}")


def _passed(cfg) -> list[int]:
    report = _out(cfg) / "qc" / "extraction_report.csv"
    ids = []
    with report.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] == "success" and int(row["pass"]) == 1:
                ids.append(int(row["clip"]))
    return ids


def _as_clip(d: dict) -> dict:
    return {"t_frame": np.asarray(d["t_frame"], float), "p_ee": np.asarray(d["p_ee"], float), "g": np.asarray(d["g"], float)}


def _hybrid_plan(clip, block_xy, target_xy, h_tgt, cfg, ee_start):
    p_h, g = path_at_robot_rate(clip, cfg)
    k_g, k_r = grasp_window(g)
    half = float(cfg.objects.sim_block_size) / 2.0
    block = np.array([float(block_xy[0]), float(block_xy[1]), half])
    target = np.array([float(target_xy[0]), float(target_xy[1]), float(h_tgt)])
    rv = cfg.retarget_v2
    p_star, grip = hybrid_path(
        p_h, k_g, k_r, block, target, z_top=float(h_tgt), block_half=half,
        fps=float(cfg.human.rate_hz), hover_z=float(rv.hover_z_m), carry_z=float(rv.carry_z_m),
        grasp_z_offset=float(rv.grasp_z_offset_m), release_clearance=float(rv.release_clearance_m),
        descend_s=float(rv.descend_s), close_hold_s=float(rv.close_hold_s), lift_s=float(rv.lift_s),
        lower_s=float(rv.lower_s), open_hold_s=float(rv.open_hold_s),
        degenerate_dist=float(cfg.retarget.degenerate_dist),
        carry_from_hand=str(rv.carry_z_source) == "hand",
    )
    if ee_start is not None and k_g > 0:
        start = np.asarray(ee_start, float).reshape(3)
        for k in range(k_g):
            w = k / float(k_g)
            p_star[k] = (1.0 - w) * start + w * p_star[k]
    alpha, beta = similarity(p_h[k_g, :2], p_h[k_r, :2], block[:2], target[:2], float(cfg.retarget.degenerate_dist))
    return Plan(p_star=p_star, grip=grip, k_close=k_g, k_open=k_r, alpha=alpha, beta=beta)


def stage_e1(cfg) -> None:
    from palm_prior.retarget.build_plan import build_plan
    from palm_prior.sim.env import Env
    from palm_prior.sim.scene import GOAL_BODIES, build_scene
    from replay import _track

    e1 = ensure_dir(_out(cfg) / "e1")
    clips = _passed(cfg)
    if not clips:
        raise SystemExit("no QC-passed success clips")
    loaded = {i: _as_clip(_load_clip(_clip_dir(cfg) / f"clip_{i:03d}.npz")) for i in clips}
    lo, hi = (int(x) for x in cfg.robot_data.eval_seeds)
    seeds = list(range(lo, hi + 1))
    scene = build_scene(cfg)
    env = Env("plate", seeds[0], cfg, scene=scene)
    env.reset()
    ref = build_plan(loaded[clips[0]], env.layout.block_xy, env.layout.target_xy, float(cfg.goals.plate), None, None, cfg)
    naive = (ref.alpha, ref.beta)
    rows = []
    for goal in GOAL_BODIES:
        env.goal = goal
        env.h_tgt = float(cfg.goals[goal])
        for seed in seeds:
            env.rng = np.random.default_rng(seed)
            env.reset()
            clip_id = clips[(seed + list(GOAL_BODIES).index(goal)) % len(clips)]
            clip = loaded[clip_id]
            saved = env.save_state()
            for method in ("naive", "two_anchor", "hybrid"):
                if method == "hybrid":
                    plan = _hybrid_plan(clip, env.layout.block_xy, env.layout.target_xy, env.h_tgt, cfg, env.p_ee_W())
                else:
                    plan = build_plan(
                        clip, env.layout.block_xy, env.layout.target_xy, env.h_tgt,
                        None, None, cfg, ee_start_W=env.p_ee_W(),
                        alpha_beta=naive if method == "naive" else None,
                    )
                _track(env, plan, cfg, record=False)
                ok = int(env.success())
                err = float(np.linalg.norm(env.p_obj_W()[:2] - env.layout.target_xy))
                rows.append({"method": method, "goal": goal, "seed": seed, "clip": clip_id, "success": ok, "xy_err_m": round(err, 4)})
                env.restore_state(saved)
            if seed % 10 == 0:
                print(f"  {goal} seed {seed}")
    env.close()
    dest = e1 / "e1.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "goal", "seed", "clip", "success", "xy_err_m"])
        writer.writeheader()
        writer.writerows(rows)
    for method in ("naive", "two_anchor", "hybrid"):
        part = [r for r in rows if r["method"] == method]
        print(f"{method:11s} {sum(r['success'] for r in part)}/{len(part)}")
    print(f"wrote {dest}")


def stage_robot(cfg) -> None:
    from palm_prior.sim.env import Env
    from palm_prior.sim.scene import GOAL_BODIES, build_scene
    from replay import _track

    robot = ensure_dir(_out(cfg) / "robot_data")
    clips = _passed(cfg)
    loaded = {i: _as_clip(_load_clip(_clip_dir(cfg) / f"clip_{i:03d}.npz")) for i in clips}
    rd = cfg.robot_data
    jobs = [(int(rd.heldout_seed_start) + i, "heldout") for i in range(int(rd.n_heldout_episodes))]
    jobs += [(int(rd.train_seed_start) + i, "train") for i in range(int(rd.n_train_episodes))]
    scene = build_scene(cfg)
    env = Env("plate", 0, cfg, scene=scene)
    episodes = []
    moved = 0
    for n, (seed, split) in enumerate(jobs):
        rng = np.random.default_rng(seed)
        goal = str(rng.choice(list(GOAL_BODIES)))
        env.goal = goal
        env.h_tgt = float(cfg.goals[goal])
        env.rng = np.random.default_rng(seed)
        env.reset()
        clip_id = int(rng.choice(clips))
        block_xy = env.layout.block_xy.copy()
        if rng.random() < float(rd.failure_prob):
            lo, hi = (float(x) for x in rd.failure_offset)
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            block_xy = block_xy + float(rng.uniform(lo, hi)) * direction
        start = env.p_obj_W().copy()
        plan = _hybrid_plan(loaded[clip_id], block_xy, env.layout.target_xy, env.h_tgt, cfg, env.p_ee_W())
        _frames, rows = _track(env, plan, cfg, record=False)
        end = env.p_obj_W()
        if float(np.linalg.norm(end[:2] - start[:2])) > 0.05:
            moved += 1
        tgt = np.array([env.layout.target_xy[0], env.layout.target_xy[1], env.h_tgt])
        s = np.stack([np.concatenate([r[0], tgt]) for r in rows])
        a = np.stack([r[1] for r in rows])
        dp = np.stack([r[2] for r in rows])
        att = np.array([r[3] for r in rows])
        episodes.append({"s": s, "a": a, "dp_obj": dp, "attach": att, "seed": seed, "goal": goal, "split": split, "clip": clip_id})
        if (n + 1) % 50 == 0:
            print(f"  robot {n + 1}/{len(jobs)}")
    env.close()
    ids = np.concatenate([np.full(len(ep["s"]), ep["seed"]) for ep in episodes])
    goal = np.concatenate([np.full(len(ep["s"]), ep["goal"]) for ep in episodes])
    split = np.array([ep["split"] for ep in episodes])
    np.savez_compressed(
        robot / "episodes.npz",
        s=np.concatenate([ep["s"] for ep in episodes]),
        a=np.concatenate([ep["a"] for ep in episodes]),
        dp_obj=np.concatenate([ep["dp_obj"] for ep in episodes]),
        attach=np.concatenate([ep["attach"] for ep in episodes]),
        episode_id=ids, episode_goal=goal, episode_split=split,
        episode_seed=np.array([ep["seed"] for ep in episodes]),
    )
    print(f"robot episodes with the block moving: {moved}/{len(episodes)} ({100 * moved / len(episodes):.1f}%)")


def _human_transitions(cfg) -> dict:
    parts = []
    for path in sorted(_clip_dir(cfg).glob("clip_*.npz")):
        d = _load_clip(path)
        tr = clip_transitions(
            np.asarray(d["t_frame"], float), np.asarray(d["p_ee"], float), np.asarray(d["p_obj"], float),
            np.asarray(d["g"], float), np.asarray(d["t_h"], float)[:2],
            float(cfg.goals[d["goal"]]), float(cfg.objects.block_size), float(cfg.human.attach_margin),
            float(d["t_frame"][-1]), float(cfg.human.rate_hz), float(cfg.human.tau),
        )
        if len(tr.s) == 0:
            continue
        tgt = np.repeat(np.asarray(d["t_h"], float).reshape(1, 3), len(tr.s), axis=0)
        # Rebuild so the target xyz matches t_h, not a second copy of the 8-D target.
        s = np.concatenate([tr.s, tgt], axis=1)
        assert s.shape[1] == 11
        parts.append((s, tr.a, tr.dp_obj, tr.attach, np.full(len(s), int(d["clip"])), d["type"] != "success"))
    s = np.concatenate([p[0] for p in parts])
    return {
        "s": s, "a": np.concatenate([p[1] for p in parts]),
        "dp_obj": np.concatenate([p[2] for p in parts]),
        "attach": np.concatenate([p[3] for p in parts]),
        "clip_id": np.concatenate([p[4] for p in parts]),
        "is_failure": np.concatenate([np.full(len(p[0]), p[5]) for p in parts]),
    }


def stage_e2(cfg) -> None:
    import torch
    from exp_e2 import _heldout_clips, _never_moves, _take
    from palm_prior.wm.model import Ensemble, fit_stats
    from palm_prior.wm.train import evaluate, train_ensemble, windows_from_ids

    e2 = ensure_dir(_out(cfg) / "e2")
    human = _human_transitions(cfg)
    robot = np.load(_out(cfg) / "robot_data" / "episodes.npz")
    train_mask = robot["episode_split"] == "train"
    # episode_split is per episode; transitions use episode_id. Rebuild a per-row split.
    split_of = {int(s): sp for s, sp in zip(robot["episode_seed"], robot["episode_split"])}
    row_split = np.array([split_of[int(i)] for i in robot["episode_id"]])
    row_train = row_split == "train"
    held_robot = (robot["s"][~row_train], robot["a"][~row_train], robot["dp_obj"][~row_train], robot["attach"][~row_train], robot["episode_id"][~row_train])
    order = np.argsort(robot["episode_seed"])
    train_seeds = [int(s) for s, sp in zip(robot["episode_seed"], robot["episode_split"]) if sp == "train"]
    ns = [int(x) for x in cfg.robot_data.subsets]
    seeds = [0, 1, 2]
    w = cfg.wm
    dest = e2 / "e2.csv"
    if dest.exists():
        dest.unlink()
    baseline = _never_moves(held_robot[2], held_robot[4], int(w.eval_horizon))
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "N", "seed", "pos_err_cm", "attach_auroc"])
        writer.writeheader()
        for variant in ("sim_only", "pretrain_ft", "cotrain"):
            for n in ns:
                for seed in seeds:
                    torch.manual_seed(seed)
                    model = Ensemble(int(w.n_members), tuple(int(h) for h in w.hidden), state_dim=11)
                    if n == 0 and variant == "sim_only":
                        pos, auroc = baseline, 0.5
                    else:
                        use = train_seeds[:n]
                        mask = np.isin(robot["episode_id"], use) & row_train
                        robot_part = (robot["s"][mask], robot["a"][mask], robot["dp_obj"][mask], robot["attach"][mask], robot["episode_id"][mask])
                        hold = _heldout_clips(human["clip_id"], human["is_failure"], int(cfg.human.n_heldout_clips), seed)
                        keep = np.array([int(c) not in hold for c in human["clip_id"]])
                        human_part = _take(human["s"], human["a"], human["dp_obj"], human["attach"], human["clip_id"], keep)
                        if variant == "sim_only":
                            stats = fit_stats(robot_part[0], robot_part[1], robot_part[2])
                            pools = [(windows_from_ids(*robot_part[:4], robot_part[4], int(w.rollout_len)), 1.0, 1.0)]
                            steps, lr = int(w.finetune_steps), float(w.lr)
                        elif variant == "pretrain_ft":
                            stats = fit_stats(human_part[0], human_part[1], human_part[2])
                            pools = [(windows_from_ids(*human_part[:4], human_part[4], int(w.rollout_len)), 0.0, 1.0)]
                            train_ensemble(
                                model, pools, int(w.pretrain_steps), float(w.lr), int(w.batch_size),
                                int(w.rollout_len), float(w.attach_weight), stats, seed,
                            )
                            if n == 0:
                                pos, auroc, _ = evaluate(model, *held_robot, int(w.eval_horizon), stats, embodiment=1.0)
                                writer.writerow({"variant": variant, "N": n, "seed": seed, "pos_err_cm": round(pos, 4), "attach_auroc": round(auroc, 4)})
                                print(f"{variant} N={n} seed={seed}  {pos:.2f} cm  auroc {auroc:.3f}")
                                continue
                            stats = fit_stats(
                                np.concatenate([human_part[0], robot_part[0]]),
                                np.concatenate([human_part[1], robot_part[1]]),
                                np.concatenate([human_part[2], robot_part[2]]),
                            )
                            pools = [
                                (windows_from_ids(*human_part[:4], human_part[4], int(w.rollout_len)), 0.0, float(w.human_batch_frac)),
                                (windows_from_ids(*robot_part[:4], robot_part[4], int(w.rollout_len)), 1.0, 1.0 - float(w.human_batch_frac)),
                            ]
                            steps, lr = int(w.finetune_steps), float(w.lr) * float(w.finetune_lr_factor)
                        else:
                            stats = fit_stats(
                                np.concatenate([human_part[0], robot_part[0]]) if n else human_part[0],
                                np.concatenate([human_part[1], robot_part[1]]) if n else human_part[1],
                                np.concatenate([human_part[2], robot_part[2]]) if n else human_part[2],
                            )
                            pools = [(windows_from_ids(*human_part[:4], human_part[4], int(w.rollout_len)), 0.0, float(w.human_batch_frac) if n else 1.0)]
                            if n:
                                pools.append((windows_from_ids(*robot_part[:4], robot_part[4], int(w.rollout_len)), 1.0, 1.0 - float(w.human_batch_frac)))
                            steps, lr = int(w.pretrain_steps), float(w.lr)
                        train_ensemble(
                            model, pools, steps, lr, int(w.batch_size),
                            int(w.rollout_len), float(w.attach_weight), stats, seed,
                        )
                        pos, auroc, _ = evaluate(model, *held_robot, int(w.eval_horizon), stats, embodiment=1.0)
                    writer.writerow({"variant": variant, "N": n, "seed": seed, "pos_err_cm": round(float(pos), 4), "attach_auroc": round(float(auroc), 4)})
                    f.flush()
                    print(f"{variant} N={n} seed={seed}  {float(pos):.2f} cm  auroc {float(auroc):.3f}")
    print(f"wrote {dest}  never-moves {baseline:.2f} cm")


def plot_clip_height(clip_path: Path, dest: Path, label: str = "Demo 2") -> None:
    """Height of one clip. Gray is the old pinch, with no measured palm. Blue uses it.

    p_ee_old[:, 2] and p_ee[:, 2] are metres in the table frame. The figure is centimetres.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(clip_path, allow_pickle=False)
    t = np.asarray(data["t_frame"], float)
    old = np.asarray(data["p_ee_old"], float)
    new = np.asarray(data["p_ee"], float)
    assert old.ndim == 2 and old.shape[1] == 3 and old.shape == new.shape
    assert t.shape == (old.shape[0],)
    kg, kr = int(data["k_g"]), int(data["k_r"])
    old_cm = old[:, 2] * 100.0
    new_cm = new[:, 2] * 100.0
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(t, old_cm, color="0.55", lw=1.15, label="without palm size (the problem)")
    ax.plot(t, new_cm, color="#1f77b4", lw=2.1, label="with palm size (the fix)")
    ax.axvline(float(t[kg]), color="#2ca02c", ls="--", lw=1.2, label="grasp")
    ax.axvline(float(t[kr]), color="#d62728", ls="--", lw=1.2, label="release")
    ax.set_title(
        f"{label}. Gray never measured the hand, so the height is not a reach.\n"
        f"It runs from {float(np.nanmin(old_cm)):.0f} cm to {float(np.nanmax(old_cm)):.0f} cm. "
        f"Palm size stays between {float(np.nanmin(new_cm)):.0f} cm and {float(np.nanmax(new_cm)):.0f} cm."
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height above the table (cm)")
    # Both spikes are the problem. Keep the legend under the axes so they stay visible.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")


def stage_figures(cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clip2 = _out(cfg) / "clips" / "clip_002.npz"
    if clip2.exists():
        plot_clip_height(clip2, _out(cfg) / "depth" / "plots" / "clip002_height.png", "Demo 2")
    e2_csv = _out(cfg) / "e2" / "e2.csv"
    if not e2_csv.exists():
        print("no e2 csv, skip figures")
        return
    rows = list(csv.DictReader(e2_csv.open(encoding="utf-8")))
    plots = ensure_dir(_out(cfg) / "e2" / "plots")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colours = {"sim_only": "C1", "pretrain_ft": "C0", "cotrain": "C2"}
    labels = {"sim_only": "robot only", "pretrain_ft": "phone clips, then robot", "cotrain": "both from the start"}
    ns = sorted({int(float(r["N"])) for r in rows})
    xs = [n if n > 0 else 0.4 for n in ns]
    for variant, colour in colours.items():
        for ax, key in ((axes[0], "pos_err_cm"), (axes[1], "attach_auroc")):
            mean, std = [], []
            for n in ns:
                vals = [float(r[key]) for r in rows if r["variant"] == variant and int(float(r["N"])) == n]
                mean.append(float(np.mean(vals)))
                std.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            mean_a, std_a = np.array(mean), np.array(std)
            ax.plot(xs, mean_a, color=colour, marker="o", label=labels[variant])
            ax.fill_between(xs, mean_a - std_a, mean_a + std_a, color=colour, alpha=0.15)
    if rows:
        base = float(np.mean([float(r["pos_err_cm"]) for r in rows if r["variant"] == "sim_only" and int(float(r["N"])) == 0]))
        axes[0].axhline(base, color="0.4", ls="--", label="lid never moves")
    axes[1].axhline(0.5, color="0.4", ls="--", label="coin toss")
    for ax, ylab, title in (
        (axes[0], "10-step block error (cm)", "Where the lid ends up"),
        (axes[1], "attach AUROC", "Is the lid in the hand?"),
    ):
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("robot episodes")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {plots / 'learning_curves.png'}")

    e1_csv = _out(cfg) / "e1" / "e1.csv"
    if e1_csv.exists():
        e1_rows = list(csv.DictReader(e1_csv.open(encoding="utf-8")))
        plots1 = ensure_dir(_out(cfg) / "e1" / "plots")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        methods = ["naive", "two_anchor", "hybrid"]
        goals = ["plate", "pad", "box"]
        width = 0.25
        for i, method in enumerate(methods):
            rates = []
            for goal in goals:
                part = [r for r in e1_rows if r["method"] == method and r["goal"] == goal]
                rates.append(100.0 * np.mean([int(r["success"]) for r in part]) if part else 0)
            ax.bar(np.arange(3) + (i - 1) * width, rates, width, label=method)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(goals)
        ax.set_ylabel("success (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots1 / "success_by_goal.png", dpi=150)
        plt.close(fig)
        print(f"wrote {plots1 / 'success_by_goal.png'}")


def stage_readme(cfg) -> None:
    """Write the numeric tables. Prose that needs the run stays short and cites the csv."""
    v2 = _out(cfg)
    lines = ["# Results v2", "", "Numbers are read from the csv next to each table.", ""]
    qc = v2 / "qc" / "extraction_report.csv"
    if qc.exists():
        rows = list(csv.DictReader(qc.open(encoding="utf-8")))
        n_pass = sum(int(r["pass"]) for r in rows)
        lines += ["## QC", "", f"Source: `results/v2/qc/extraction_report.csv`. {n_pass}/{len(rows)} passed.", ""]
    e1 = v2 / "e1" / "e1.csv"
    if e1.exists():
        rows = list(csv.DictReader(e1.open(encoding="utf-8")))
        lines += ["## E1", "", "Source: `results/v2/e1/e1.csv`.", "", "| Method | successes |", "|---|---:|"]
        for method in ("naive", "two_anchor", "hybrid"):
            part = [r for r in rows if r["method"] == method]
            lines.append(f"| {method} | {sum(int(r['success']) for r in part)}/{len(part)} |")
        lines += ["", "![Success by goal](results/v2/e1/plots/success_by_goal.png)", ""]
    e2 = v2 / "e2" / "e2.csv"
    if e2.exists():
        rows = list(csv.DictReader(e2.open(encoding="utf-8")))
        lines += ["## E2", "", "Source: `results/v2/e2/e2.csv`. Mean over 3 seeds.", "",
                  "| N | robot only cm | phone then robot cm | both cm | phone AUROC |", "|---:|---:|---:|---:|---:|"]
        ns = sorted({int(float(r["N"])) for r in rows})
        for n in ns:
            def cell(variant, key):
                vals = [float(r[key]) for r in rows if r["variant"] == variant and int(float(r["N"])) == n]
                return float(np.mean(vals)) if vals else float("nan")
            lines.append(
                f"| {n} | {cell('sim_only','pos_err_cm'):.2f} | {cell('pretrain_ft','pos_err_cm'):.2f} | "
                f"{cell('cotrain','pos_err_cm'):.2f} | {cell('pretrain_ft','attach_auroc'):.2f} |"
            )
        lines += ["", "![Learning curves](results/v2/e2/plots/learning_curves.png)", ""]
    dest = v2 / "auto_tables.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    print("RESULTS.md is the written account and was left in place")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="check_inputs")
    parser.add_argument("--only", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-e3", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config([])
    set_seed(int(cfg.seed))
    if args.only:
        todo = [args.only]
    else:
        if args.start not in STAGES:
            raise SystemExit(f"unknown stage {args.start}")
        todo = STAGES[STAGES.index(args.start):]
    with Timer("fix_all"):
        for stage in todo:
            print(f"\n== {stage} ==")
            if stage == "check_inputs":
                _need(cfg)
                print("measurements and intrinsics are present")
            elif stage == "discover":
                missing = [row["clip"] for row in _log(cfg) if not _video(cfg, row["clip"]).exists()]
                if missing:
                    raise SystemExit(f"missing videos for clips {missing}")
                print(f"{len(_log(cfg))} clips in the log, videos present")
            elif stage == "extract":
                stage_extract(cfg, args.force)
            elif stage in ("objects", "anchor"):
                print("done inside extract")
            elif stage == "validate_depth":
                stage_validate(cfg)
            elif stage == "qc":
                stage_qc(cfg)
            elif stage == "e1":
                stage_e1(cfg)
            elif stage == "robot_data":
                stage_robot(cfg)
            elif stage == "e2":
                stage_e2(cfg)
            elif stage == "figures":
                stage_figures(cfg)
            elif stage == "readme":
                stage_readme(cfg)
            elif stage == "e3":
                print("E3 is off. Pass --with-e3 after E2 if you want it.")
    if args.with_e3:
        print("E3 is not run in this pass. The open-loop numbers are in results/v2/.")


if __name__ == "__main__":
    main()
