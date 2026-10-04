"""Phase 1: draw the table frame on a clip and measure how still it is.

The jitter number is the check that matters: the camera is on a tripod, so the projected
table-frame origin should not move. If it does, video stabilisation is warping the image
and the fixed-intrinsics assumption of EXPLAINER §7 is broken.

Usage:
    uv run python scripts/debug_aruco.py clip=1
    uv run python scripts/debug_aruco.py video=data/raw/empty.mp4 overwrite=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

import _bootstrap  # noqa: F401  puts src/ on sys.path

from palm_prior.perception.aruco import board_pose, build_board, make_detector, project
from palm_prior.utils import (
    Timer,
    ensure_dir,
    load_config,
    resolve,
    set_seed,
    should_skip,
    video_frames,
    video_info,
)


def clip_path(cfg) -> Path:
    """Resolve video=<path> or clip=<n> into a file under data/raw/."""
    if cfg.get("video"):
        return resolve(str(cfg.video))
    if cfg.get("clip") is None:
        raise SystemExit("pass clip=<n> or video=<path>")
    return resolve(cfg.paths.raw_dir) / f"demo_{int(cfg.clip):03d}.mp4"


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    video = clip_path(cfg)
    out_mp4 = resolve(cfg.paths.results) / f"debug_aruco_{video.stem}.mp4"
    if should_skip(out_mp4, bool(cfg.get("overwrite", False))):
        return
    if not video.exists():
        raise FileNotFoundError(f"{video} not found")

    cam = OmegaConf.load(resolve(cfg.paths.camera_yaml))
    K = np.array(cam.K, float)
    dist = np.array(cam.dist, float)

    tb = build_board(cfg)
    detector = make_detector(tb)
    info = video_info(video)
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )

    axis_len = float(tb.size[0])
    origins: list[np.ndarray] = []
    n_frames = 0
    with Timer("debug_aruco"):
        try:
            for i, frame in video_frames(video):
                n_frames += 1
                pose = board_pose(frame, tb, detector, K, dist, int(cfg.aruco.min_markers))
                vis = frame
                if pose is not None:
                    R_CT, t_CT = pose
                    rvec, _ = cv2.Rodrigues(R_CT)
                    cv2.drawFrameAxes(vis, K, dist, rvec, t_CT, axis_len, 3)
                    uv = project(np.zeros(3), R_CT, t_CT, K, dist)
                    origins.append(uv)
                    cv2.circle(vis, (int(uv[0]), int(uv[1])), 6, (0, 255, 255), -1)
                    label = f"frame {i}  origin ({uv[0]:.1f}, {uv[1]:.1f})"
                    colour = (0, 255, 0)
                else:
                    label = f"frame {i}  BOARD NOT FOUND"
                    colour = (0, 0, 255)
                cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                writer.write(vis)
        finally:
            writer.release()

    detected = len(origins)
    print(f"board detected in {detected}/{n_frames} frames ({100 * detected / max(n_frames, 1):.1f}%)")
    if detected < 2:
        raise RuntimeError("board almost never detected: more light, flatten it, keep it in frame")

    o = np.stack(origins)
    centre = np.median(o, axis=0)
    deviation = np.linalg.norm(o - centre, axis=1)
    jitter = float(deviation.max())
    limit = float(cfg.aruco.max_origin_jitter_px)
    print(f"origin at ({centre[0]:.1f}, {centre[1]:.1f}) px")
    print(f"origin jitter: max {jitter:.2f} px, std {deviation.std():.2f} px (limit {limit})")
    if jitter < limit:
        print(f"PASS: jitter below {limit} px")
    else:
        print(
            f"FAIL: jitter above {limit} px — stabilisation is still on. "
            "See FILMING_GUIDE Step 10."
        )
    print(f"wrote {out_mp4}")


if __name__ == "__main__":
    main(sys.argv[1:])
