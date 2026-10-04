"""Phase 1: camera calibration from calib.mp4 (EXPLAINER Module 1).

Usage:
    uv run python scripts/calibrate.py
    uv run python scripts/calibrate.py overwrite=true calib.n_frames=60
"""

from __future__ import annotations

import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import _bootstrap  # noqa: E402,F401  puts src/ on sys.path

from palm_prior.perception.calibration import (  # noqa: E402
    board_points,
    calibrate,
    detect_corners,
    select_spread,
    view_features,
)
from palm_prior.utils import (  # noqa: E402
    Timer,
    ensure_dir,
    load_config,
    resolve,
    set_seed,
    should_skip,
    video_frames,
    video_info,
)


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    video = resolve(cfg.paths.raw_dir) / "calib.mp4"
    out_yaml = resolve(cfg.paths.camera_yaml)
    out_png = resolve(cfg.paths.results) / "calib_check.png"
    if should_skip(out_yaml, bool(cfg.get("overwrite", False))):
        return
    if not video.exists():
        raise FileNotFoundError(f"{video} not found — record it first (FILMING_GUIDE Step 5)")

    info = video_info(video)
    image_size = (info.width, info.height)
    inner = (int(cfg.calib.checker_inner[0]), int(cfg.calib.checker_inner[1]))
    square = float(cfg.calib.checker_square)
    stride = int(cfg.calib.detect_stride)
    print(f"{video.name}: {info.n_frames} frames, {info.fps:.1f} fps, {info.width}x{info.height}")

    with Timer("calibrate"):
        found_corners: list[np.ndarray] = []
        found_frames: list[np.ndarray] = []
        n_tested = 0
        for _, frame in video_frames(video, stride=stride):
            n_tested += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_corners(gray, inner)
            if corners is not None:
                found_corners.append(corners)
                found_frames.append(frame)
        print(f"checkerboard found in {len(found_corners)}/{n_tested} tested frames")
        if len(found_corners) < 3:
            raise RuntimeError(
                "fewer than 3 usable views: re-record calib.mp4 more slowly, with the "
                "whole board visible and better light (FILMING_GUIDE Step 10)"
            )

        features = np.stack([view_features(c, image_size) for c in found_corners])
        keep = select_spread(features, int(cfg.calib.n_frames))
        print(f"using {len(keep)} well-spread views")

        grid = board_points(inner, square)
        result = calibrate([grid] * len(keep), [found_corners[i] for i in keep], image_size)

        # One mis-detected view can dominate the RMS, so drop the views that disagree
        # with the rest by a wide margin and fit again on what is left.
        limit = float(cfg.calib.outlier_factor) * float(np.median(result.per_view_rms))
        good = [k for k, rms in zip(keep, result.per_view_rms) if rms <= limit]
        if len(good) < len(keep) and len(good) >= 3:
            print(f"dropping {len(keep) - len(good)} views above {limit:.2f} px and refitting")
            keep = good
            result = calibrate([grid] * len(keep), [found_corners[i] for i in keep], image_size)

    fx, fy = result.K[0, 0], result.K[1, 1]
    cx, cy = result.K[0, 2], result.K[1, 2]
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"dist={np.array2string(result.dist, precision=4)}")
    print(f"reprojection RMS = {result.rms:.3f} px (limit {cfg.calib.max_rms_px})")
    verdict = "PASS" if result.rms < float(cfg.calib.max_rms_px) else "FAIL"
    print(f"{verdict}: RMS {'below' if verdict == 'PASS' else 'above'} {cfg.calib.max_rms_px} px")

    ensure_dir(out_yaml.parent)
    OmegaConf.save(
        OmegaConf.create(
            {
                "K": result.K.tolist(),
                "dist": result.dist.tolist(),
                "rms": result.rms,
                "image_size": list(result.image_size),
                "n_views": len(keep),
            }
        ),
        out_yaml,
    )
    print(f"wrote {out_yaml}")

    ensure_dir(out_png.parent)
    show = keep[:: max(1, len(keep) // 6)][:6]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, i in zip(axes.flat[:6], show):
        vis = found_frames[i].copy()
        cv2.drawChessboardCorners(vis, inner, found_corners[i].reshape(-1, 1, 2), True)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"view {i}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(show) : 6]:
        ax.axis("off")

    ax = axes.flat[6]
    ax.bar(range(len(result.per_view_rms)), result.per_view_rms, color="tab:blue")
    ax.axhline(float(cfg.calib.max_rms_px), color="tab:red", lw=1, label=f"{cfg.calib.max_rms_px} px")
    ax.set_xlabel("view")
    ax.set_ylabel("reprojection RMS (px)")
    ax.legend(fontsize=8)

    ax = axes.flat[7]
    for i in keep:
        ax.scatter(found_corners[i][:, 0], found_corners[i][:, 1], s=1, alpha=0.3)
    ax.set_xlim(0, info.width)
    ax.set_ylim(info.height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"corner coverage, overall RMS {result.rms:.3f} px", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main(sys.argv[1:])
