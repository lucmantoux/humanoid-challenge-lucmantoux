"""Phase 1: overlay hand landmarks and the block mask on a clip.

On the first run it opens the first frame of empty.mp4 and asks you to click the block
once; that fits the HSV range and saves it, so later runs are non-interactive.

Usage:
    uv run python scripts/preview_hand.py clip=1
    uv run python scripts/preview_hand.py clip=1 overwrite=true refit_color=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401  puts src/ on sys.path

from palm_prior.perception import block as blockmod
from palm_prior.perception.hand import (
    CONNECTIONS,
    INDEX_TIP,
    THUMB_TIP,
    aperture,
    detect,
    ensure_model,
    open_landmarker,
    pinch_px,
)
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


def click_block(frame_bgr: np.ndarray) -> tuple[int, int]:
    """Show a frame and return the pixel the user clicks on. Needs a desktop session."""
    clicked: list[tuple[int, int]] = []
    window = "click the centre of the block, then press any key"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            marked = frame_bgr.copy()
            cv2.circle(marked, (x, y), 8, (0, 0, 255), 2)
            cv2.imshow(window, marked)

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, frame_bgr)
        cv2.setMouseCallback(window, on_mouse)
        while not clicked:
            if cv2.waitKey(50) == 27:
                break
        cv2.waitKey(300)
        cv2.destroyWindow(window)
    except cv2.error as exc:
        raise RuntimeError(
            "could not open a window to click the block. Run this on a desktop session, "
            "or write the HSV range into the file at paths.block_hsv by hand."
        ) from exc
    if not clicked:
        raise RuntimeError("no click received")
    return clicked[0]


def block_hsv_range(cfg) -> blockmod.HsvRange:
    """Load the fitted block colour, fitting it from empty.mp4 the first time."""
    path = resolve(cfg.paths.block_hsv)
    if path.exists() and not bool(cfg.get("refit_color", False)):
        return blockmod.load_hsv(path)

    empty = resolve(cfg.paths.raw_dir) / "empty.mp4"
    if not empty.exists():
        raise FileNotFoundError(f"{empty} not found — record it first (FILMING_GUIDE Step 6)")
    _, frame = next(video_frames(empty))
    u, v = click_block(frame)
    half = int(cfg.block_color.roi_half)
    h, w = frame.shape[:2]
    x0, y0 = max(0, u - half), max(0, v - half)
    roi = (x0, y0, min(w - x0, 2 * half), min(h - y0, 2 * half))
    hsv = blockmod.fit_hsv(frame, roi, cfg)
    blockmod.save_hsv(path, hsv)
    print(f"fitted block HSV lo={np.round(hsv.lo, 1)} hi={np.round(hsv.hi, 1)} -> {path}")
    return hsv


def clip_path(cfg) -> Path:
    if cfg.get("video"):
        return resolve(str(cfg.video))
    if cfg.get("clip") is None:
        raise SystemExit("pass clip=<n> or video=<path>")
    return resolve(cfg.paths.raw_dir) / f"demo_{int(cfg.clip):03d}.mp4"


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    video = clip_path(cfg)
    out_mp4 = resolve(cfg.paths.results) / f"preview_hand_{video.stem}.mp4"
    if should_skip(out_mp4, bool(cfg.get("overwrite", False))):
        return
    if not video.exists():
        raise FileNotFoundError(f"{video} not found")

    hsv = block_hsv_range(cfg)
    model = ensure_model(resolve(cfg.paths.hand_model))
    info = video_info(video)
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )

    n_hand = n_block = n_frames = 0
    landmarker = open_landmarker(model)
    with Timer("preview_hand"):
        try:
            for i, frame in video_frames(video):
                n_frames += 1
                vis = frame.copy()

                blob = blockmod.mask(frame, hsv, cfg)
                if blob.centroid is not None:
                    n_block += 1
                    contours, _ = cv2.findContours(
                        blob.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, (255, 128, 0), 2)
                    cu, cv_ = blob.centroid
                    cv2.drawMarker(vis, (int(cu), int(cv_)), (255, 128, 0), cv2.MARKER_CROSS, 18, 2)

                timestamp_ms = int(round(1000.0 * i / info.fps))
                hand = detect(landmarker, frame, timestamp_ms)
                if hand is not None:
                    n_hand += 1
                    for a, b in CONNECTIONS:
                        pa = tuple(int(x) for x in hand.px[a])
                        pb = tuple(int(x) for x in hand.px[b])
                        cv2.line(vis, pa, pb, (200, 200, 200), 2)
                    for j, (x, y) in enumerate(hand.px):
                        colour = (0, 255, 255) if j in (THUMB_TIP, INDEX_TIP) else (180, 180, 180)
                        cv2.circle(vis, (int(x), int(y)), 5 if j in (THUMB_TIP, INDEX_TIP) else 3, colour, -1)
                    p = pinch_px(hand)
                    cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 255, 0), -1)
                    text = f"frame {i}  aperture {aperture(hand):.2f}"
                else:
                    text = f"frame {i}  NO HAND"
                cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if blob.centroid is None:
                    cv2.putText(
                        vis, "NO BLOCK", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                    )
                writer.write(vis)
        finally:
            writer.release()
            landmarker.close()

    print(f"hand found in {n_hand}/{n_frames} frames ({100 * n_hand / max(n_frames, 1):.1f}%)")
    print(f"block found in {n_block}/{n_frames} frames ({100 * n_block / max(n_frames, 1):.1f}%)")
    print(f"wrote {out_mp4}")
    print("Watch for: green pinch dot between the fingertips, and an orange contour that "
          "hugs the block only — including while you hold it.")


if __name__ == "__main__":
    main(sys.argv[1:])
