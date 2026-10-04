"""Finding the block by its colour (EXPLAINER Module 4, first half).

The block is the only saturated blue or green object in the frame, so an HSV threshold
plus "take the largest blob" is enough. The range is fitted once from a patch of
empty.mp4, which removes the guesswork about your particular block and lighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from omegaconf import OmegaConf


class HsvRange(NamedTuple):
    """Inclusive OpenCV HSV bounds: hue 0-179, saturation and value 0-255."""

    lo: np.ndarray  # (3,) uint8-range floats
    hi: np.ndarray  # (3,)


class Blob(NamedTuple):
    """centroid (u, v) in pixels, area in pixels, and the cleaned binary mask."""

    centroid: tuple[float, float] | None
    area: float
    mask: np.ndarray


def fit_hsv(frame_bgr: np.ndarray, roi: tuple[int, int, int, int], cfg) -> HsvRange:
    """Fit an HSV range from a patch of pixels known to be on the block.

    frame_bgr (H, W, 3); roi (x, y, w, h) in pixels.

    Only the hue is fitted. Hue is the channel that survives a change of lighting, which
    is the whole reason for working in HSV; saturation and value drop sharply whenever
    the hand shadows the block, so pinning them to one patch would lose the block exactly
    when it is being grasped. They get fixed floors instead, high enough to exclude the
    grey table and the dark gaps between fingers.

    Percentiles rather than min/max so a few stray edge pixels cannot widen the hue band.
    """
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, frame_bgr.shape
    x, y, w, h = (int(v) for v in roi)
    assert w > 0 and h > 0, roi
    patch = cv2.cvtColor(frame_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    assert len(patch) > 0, "empty roi"

    c = cfg.block_color
    hue_lo = float(np.percentile(patch[:, 0], float(c.pct_lo))) - float(c.hue_margin)
    hue_hi = float(np.percentile(patch[:, 0], float(c.pct_hi))) + float(c.hue_margin)
    if hue_lo < 0 or hue_hi > 179:
        raise ValueError(
            f"the block's hue band [{hue_lo:.0f}, {hue_hi:.0f}] runs off the end of the "
            "hue circle, which a single inRange cannot express — this happens for red. "
            "Use a blue or green block as the filming guide asks."
        )

    return HsvRange(
        lo=np.array([hue_lo, float(c.sat_min), float(c.val_min)]),
        hi=np.array([hue_hi, 255.0, 255.0]),
    )


def mask(frame_bgr: np.ndarray, hsv: HsvRange, cfg) -> Blob:
    """Threshold, clean up, and return the largest blob's centroid and area.

    frame_bgr (H, W, 3). The centroid is None when nothing survives the area floor,
    which is how occlusion is detected upstream.
    """
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, frame_bgr.shape
    c = cfg.block_color
    hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv_img, hsv.lo.astype(np.uint8), hsv.hi.astype(np.uint8))

    k = int(c.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Blob(centroid=None, area=0.0, mask=cleaned)

    biggest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(biggest))
    if area < float(c.min_area_px):
        return Blob(centroid=None, area=area, mask=cleaned)

    m = cv2.moments(biggest)
    if m["m00"] <= 0:
        return Blob(centroid=None, area=area, mask=cleaned)
    only_largest = np.zeros_like(cleaned)
    cv2.drawContours(only_largest, [biggest], -1, 255, cv2.FILLED)
    return Blob(centroid=(m["m10"] / m["m00"], m["m01"] / m["m00"]), area=area, mask=only_largest)


def save_hsv(path: str | Path, hsv: HsvRange) -> None:
    """Write the fitted range so later stages never ask for a click again."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"lo": hsv.lo.tolist(), "hi": hsv.hi.tolist()}), p)


def load_hsv(path: str | Path) -> HsvRange:
    cfg = OmegaConf.load(Path(path))
    return HsvRange(lo=np.array(cfg.lo, float), hi=np.array(cfg.hi, float))
