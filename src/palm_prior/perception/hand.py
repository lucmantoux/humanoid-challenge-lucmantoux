"""MediaPipe hand landmarks (EXPLAINER Module 3, detection half).

The Tasks API returns two things per frame: 21 landmarks in image coordinates, and the
same 21 in metres but scaled to an average hand. Module 3 feeds both to PnP.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe landmark indices used throughout: thumb tip, index tip, wrist, middle MCP.
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9

# The 21-landmark skeleton, for drawing.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandFrame(NamedTuple):
    """px (21, 2) image landmarks in pixels; world (21, 3) in metres, average-hand scale."""

    px: np.ndarray
    world: np.ndarray


def ensure_model(path: str | Path) -> Path:
    """Download the hand_landmarker task file on first use. Returns the local path."""
    p = Path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {p}")
    urllib.request.urlretrieve(MODEL_URL, p)
    return p


def open_landmarker(model_path: str | Path, num_hands: int = 1) -> vision.HandLandmarker:
    """Create a VIDEO-mode landmarker. Call .close() when done, or use it in a `with`."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(
    landmarker: vision.HandLandmarker, frame_bgr: np.ndarray, timestamp_ms: int
) -> HandFrame | None:
    """Run the landmarker on one frame. Returns None when no hand is found.

    Timestamps must increase strictly between calls in VIDEO mode.
    """
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, frame_bgr.shape
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(timestamp_ms)
    )
    if not result.hand_landmarks:
        return None
    px = np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]], float)
    world = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]], float)
    assert px.shape == (21, 2) and world.shape == (21, 3), (px.shape, world.shape)
    return HandFrame(px=px, world=world)


def pinch_px(hand: HandFrame) -> np.ndarray:
    """Midpoint of the thumb and index tips, in pixels. (2,)"""
    return 0.5 * (hand.px[THUMB_TIP] + hand.px[INDEX_TIP])


def aperture(hand: HandFrame) -> float:
    """Pinch size divided by hand size, so camera distance cancels (EXPLAINER §7).

    a = ||l4 - l8|| / ||l0 - l9||
    """
    pinch = np.linalg.norm(hand.px[THUMB_TIP] - hand.px[INDEX_TIP])
    scale = np.linalg.norm(hand.px[WRIST] - hand.px[MIDDLE_MCP])
    assert scale > 0, "degenerate hand landmarks"
    return float(pinch / scale)


class PnPPinch(NamedTuple):
    """One frame of solvePnP on the 21 world landmarks.

    R (3, 3), t (3,) take a world-landmark point into the camera: X_C = R X + t.
    z_pnp is the camera-frame depth of the pinch, before the rest-pose scale k.
    reproj is the pixel error of that pinch point.
    """

    R: np.ndarray
    t: np.ndarray
    z_pnp: float
    reproj: float


def solve_pinch(hand: HandFrame, K: np.ndarray, dist: np.ndarray) -> PnPPinch | None:
    """PnP of the 21 landmarks. None when the pose is behind the camera or the fit fails.

    K (3, 3), dist (n,). The world landmarks are in metres at an average-hand scale,
    so the depth this returns is Z_pnp in EXPLAINER §7, not yet the true depth.
    """
    assert K.shape == (3, 3), K.shape
    obj = np.ascontiguousarray(hand.world, np.float64)
    img = np.ascontiguousarray(hand.px, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    pinch_w = 0.5 * (hand.world[THUMB_TIP] + hand.world[INDEX_TIP])
    p_C = R @ pinch_w + t
    if p_C[2] <= 1e-6:
        return None
    proj, _ = cv2.projectPoints(pinch_w.reshape(1, 1, 3), rvec, t.reshape(3, 1), K, dist)
    reproj = float(np.linalg.norm(proj.reshape(2) - pinch_px(hand)))
    return PnPPinch(R=R, t=t, z_pnp=float(p_C[2]), reproj=reproj)


def rest_scale(z_plane: np.ndarray, z_pnp: np.ndarray) -> float:
    """k = median over rest frames of Z_plane / Z_pnp (EXPLAINER §7).

    Both arrays are (N,). Non-finite samples, and non-positive PnP depths, are ignored.
    """
    z_plane = np.asarray(z_plane, float)
    z_pnp = np.asarray(z_pnp, float)
    assert z_plane.shape == z_pnp.shape, (z_plane.shape, z_pnp.shape)
    ok = np.isfinite(z_plane) & np.isfinite(z_pnp) & (z_pnp > 1e-6) & (z_plane > 1e-6)
    if int(ok.sum()) == 0:
        return float("nan")
    return float(np.median(z_plane[ok] / z_pnp[ok]))


def pinch_camera(pnp: PnPPinch, world_pinch: np.ndarray, k: float) -> np.ndarray:
    """True pinch position in the camera frame, p_C = k (R X + t). (3,)"""
    assert world_pinch.shape == (3,), world_pinch.shape
    return float(k) * (pnp.R @ world_pinch + pnp.t)


def gripper_signal(aperture_series: np.ndarray, cfg, rest: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Per-frame gripper g in {0, 1}.

    aperture_series (N,); missing frames are NaN and stay NaN. The clip starts open.

    EXPLAINER §7 closes when the aperture is small. That is right when the resting
    hand is open, so a pinch is the low value. These clips rest with the fingers
    together, and the thumb and index spread to get around the lid, so the rest
    pose is the low value and the grasp is the departure from it. When `rest` is
    given (a bool mask of the still seconds), the open pose is the median aperture
    there, and the same 0.3 / 0.6 fractions of the clip's aperture span are applied
    to the distance from that rest value: far from rest closes, near rest opens.

    Returns g (N,), and the two distance thresholds (close_above, open_below).
    """
    a = np.asarray(aperture_series, float)
    assert a.ndim == 1, a.shape
    known = a[np.isfinite(a)]
    assert len(known) > 0, "no aperture samples"
    h = cfg.hand
    a_lo = float(np.percentile(known, float(h.aperture_lo_pct)))
    a_hi = float(np.percentile(known, float(h.aperture_hi_pct)))
    span = max(a_hi - a_lo, 1e-6)
    if rest is None:
        close_thr = a_lo + float(h.close_frac) * span
        open_thr = a_lo + float(h.open_frac) * span
        g = np.full(len(a), np.nan)
        state = 0.0
        for i, ai in enumerate(a):
            if not np.isfinite(ai):
                continue
            if ai < close_thr:
                state = 1.0
            elif ai > open_thr:
                state = 0.0
            g[i] = state
        return g, close_thr, open_thr

    rest = np.asarray(rest, bool)
    assert rest.shape == a.shape, (rest.shape, a.shape)
    rest_a = a[rest & np.isfinite(a)]
    assert len(rest_a) > 0, "no rest frames for the open aperture"
    a_rest = float(np.median(rest_a))
    close_above = float(h.open_frac) * span
    open_below = float(h.close_frac) * span
    g = np.full(len(a), np.nan)
    state = 0.0
    for i, ai in enumerate(a):
        if not np.isfinite(ai):
            continue
        dev = abs(ai - a_rest)
        if dev > close_above:
            state = 1.0
        elif dev < open_below:
            state = 0.0
        g[i] = state
    return g, close_above, open_below


def fill_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-fill interior NaN runs of length <= max_gap. Longer runs stay NaN.

    values (N,) or (N, D). A row is missing when any component is non-finite.
    Ends are not extrapolated.
    """
    assert max_gap >= 0, max_gap
    src = np.asarray(values, float)
    squeeze = src.ndim == 1
    out = src.reshape(len(src), -1).copy() if squeeze else src.copy()
    assert out.ndim == 2, out.shape
    valid = np.isfinite(out).all(axis=1)
    n = len(out)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            weight = np.linspace(0.0, 1.0, j - i + 2)[1:-1, None]
            out[i:j] = (1.0 - weight) * out[i - 1] + weight * out[j]
        i = j
    return out[:, 0] if squeeze else out


def smooth_series(values: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay along time, separately on each finite run long enough for the window.

    values (N,) or (N, D). NaN rows are left as NaN, so a long detection gap is not
    smeared into the samples on either side.
    """
    from scipy.signal import savgol_filter

    assert window % 2 == 1 and window > order >= 0, (window, order)
    src = np.asarray(values, float)
    squeeze = src.ndim == 1
    out = src.reshape(len(src), -1).copy() if squeeze else src.copy()
    valid = np.isfinite(out).all(axis=1)
    n = len(out)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        length = j - i
        if length >= window:
            out[i:j] = savgol_filter(out[i:j], window, order, axis=0, mode="interp")
        elif length > order + 1 and length % 2 == 1:
            out[i:j] = savgol_filter(out[i:j], length, order, axis=0, mode="interp")
        i = j
    return out[:, 0] if squeeze else out
