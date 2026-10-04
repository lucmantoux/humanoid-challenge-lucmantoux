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
