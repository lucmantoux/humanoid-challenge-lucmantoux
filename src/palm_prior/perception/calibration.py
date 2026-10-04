"""Camera calibration from a checkerboard video (EXPLAINER Module 1)."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np


class Calibration(NamedTuple):
    """K (3,3), dist (5,), overall reprojection RMS in px, per-view RMS, image size (w, h)."""

    K: np.ndarray
    dist: np.ndarray
    rms: float
    per_view_rms: np.ndarray
    image_size: tuple[int, int]


def board_points(inner: tuple[int, int], square: float) -> np.ndarray:
    """Checkerboard corners in the board's own frame, z = 0.

    inner (cols, rows) of *inner* corners; square the side length in metres.
    returns (cols*rows, 3) in the order OpenCV returns detections.
    """
    cols, rows = int(inner[0]), int(inner[1])
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def detect_corners(gray: np.ndarray, inner: tuple[int, int]) -> np.ndarray | None:
    """Find the checkerboard in one grayscale frame. Returns (N, 2) float32 or None.

    findChessboardCornersSB is both more robust and already sub-pixel accurate; the older
    detector plus cornerSubPix is the fallback for frames it refuses.
    """
    assert gray.ndim == 2, gray.shape
    size = (int(inner[0]), int(inner[1]))
    ok, corners = cv2.findChessboardCornersSB(
        gray, size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if ok:
        return corners.reshape(-1, 2).astype(np.float32)

    ok, corners = cv2.findChessboardCorners(
        gray, size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float32)


def view_features(corners: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Summarise one detection so that views can be compared for diversity.

    [centre x, centre y, apparent size, aspect], all roughly in [0, 1]. Two views with
    similar features show the board in a similar place, at a similar distance and tilt.
    """
    assert corners.ndim == 2 and corners.shape[1] == 2, corners.shape
    w, h = image_size
    span_x = corners[:, 0].ptp()
    span_y = corners[:, 1].ptp()
    return np.array(
        [
            corners[:, 0].mean() / w,
            corners[:, 1].mean() / h,
            np.sqrt(span_x * span_y) / np.sqrt(w * h),
            span_y / max(span_x, 1e-6),
        ]
    )


def select_spread(features: np.ndarray, n: int) -> list[int]:
    """Pick n views that are as different from each other as possible.

    Farthest-point sampling: start from the view closest to the mean, then repeatedly add
    whichever remaining view is farthest from everything already chosen. Deterministic.

    features (M, D); returns n indices (or all M of them if M <= n).
    """
    assert features.ndim == 2, features.shape
    m = len(features)
    if m <= n:
        return list(range(m))
    chosen = [int(np.argmin(np.linalg.norm(features - features.mean(0), axis=1)))]
    dist = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(features - features[nxt], axis=1))
    return sorted(chosen)


def calibrate(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> Calibration:
    """Run cv2.calibrateCamera and compute the per-view reprojection RMS."""
    assert len(object_points) == len(image_points) and len(object_points) >= 3
    obj = [p.reshape(-1, 1, 3).astype(np.float32) for p in object_points]
    img = [p.reshape(-1, 1, 2).astype(np.float32) for p in image_points]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None)

    per_view = np.empty(len(obj))
    for i, (o, p, rvec, tvec) in enumerate(zip(obj, img, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        per_view[i] = float(np.sqrt(np.mean(np.sum((proj - p) ** 2, axis=2))))

    return Calibration(
        K=np.asarray(K, float),
        dist=np.asarray(dist, float).reshape(-1),
        rms=float(rms),
        per_view_rms=per_view,
        image_size=(int(image_size[0]), int(image_size[1])),
    )
