"""Calibration tests on synthetic views (EXPLAINER Module 1).

Projecting a known board with a known K and reading K back is the only way to check the
calibration path before any real footage exists.
"""

import cv2
import numpy as np

from palm_prior.perception.calibration import (
    board_points,
    calibrate,
    detect_corners,
    select_spread,
    view_features,
)
from palm_prior.utils import load_config

CFG = load_config()
INNER = (int(CFG.calib.checker_inner[0]), int(CFG.calib.checker_inner[1]))
SQUARE = float(CFG.calib.checker_square)
IMAGE_SIZE = (1920, 1080)
K_TRUE = np.array([[1500.0, 0.0, 950.0], [0.0, 1500.0, 545.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.18, 0.05, 0.0008, -0.0006, 0.0])


def _views(n: int, dist: np.ndarray, seed: int = 0):
    """n synthetic detections of the board, varied in pose so the problem is well posed."""
    rng = np.random.default_rng(seed)
    grid = board_points(INNER, SQUARE)
    centred = grid - grid.mean(axis=0)
    views = []
    while len(views) < n:
        rvec = rng.uniform(-0.45, 0.45, size=3)
        tvec = np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), rng.uniform(0.35, 0.75)]
        )
        uv, _ = cv2.projectPoints(centred, rvec, tvec, K_TRUE, dist)
        uv = uv.reshape(-1, 2)
        inside = (
            (uv[:, 0] > 20)
            & (uv[:, 0] < IMAGE_SIZE[0] - 20)
            & (uv[:, 1] > 20)
            & (uv[:, 1] < IMAGE_SIZE[1] - 20)
        )
        if inside.all():
            views.append(uv.astype(np.float32))
    return grid, views


def test_board_points_match_the_printed_checkerboard():
    grid = board_points(INNER, SQUARE)
    assert grid.shape == (INNER[0] * INNER[1], 3)
    assert np.allclose(grid[:, 2], 0.0)
    assert np.isclose(grid[:, 0].max(), (INNER[0] - 1) * SQUARE)
    assert np.isclose(grid[:, 1].max(), (INNER[1] - 1) * SQUARE)
    # consecutive points walk along x first, which is the order OpenCV detects them in
    assert np.isclose(np.linalg.norm(grid[1] - grid[0]), SQUARE)


def test_calibrate_recovers_a_known_pinhole():
    grid, views = _views(15, np.zeros(5))
    result = calibrate([grid] * len(views), views, IMAGE_SIZE)
    assert result.rms < 0.01, result.rms
    np.testing.assert_allclose(result.K[:2, :2], K_TRUE[:2, :2], rtol=2e-3)
    np.testing.assert_allclose(result.K[:2, 2], K_TRUE[:2, 2], atol=2.0)
    assert np.abs(result.dist).max() < 0.02
    assert result.per_view_rms.shape == (len(views),)


def test_calibrate_recovers_a_known_distortion():
    grid, views = _views(20, DIST_TRUE, seed=3)
    result = calibrate([grid] * len(views), views, IMAGE_SIZE)
    assert result.rms < 0.02, result.rms
    np.testing.assert_allclose(result.dist[:2], DIST_TRUE[:2], atol=0.02)
    np.testing.assert_allclose(result.K[:2, :2], K_TRUE[:2, :2], rtol=5e-3)


def test_detect_corners_finds_a_rendered_checkerboard():
    """A clean, fronto-parallel render must be found, and found accurately."""
    cols, rows = INNER
    cell = 90
    board = np.zeros(((rows + 1) * cell, (cols + 1) * cell), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = 255
    image = np.full((board.shape[0] + 200, board.shape[1] + 200), 128, np.uint8)
    image[100 : 100 + board.shape[0], 100 : 100 + board.shape[1]] = board

    corners = detect_corners(image, INNER)
    assert corners is not None
    assert corners.shape == (cols * rows, 2)
    expected = np.mgrid[1 : cols + 1, 1 : rows + 1].T.reshape(-1, 2) * cell + 100
    error = np.linalg.norm(np.sort(corners, axis=0) - np.sort(expected, axis=0), axis=1)
    assert error.max() < 1.0, error.max()


def test_detect_corners_returns_none_on_a_blank_frame():
    assert detect_corners(np.full((480, 640), 200, np.uint8), INNER) is None


def test_select_spread_keeps_everything_when_there_is_little():
    features = np.random.default_rng(0).normal(size=(5, 4))
    assert select_spread(features, 40) == [0, 1, 2, 3, 4]


def test_select_spread_prefers_the_corners_of_feature_space():
    """A tight cluster plus four outliers: picking 4 must spend 3 of them on outliers."""
    outliers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cluster = 0.5 + 0.01 * np.random.default_rng(0).normal(size=(30, 2))
    chosen = select_spread(np.vstack([cluster, outliers]), 4)
    assert chosen == sorted(chosen)
    # the seed is the view nearest the mean, i.e. one of the cluster; the rest are corners
    assert len(set(chosen) & {30, 31, 32, 33}) == 3, chosen


def test_select_spread_is_deterministic():
    features = np.random.default_rng(7).normal(size=(60, 4))
    assert select_spread(features, 12) == select_spread(features, 12)


def test_view_features_separate_near_from_far():
    """A board filling the frame must score a bigger apparent size than a distant one."""
    near = np.array([[100.0, 100.0], [1800.0, 980.0]])
    far = np.array([[900.0, 500.0], [1000.0, 580.0]])
    assert view_features(near, IMAGE_SIZE)[2] > view_features(far, IMAGE_SIZE)[2]
