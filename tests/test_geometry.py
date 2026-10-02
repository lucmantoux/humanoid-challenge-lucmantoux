"""Geometry tests for the table frame T (EXPLAINER Module 2)."""

import cv2
import numpy as np

from palm_prior.perception.aruco import build_board
from palm_prior.utils import load_config

CFG = load_config()


def _detect_printed_board(px: int = 900, margin: int = 60):
    """Render the board exactly as it is printed and detect its markers.

    GridBoard.generateImage puts the board's top edge (max y in T) at image row 0,
    which is what the printed sheet looks like seen from above.
    """
    tb = build_board(CFG)
    grid = cv2.aruco.GridBoard(
        (int(CFG.aruco.grid[0]), int(CFG.aruco.grid[1])),
        float(CFG.aruco.marker_len),
        float(CFG.aruco.marker_gap),
        tb.dictionary,
    )
    img = grid.generateImage((px, px), marginSize=margin)
    detector = cv2.aruco.ArucoDetector(tb.dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    assert ids is not None and len(ids) == int(CFG.aruco.grid[0]) * int(CFG.aruco.grid[1])
    return tb, img, corners, ids


def test_board_object_points_span_the_printed_sheet():
    tb = build_board(CFG)
    obj = np.array(tb.board.getObjPoints())
    atol = 1e-6  # the object points are float32, so compare at the micrometre level
    assert np.allclose(obj[..., 2], 0.0, atol=atol)
    assert np.isclose(obj[..., 0].min(), 0.0, atol=atol)
    assert np.isclose(obj[..., 0].max(), tb.size[0], atol=atol)
    assert np.isclose(obj[..., 1].min(), 0.0, atol=atol)
    assert np.isclose(obj[..., 1].max(), tb.size[1], atol=atol)
    assert np.isclose(tb.size[0], 0.135) and np.isclose(tb.size[1], 0.135)


def test_table_frame_z_points_up_out_of_the_table():
    """A camera above the sheet must see the T z axis pointing back at it."""
    tb, img, corners, ids = _detect_printed_board()
    obj_pts, img_pts = tb.board.matchImagePoints(corners, ids)
    c = img.shape[0] / 2.0
    K = np.array([[4000.0, 0, c], [0, 4000.0, c], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5))
    assert ok
    R_CT, _ = cv2.Rodrigues(rvec)

    # camera looks along +z_C, so the board z axis must have a negative z component
    assert R_CT[2, 2] < -0.99, R_CT
    # x_T stays to the right, y_T is up in the image (negative image y)
    np.testing.assert_allclose(R_CT, np.diag([1.0, -1.0, -1.0]), atol=1e-3)
    assert tvec[2] > 0


def test_origin_corner_is_bottom_left_of_the_printed_sheet():
    tb, img, corners, ids = _detect_printed_board()
    obj_pts, img_pts = tb.board.matchImagePoints(corners, ids)
    obj_pts = obj_pts.reshape(-1, 3)
    img_pts = img_pts.reshape(-1, 2)
    origin = np.argmin(np.linalg.norm(obj_pts[:, :2], axis=1))
    np.testing.assert_allclose(obj_pts[origin, :2], [0.0, 0.0], atol=1e-6)
    # that corner is the left-most and the lowest point in the image
    assert img_pts[origin, 0] == img_pts[:, 0].min()
    assert img_pts[origin, 1] == img_pts[:, 1].max()


def test_round_trip_point_through_a_synthetic_pose():
    """X_C = R_CT X_T + t_CT inverts to better than 1e-6 m."""
    rng = np.random.default_rng(0)
    rvec = rng.normal(scale=0.4, size=3)
    R_CT, _ = cv2.Rodrigues(rvec)
    t_CT = np.array([0.02, -0.05, 0.78])
    X_T = rng.uniform(-0.3, 0.3, size=(50, 3))
    X_C = X_T @ R_CT.T + t_CT
    X_T_back = (X_C - t_CT) @ R_CT
    assert np.abs(X_T_back - X_T).max() < 1e-6
