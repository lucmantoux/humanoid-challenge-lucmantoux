"""Geometry tests for the table frame T (EXPLAINER Module 2)."""

import cv2
import numpy as np

from palm_prior.perception.aruco import (
    board_pose,
    build_board,
    intersect_plane_z,
    make_detector,
    project,
    ray,
    static_pose,
)
from palm_prior.utils import load_config

CFG = load_config()

# A plausible 1080p phone camera looking down at the table from 80 cm.
K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
R_CT = cv2.Rodrigues(np.array([np.pi - 0.55, 0.0, 0.0]))[0] @ cv2.Rodrigues(np.array([0.0, 0.0, 0.1]))[0]
T_CT = np.array([-0.07, 0.10, 0.80])


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
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([0.02, -0.05, 0.78])
    X_T = rng.uniform(-0.3, 0.3, size=(50, 3))
    X_C = X_T @ R.T + t
    X_T_back = (X_C - t) @ R
    assert np.abs(X_T_back - X_T).max() < 1e-6


def test_ray_then_intersect_inverts_project():
    """project -> ray -> intersect_plane_z returns the original point to under 1 um."""
    rng = np.random.default_rng(1)
    for h in (0.0, 0.02, 0.09):
        X_T = np.concatenate([rng.uniform(0.0, 0.5, size=2), [h]])
        uv = project(X_T, R_CT, T_CT, K, DIST)
        back = intersect_plane_z(ray(uv[0], uv[1], K, DIST), R_CT, T_CT, h)
        assert np.abs(back - X_T).max() < 1e-6, (X_T, back)


def test_ray_then_intersect_inverts_project_with_distortion():
    """The same round trip survives a realistically distorted lens."""
    dist = np.array([-0.21, 0.07, 0.001, -0.0008, 0.0])
    rng = np.random.default_rng(2)
    X_T = np.stack(
        [np.concatenate([rng.uniform(0.0, 0.5, size=2), [0.02]]) for _ in range(20)]
    )
    for p in X_T:
        uv = project(p, R_CT, T_CT, K, dist)
        back = intersect_plane_z(ray(uv[0], uv[1], K, dist), R_CT, T_CT, 0.02)
        assert np.abs(back - p).max() < 1e-5, (p, back)


def test_intersect_plane_z_is_linear_in_height():
    """Looking straight down the optical axis, raising the plane only changes z."""
    uv = project(np.array([0.2, 0.15, 0.0]), R_CT, T_CT, K, DIST)
    low = intersect_plane_z(ray(uv[0], uv[1], K, DIST), R_CT, T_CT, 0.0)
    high = intersect_plane_z(ray(uv[0], uv[1], K, DIST), R_CT, T_CT, 0.05)
    assert np.isclose(high[2] - low[2], 0.05)
    # a slanted camera makes a raised point appear to slide: that is the parallax that
    # forces Module 4 to know the block height before it can place it on the table
    assert np.linalg.norm(high[:2] - low[:2]) > 0.01


def _render_board(image_size=(1920, 1080), px=1200, margin=80):
    """Render the printed sheet as the camera at (R_CT, T_CT) would see it.

    Returns a BGR image. The printed image is mapped into the camera by the homography
    K [r1 r2 t] A, where A takes printed pixels to table-frame metres on z_T = 0.
    """
    tb = build_board(CFG)
    grid = cv2.aruco.GridBoard(
        (int(CFG.aruco.grid[0]), int(CFG.aruco.grid[1])),
        float(CFG.aruco.marker_len),
        float(CFG.aruco.marker_gap),
        tb.dictionary,
    )
    sheet = grid.generateImage((px, px), marginSize=margin)
    metres_per_px = tb.size[0] / (px - 2 * margin)
    A = np.array(
        [
            [metres_per_px, 0.0, -margin * metres_per_px],
            [0.0, -metres_per_px, (px - margin) * metres_per_px],
            [0.0, 0.0, 1.0],
        ]
    )
    H = K @ np.column_stack([R_CT[:, 0], R_CT[:, 1], T_CT]) @ A
    warped = cv2.warpPerspective(
        sheet, H, image_size, flags=cv2.INTER_CUBIC, borderValue=255
    )
    return tb, cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def test_board_pose_recovers_a_synthetic_camera_pose():
    tb, image = _render_board()
    pose = board_pose(image, tb, make_detector(tb), K, DIST, int(CFG.aruco.min_markers))
    assert pose is not None, "the board was not detected in the synthetic render"
    R_hat, t_hat = pose
    # sideways the board is pinned to a tenth of a millimetre; range is the weak axis of
    # any PnP on a small planar target, so 5 mm in 800 mm (0.6%) is the honest limit
    assert np.abs(t_hat[:2] - T_CT[:2]).max() < 2e-4, (t_hat, T_CT)
    assert abs(t_hat[2] - T_CT[2]) < 5e-3, (t_hat, T_CT)
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R_hat.T @ R_CT)[0]))
    assert angle < 0.3, angle


def test_board_pose_returns_none_without_a_board():
    tb = build_board(CFG)
    blank = np.full((1080, 1920, 3), 255, np.uint8)
    assert board_pose(blank, tb, make_detector(tb), K, DIST) is None


def test_static_pose_rejects_a_single_bad_frame():
    """A median translation and a chordal rotation mean shrug off one outlier."""
    good = [(R_CT, T_CT) for _ in range(9)]
    bad = (cv2.Rodrigues(np.array([0.0, 0.0, 1.2]))[0] @ R_CT, T_CT + 0.3)
    R_hat, t_hat = static_pose(good + [bad])
    assert np.abs(t_hat - T_CT).max() < 1e-9
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R_hat.T @ R_CT)[0]))
    assert angle < 7.0, angle
