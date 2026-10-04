"""The ArUco table board and the table frame T (EXPLAINER Module 2)."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class TableBoard(NamedTuple):
    """An ArUco board whose object points are expressed in the table frame T.

    board      cv2.aruco.Board, object points (n_markers, 4, 3) in T, metres
    dictionary cv2.aruco.Dictionary used to detect it
    size       (width_x, height_y) of the printed board in metres
    """

    board: cv2.aruco.Board
    dictionary: cv2.aruco.Dictionary
    size: tuple[float, float]


def build_board(cfg) -> TableBoard:
    """Build the printed board with object points in the table frame T.

    T has its origin at the ORIGIN corner of the printed sheet (bottom-left as printed),
    x to the right along the bottom edge, y up the left edge, z out of the page.

    cv2.aruco.GridBoard numbers its markers from y = 0 at the *top* of the generated
    image, which makes its own z axis point into the page. Mirroring y about the board
    height is a 180 deg rotation about x (det = +1), so it turns that frame into T
    without touching the printed picture.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco.dict))
    n_x, n_y = int(cfg.aruco.grid[0]), int(cfg.aruco.grid[1])
    side = float(cfg.aruco.marker_len)
    gap = float(cfg.aruco.marker_gap)

    grid = cv2.aruco.GridBoard((n_x, n_y), side, gap, dictionary)
    width = n_x * side + (n_x - 1) * gap
    height = n_y * side + (n_y - 1) * gap

    obj_T = np.array(grid.getObjPoints(), dtype=np.float32)
    assert obj_T.shape == (n_x * n_y, 4, 3), obj_T.shape
    obj_T[..., 1] = height - obj_T[..., 1]

    # corner 0 of every marker is its top-left as printed, i.e. (min x, max y) in T
    for marker in obj_T:
        assert np.allclose(marker[0], [marker[:, 0].min(), marker[:, 1].max(), 0.0], atol=1e-6)

    board = cv2.aruco.Board(obj_T, dictionary, grid.getIds())
    return TableBoard(board=board, dictionary=dictionary, size=(width, height))


def make_detector(tb: TableBoard) -> cv2.aruco.ArucoDetector:
    """One detector reused across a whole video."""
    return cv2.aruco.ArucoDetector(tb.dictionary, cv2.aruco.DetectorParameters())


def board_pose(
    image: np.ndarray,
    tb: TableBoard,
    detector: cv2.aruco.ArucoDetector,
    K: np.ndarray,
    dist: np.ndarray,
    min_markers: int = 3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose of the table frame T in the camera frame C for one frame.

    image  (H, W, 3) BGR or (H, W) grayscale
    K      (3, 3) camera matrix
    dist   (5,) or (n,) distortion coefficients

    returns (R_CT (3,3), t_CT (3,)) with X_C = R_CT X_T + t_CT, or None if too few
    markers were found.
    """
    assert K.shape == (3, 3), K.shape
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < min_markers:
        return None
    obj_pts, img_pts = tb.board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R_CT, _ = cv2.Rodrigues(rvec)
    return R_CT, tvec.reshape(3)


def static_pose(poses: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-frame poses of a tripod-mounted camera into one.

    Translations are combined with a median (robust to the odd bad frame); rotations with
    the chordal mean of scipy's Rotation, which is the right average on SO(3).

    returns (R_CT (3,3), t_CT (3,))
    """
    assert len(poses) > 0, "no poses to average"
    rots = Rotation.from_matrix(np.stack([R for R, _ in poses]))
    t_CT = np.median(np.stack([t for _, t in poses]), axis=0)
    return rots.mean().as_matrix(), t_CT


def ray(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Direction in C of the ray through pixel (u, v), with lens distortion removed.

    Returns [x, y, 1] in normalised camera coordinates — not unit length, because every
    use scales it anyway. For dist = 0 this is exactly K^-1 [u, v, 1] of EXPLAINER §7.
    """
    assert K.shape == (3, 3), K.shape
    undistorted = cv2.undistortPoints(np.array([[[float(u), float(v)]]]), K, dist)
    x, y = undistorted[0, 0]
    return np.array([x, y, 1.0])


def intersect_plane_z(
    ray_C: np.ndarray, R_CT: np.ndarray, t_CT: np.ndarray, h: float
) -> np.ndarray:
    """Where a camera ray meets the horizontal table-frame plane z_T = h.

    ray_C (3,) direction in C, R_CT (3,3), t_CT (3,), h metres above the table.
    returns X_T (3,) with X_T[2] == h.

    With n = R_CT e_z and c = n . t_CT, the scale along the ray is (c + h) / (n . ray),
    which reduces to the Z_plane of EXPLAINER §7 when h = 0.
    """
    assert ray_C.shape == (3,), ray_C.shape
    assert R_CT.shape == (3, 3) and t_CT.shape == (3,), (R_CT.shape, t_CT.shape)
    n = R_CT[:, 2]
    denom = float(n @ ray_C)
    if abs(denom) < 1e-12:
        raise ValueError("ray is parallel to the table plane")
    s = (float(n @ t_CT) + h) / denom
    X_C = s * ray_C
    X_T = R_CT.T @ (X_C - t_CT)
    assert abs(X_T[2] - h) < 1e-9, (X_T[2], h)
    return X_T


def project(X_T: np.ndarray, R_CT: np.ndarray, t_CT: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Project a table-frame point to pixels. X_T (3,) -> (2,). Inverse of ray + intersect."""
    assert X_T.shape == (3,), X_T.shape
    rvec, _ = cv2.Rodrigues(R_CT)
    uv, _ = cv2.projectPoints(X_T.reshape(1, 1, 3), rvec, t_CT.reshape(3, 1), K, dist)
    return uv.reshape(2)
