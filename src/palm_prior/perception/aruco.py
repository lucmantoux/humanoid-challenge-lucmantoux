"""The ArUco table board and the table frame T (EXPLAINER Module 2)."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


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
