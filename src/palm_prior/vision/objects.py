"""Lid and plate positions by intersecting a pixel ray with a height plane (FIX.md §4)."""

from __future__ import annotations

import numpy as np

from palm_prior.perception.aruco import intersect_plane_z, ray


def ray_plane(u: float, v: float, h: float, k: np.ndarray, dist: np.ndarray, r_ct: np.ndarray, t_ct: np.ndarray) -> np.ndarray:
    """Pixel (u, v) on the plane z = h, in the table frame. Returns (3,)."""
    direction = ray(float(u), float(v), k, dist)
    return intersect_plane_z(direction, r_ct, t_ct, float(h))


def median_point(points: np.ndarray) -> np.ndarray:
    """Median of (n, 3), ignoring non-finite rows. Returns (3,)."""
    points = np.asarray(points, float)
    good = np.isfinite(points).all(axis=1)
    if not good.any():
        return np.full(3, np.nan)
    return np.median(points[good], axis=0)
