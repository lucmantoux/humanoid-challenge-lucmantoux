"""Two-anchor retargeting (EXPLAINER §8).

A 2-D similarity has four degrees of freedom and two point correspondences pin it down
exactly. Writing the plane as complex numbers turns that into one division.
"""

from __future__ import annotations

import numpy as np


def _complex(xy: np.ndarray) -> np.ndarray:
    """(2,) or (..., 2) real xy, in metres, as complex numbers."""
    xy = np.asarray(xy, float)
    assert xy.shape[-1] == 2, xy.shape
    return xy[..., 0] + 1j * xy[..., 1]


def _xy(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    return np.stack([np.real(z), np.imag(z)], axis=-1)


def similarity(
    x_g: np.ndarray,
    x_r: np.ndarray,
    block_xy: np.ndarray,
    target_xy: np.ndarray,
    degenerate_dist: float,
) -> tuple[complex, complex]:
    """The unique similarity taking the grasp to the block and the release to the target.

    x_g, x_r       (2,) grasp and release xy in the source frame, metres
    block_xy       (2,) where the grasp must land (G in EXPLAINER §8)
    target_xy      (2,) where the release must land (P)
    degenerate_dist  metres; below this the two anchors are one point and alpha = 1

    returns alpha, beta with T(x) = alpha x + beta.
    """
    x_g_c = complex(_complex(x_g))
    x_r_c = complex(_complex(x_r))
    grasp_anchor = complex(_complex(block_xy))
    release_anchor = complex(_complex(target_xy))
    separation = abs(x_r_c - x_g_c)
    if separation < float(degenerate_dist):
        alpha = 1 + 0j
    else:
        alpha = (release_anchor - grasp_anchor) / (x_r_c - x_g_c)
    beta = grasp_anchor - alpha * x_g_c
    return alpha, beta


def apply_xy(xy: np.ndarray, alpha: complex, beta: complex) -> np.ndarray:
    """Apply T(x) = alpha x + beta. xy (2,) or (..., 2) -> same shape."""
    return _xy(alpha * _complex(xy) + beta)


def kappa(alpha: complex, kappa_min: float, kappa_max: float) -> float:
    """Vertical scale: clip(|alpha|, kappa_min, kappa_max). A stretched plan also lifts more."""
    return float(np.clip(abs(alpha), float(kappa_min), float(kappa_max)))


def release_ramp(index: np.ndarray, k_close: int, k_open: int) -> np.ndarray:
    """s(t) in EXPLAINER §8: 0 at the grasp, 1 at and after the release, linear between.

    index is in control steps at 10 Hz. Returns float array, same shape as index.
    """
    index = np.asarray(index, float)
    if int(k_open) <= int(k_close):
        return (index >= int(k_close)).astype(float)
    span = float(k_open - k_close)
    return np.clip((index - float(k_close)) / span, 0.0, 1.0)


def vertical(
    z_h: np.ndarray, z_at_grasp: float, kappa_value: float, h_tgt: float, s: np.ndarray
) -> np.ndarray:
    """Retargeted hand height, metres.

    z*(t) = z_h(t_g) + kappa (z_h(t) - z_h(t_g)) + h_tgt s(t)

    s is 0 before the grasp and 1 after the release, so the full target height is added
    only once the hand has let go.
    """
    z_h = np.asarray(z_h, float)
    s = np.asarray(s, float)
    assert z_h.shape == s.shape, (z_h.shape, s.shape)
    return float(z_at_grasp) + float(kappa_value) * (z_h - float(z_at_grasp)) + float(h_tgt) * s
