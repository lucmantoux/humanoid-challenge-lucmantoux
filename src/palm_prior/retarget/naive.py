"""The baseline retarget (EXPLAINER §8, last paragraph).

One fixed similarity, chosen once and applied to every clip. Where the block and the
target actually are is not an input: that is the whole point of the baseline, and the
thing the two-anchor transform is there to beat.
"""

from __future__ import annotations

import numpy as np

from palm_prior.retarget.two_anchor import apply_xy


def naive_global(xy: np.ndarray, alpha: complex, beta: complex) -> np.ndarray:
    """xy (2,) or (..., 2) through a fixed T(x) = alpha x + beta. Same shape out."""
    assert isinstance(alpha, complex) and isinstance(beta, complex)
    return apply_xy(np.asarray(xy, float), alpha, beta)
