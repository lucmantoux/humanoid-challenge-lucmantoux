"""CEM on a quadratic whose minimum is known (EXPLAINER §10)."""

import numpy as np

from palm_prior.plan.cem import cem

# Minimum of (x - 1)^2 + (y + 2)^2 is at (1, -2).
OPT = np.array([1.0, -2.0])


def _quadratic(theta: np.ndarray) -> np.ndarray:
    return ((theta - OPT) ** 2).sum(axis=1)


def test_quadratic_mean_reaches_the_optimum():
    rng = np.random.default_rng(0)
    mu, sigma = cem(
        np.zeros(2), np.ones(2), _quadratic,
        n_samples=256, n_elites=25, n_iters=6, sigma_min=1e-3, rng=rng,
    )
    np.testing.assert_allclose(mu, OPT, atol=0.05)
    assert np.all(sigma >= 1e-3 - 1e-12)


def test_integer_dimension_is_rounded():
    """The second coordinate is integer; the minimum of (x-0.4)^2 + (k-3)^2 is k=3."""
    rng = np.random.default_rng(1)

    def score(theta: np.ndarray) -> np.ndarray:
        return (theta[:, 0] - 0.4) ** 2 + (theta[:, 1] - 3.0) ** 2

    mu, _sigma = cem(
        np.zeros(2), np.array([1.0, 2.0]), score,
        n_samples=200, n_elites=20, n_iters=5, sigma_min=0.05, rng=rng,
        integer_dims=(1,),
    )
    assert abs(mu[1] - 3.0) < 0.5
    assert abs(mu[0] - 0.4) < 0.1


def test_shifting_knots_pins_the_origin_and_moves_the_bump():
    from palm_prior.plan.mpc import shift_knots

    knots = np.zeros((8, 3))
    knots[4] = [0.0, 0.0, 0.1]
    shifted = shift_knots(knots, 5, n=80)
    assert shifted.shape == (8, 3)
    np.testing.assert_allclose(shifted[0], 0.0, atol=1e-9)
    # The bump was halfway along the path; five steps later its mass is earlier.
    index = np.arange(8)
    before = np.dot(index, knots[:, 2]) / knots[:, 2].sum()
    after = np.dot(index, shifted[:, 2]) / shifted[:, 2].sum()
    assert after < before
