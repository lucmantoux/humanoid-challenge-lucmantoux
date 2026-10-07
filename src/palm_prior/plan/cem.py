"""Cross-entropy method (EXPLAINER §10).

A diagonal Gaussian is sampled, the best candidates are kept, and the Gaussian is
refit to them. Integer coordinates are rounded before they are scored. Lower score
is better.
"""

from __future__ import annotations

import numpy as np


def cem(
    mu: np.ndarray,
    sigma: np.ndarray,
    score,
    *,
    n_samples: int,
    n_elites: int,
    n_iters: int,
    sigma_min: np.ndarray | float,
    rng: np.random.Generator,
    integer_dims: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Refit (mu, sigma) for n_iters. score(theta) maps (N, D) to (N,) costs.

    returns the final mean and per-coordinate std, both (D,).
    """
    mu = np.asarray(mu, float).copy()
    sigma = np.asarray(sigma, float).copy()
    floor = np.broadcast_to(np.asarray(sigma_min, float), mu.shape).copy()
    assert mu.shape == sigma.shape, (mu.shape, sigma.shape)
    assert n_elites < n_samples, (n_elites, n_samples)
    dim = mu.shape[0]
    for _ in range(int(n_iters)):
        theta = rng.normal(mu, sigma, size=(int(n_samples), dim))
        if integer_dims:
            theta[:, list(integer_dims)] = np.rint(theta[:, list(integer_dims)])
        costs = np.asarray(score(theta), float)
        assert costs.shape == (int(n_samples),), costs.shape
        elite = theta[np.argpartition(costs, int(n_elites) - 1)[: int(n_elites)]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), floor)
    return mu, sigma
