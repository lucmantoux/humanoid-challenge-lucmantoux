"""Ensemble dynamics model (EXPLAINER Module 9).

Input is the standardised state and action plus the embodiment flag, 13 numbers.
Output is a standardised block displacement and one attach logit. The displacement
is turned back into metres before it is returned.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from palm_prior.state import ACTION_DIM, STATE_DIM, next_state


class NormStats(NamedTuple):
    """Per-coordinate mean and std, fitted on the data a model trains on."""

    s_mean: np.ndarray   # (8,)
    s_std: np.ndarray
    a_mean: np.ndarray   # (4,)
    a_std: np.ndarray
    dp_mean: np.ndarray  # (3,)
    dp_std: np.ndarray

    def save(self) -> dict[str, np.ndarray]:
        return self._asdict()

    @classmethod
    def load(cls, blob: dict) -> "NormStats":
        return cls(**{k: np.asarray(blob[k], float) for k in cls._fields})


def fit_stats(s: np.ndarray, a: np.ndarray, dp: np.ndarray) -> NormStats:
    """Column-wise mean and std. A constant column gets std 1 so division is safe."""
    def _ms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    return NormStats(*_ms(s), *_ms(a), *_ms(dp))


class Ensemble(nn.Module):
    """M independent MLPs, 13 → 128 → 128 → 128 → 4, SiLU.

    The M networks are stored as batched weights and run in one forward pass.
    A leading axis of size M on the input selects them.
    """

    def __init__(self, n_members: int = 5, hidden: tuple[int, ...] = (128, 128, 128)):
        super().__init__()
        self.n_members = int(n_members)
        dims = [STATE_DIM + ACTION_DIM + 1, *[int(h) for h in hidden], 4]
        weights = []
        biases = []
        for din, dout in zip(dims[:-1], dims[1:]):
            w = torch.empty(self.n_members, din, dout)
            b = torch.empty(self.n_members, dout)
            for m in range(self.n_members):
                layer = nn.Linear(din, dout)
                w[m] = layer.weight.T.detach()
                b[m] = layer.bias.detach()
            weights.append(nn.Parameter(w))
            biases.append(nn.Parameter(b))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.n_layers = len(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (M, B, 13) → (M, B, 4)."""
        assert x.shape[0] == self.n_members and x.shape[-1] == STATE_DIM + ACTION_DIM + 1, x.shape
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("mbi,mio->mbo", h, w) + b[:, None, :]
            if i < self.n_layers - 1:
                h = torch.nn.functional.silu(h)
        return h


def pack(s: torch.Tensor, a: torch.Tensor, embodiment: float, stats: NormStats, device) -> torch.Tensor:
    """Standardised [s, a, e]. s (..., 8), a (..., 4) → (..., 13)."""
    sm = torch.as_tensor(stats.s_mean, dtype=torch.float32, device=device)
    ss = torch.as_tensor(stats.s_std, dtype=torch.float32, device=device)
    am = torch.as_tensor(stats.a_mean, dtype=torch.float32, device=device)
    as_ = torch.as_tensor(stats.a_std, dtype=torch.float32, device=device)
    e = torch.full(s.shape[:-1] + (1,), float(embodiment), device=device, dtype=s.dtype)
    return torch.cat([(s - sm) / ss, (a - am) / as_, e], dim=-1)


def denorm_dp(raw: torch.Tensor, stats: NormStats, device) -> torch.Tensor:
    """The first 3 outputs, back in metres. raw (..., 4) → (..., 3)."""
    mean = torch.as_tensor(stats.dp_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(stats.dp_std, dtype=torch.float32, device=device)
    return raw[..., :3] * std + mean


def rollout(
    model: Ensemble,
    s0: torch.Tensor,
    actions: torch.Tensor,
    embodiment: float,
    stats: NormStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Open-loop rollout of every member.

    s0 (M, B, 8), actions (M, B, H, 4).
    returns dp (M, B, H, 3) in metres and attach logits (M, B, H).
    Step k is fed the state built from that member's own earlier predictions.
    """
    assert s0.shape[0] == model.n_members and actions.shape[0] == model.n_members, (s0.shape, actions.shape)
    device = s0.device
    state = s0
    dps = []
    logits = []
    for k in range(actions.shape[2]):
        out = model(pack(state, actions[:, :, k], embodiment, stats, device))
        dp = denorm_dp(out, stats, device)
        dps.append(dp)
        logits.append(out[..., 3])
        state = next_state(state, actions[:, :, k], dp)
    return torch.stack(dps, dim=2), torch.stack(logits, dim=2)
