"""Training and the E2 evaluation (EXPLAINER Module 9).

The loss is the 5-step self-rollout of §5: MSE on the block displacement plus
0.5 · BCE on the attach label. Each member has its own bootstrap resample.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from palm_prior.wm.model import Ensemble, NormStats, rollout


class Windows:
    """Consecutive transitions that a rollout of `horizon` steps can start from."""

    def __init__(self, s, a, dp, att, starts: np.ndarray):
        self.s = torch.as_tensor(np.asarray(s, np.float32))
        self.a = torch.as_tensor(np.asarray(a, np.float32))
        self.dp = torch.as_tensor(np.asarray(dp, np.float32))
        self.att = torch.as_tensor(np.asarray(att, np.float32))
        self.starts = np.asarray(starts, np.int64)
        assert len(self.starts) > 0, "no rollout windows"

    def batch(self, idx: np.ndarray, horizon: int, device) -> tuple[torch.Tensor, ...]:
        """idx indexes self.starts. Returns s0 (B,8), actions (B,H,4), dp (B,H,3), att (B,H)."""
        origin = self.starts[idx]
        win = origin[:, None] + np.arange(horizon)[None, :]
        s0 = self.s[origin].to(device)
        actions = self.a[win].to(device)
        dp = self.dp[win].to(device)
        att = self.att[win].to(device)
        return s0, actions, dp, att


def windows_from_ids(s, a, dp, att, ids, horizon: int) -> Windows:
    """One window start per position that still has `horizon` steps inside its episode."""
    ids = np.asarray(ids)
    cuts = np.where(np.diff(ids))[0] + 1
    bounds = np.concatenate([[0], cuts, [len(ids)]])
    starts = []
    for b, e in zip(bounds[:-1], bounds[1:]):
        if e - b >= horizon:
            starts.extend(range(b, e - horizon + 1))
    return Windows(s, a, dp, att, np.array(starts, np.int64))


def episode_starts(ids, horizon: int, dp=None) -> np.ndarray:
    """One rollout start per episode that is long enough.

    With `dp` (N, 3), the start is the window in which the block travels farthest.
    Robot episodes open with an approach that leaves the block still, so a rollout
    from the first step is the same number for every model.
    """
    ids = np.asarray(ids)
    cuts = np.where(np.diff(ids))[0] + 1
    bounds = np.concatenate([[0], cuts, [len(ids)]])
    starts = []
    for b, e in zip(bounds[:-1], bounds[1:]):
        if e - b < horizon:
            continue
        if dp is None:
            starts.append(b)
            continue
        best_i, best = b, -1.0
        block = np.asarray(dp[b:e], float)
        for i in range(0, e - b - horizon + 1):
            travel = float(np.linalg.norm(block[i : i + horizon].sum(axis=0)))
            if travel > best:
                best, best_i = travel, b + i
        starts.append(best_i)
    return np.array(starts, np.int64)


def _loss(model, s0, actions, dp_true, att_true, embodiment, stats, w_a) -> torch.Tensor:
    """s0 (M, B, 8), actions (M, B, H, 4), dp_true (M, B, H, 3), att_true (M, B, H)."""
    dp_hat, logits = rollout(model, s0, actions, embodiment, stats)
    mse = ((dp_hat - dp_true) ** 2).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, att_true)
    return mse + float(w_a) * bce


def train_ensemble(
    model: Ensemble,
    pools: list[tuple[Windows, float, float]],
    steps: int,
    lr: float,
    batch_size: int,
    horizon: int,
    w_a: float,
    stats: NormStats,
    seed: int,
) -> None:
    """All members train in one forward. Each has its own bootstrap RNG.

    pools is a list of (windows, embodiment, fraction of the batch). Fractions
    should sum to 1. A single pool is the whole batch.
    """
    if int(steps) <= 0:
        return
    rngs = [np.random.default_rng(seed + 1000 * m) for m in range(model.n_members)]
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    device = next(model.parameters()).device
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = 0.0
        for windows, embodiment, frac in pools:
            n = max(1, int(round(batch_size * frac)))
            parts = [
                windows.batch(rng.integers(0, len(windows.starts), size=n), horizon, device)
                for rng in rngs
            ]
            s0 = torch.stack([p[0] for p in parts])
            actions = torch.stack([p[1] for p in parts])
            dp = torch.stack([p[2] for p in parts])
            att = torch.stack([p[3] for p in parts])
            loss = loss + float(frac) * _loss(model, s0, actions, dp, att, embodiment, stats, w_a)
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(
    model: Ensemble,
    s, a, dp, att, ids,
    horizon: int,
    stats: NormStats,
    embodiment: float,
) -> tuple[float, float, float]:
    """10-step block-position error in cm, attach AUROC, and mean ensemble std of Δz.

    One rollout per episode, from its first step. The position error is the error
    in the summed block displacement, which is the error in where the block ends up.
    """
    model.eval()
    device = next(model.parameters()).device
    starts = episode_starts(ids, horizon, dp)
    if len(starts) == 0:
        raise RuntimeError(f"no episode is {horizon} steps long")
    # Evaluate in chunks so a long held-out set stays small.
    err = []
    scores = []
    labels = []
    spreads = []
    for chunk in np.array_split(starts, max(1, len(starts) // 64)):
        if len(chunk) == 0:
            continue
        s0 = torch.as_tensor(s[chunk], dtype=torch.float32, device=device)
        offs = np.arange(horizon)
        win = chunk[:, None] + offs[None, :]
        actions = torch.as_tensor(a[win], dtype=torch.float32, device=device)
        dp_true = np.asarray(dp[win], np.float32)
        att_true = np.asarray(att[win], np.float32)
        s0m = s0.unsqueeze(0).expand(model.n_members, -1, -1)
        actm = actions.unsqueeze(0).expand(model.n_members, -1, -1, -1)
        stacked, logits = rollout(model, s0m, actm, embodiment, stats)
        mean_dp = stacked.mean(dim=0).cpu().numpy()
        spreads.append(stacked[..., 2].std(dim=0).mean().item())
        err.append(np.linalg.norm(mean_dp.sum(axis=1) - dp_true.sum(axis=1), axis=1))
        prob = torch.sigmoid(logits).mean(dim=0).cpu().numpy()
        scores.append(prob.reshape(-1))
        labels.append(att_true.reshape(-1))
    pos_cm = float(np.concatenate(err).mean() * 100.0)
    return pos_cm, _auroc(np.concatenate(scores), np.concatenate(labels)), float(np.mean(spreads))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC. Constant scores return 0.5."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    # Average rank for ties.
    sorted_s = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def probe_lift(model: Ensemble, stats: NormStats, embodiment: float, lateral_m: float) -> tuple[float, float]:
    """Predicted Δz and the member spread for a closed gripper lifting 2 cm.

    lateral_m is the horizontal miss between the gripper and the block. 0 is a
    centred grasp; a few centimetres is the F1 situation the failure clips exist for.
    """
    model.eval()
    device = next(model.parameters()).device
    s = torch.tensor([[lateral_m, 0.0, 0.01, 0.15, 0.0, 0.02, 0.015, 1.0]], device=device)
    a = torch.tensor([[[0.0, 0.0, 0.02, 1.0]]], device=device)
    s = s.unsqueeze(0).expand(model.n_members, -1, -1)
    a = a.unsqueeze(0).expand(model.n_members, -1, -1, -1)
    with torch.no_grad():
        dp, _ = rollout(model, s, a, embodiment, stats)
    dz = dp[:, 0, 0, 2].cpu().numpy()
    return float(dz.mean()), float(dz.std())
