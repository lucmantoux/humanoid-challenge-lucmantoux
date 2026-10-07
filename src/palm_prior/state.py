"""The shared 8-D state and 4-D action of EXPLAINER §2.

The same state is used for my hand (embodiment 0) and the Panda (embodiment 1):

    index 0:3  d_eo   = p_ee - p_obj                  (3,) hand/gripper relative to the block
    index 3:5  d_to   = p_tgt[:2] - p_obj[:2]         (2,) target relative to the block, horizontal
    index 5    z_obj  = block centre height above the surface
    index 6    h_tgt  = target top height
    index 7    g      = gripper closed (1) or open (0)

    action     a = [p_ee(t+1) - p_ee(t), g(t+1)]      (4,)

Positions are in the table frame T for human data and in the world frame W for sim data;
the state itself is frame-agnostic in xy because every horizontal quantity is a difference.

Every function here accepts numpy arrays or torch tensors and is batched: a leading
batch shape (...) is carried through unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

STATE_DIM = 8
STATE_DIM_V2 = 11          # v1 state plus target xyz, constant within an episode
ACTION_DIM = 4

I_D_EO = slice(0, 3)
I_D_TO = slice(3, 5)
I_Z_OBJ = 5
I_H_TGT = 6
I_G = 7

Array = np.ndarray | torch.Tensor


class StateParts(NamedTuple):
    """Views into a state array of shape (..., 8). Shapes as in EXPLAINER §2."""

    d_eo: Array  # (..., 3)
    d_to: Array  # (..., 2)
    z_obj: Array  # (...,)
    h_tgt: Array  # (...,)
    g: Array  # (...,)


def _cat(parts: list[Array]) -> Array:
    """Concatenate along the last axis with the backend of parts[0]."""
    if isinstance(parts[0], torch.Tensor):
        return torch.cat(parts, dim=-1)
    return np.concatenate(parts, axis=-1)


def _like(x, ref: Array) -> Array:
    """Return x as an array of ref's backend, dtype and device."""
    if isinstance(ref, torch.Tensor):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=ref.dtype, device=ref.device)
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return np.asarray(x, dtype=ref.dtype)


def _column(x, batch: tuple[int, ...], ref: Array) -> Array:
    """Broadcast a scalar or (...,) array to shape batch + (1,) in ref's backend."""
    x = _like(x, ref)
    if x.ndim == len(batch) + 1 and x.shape[-1] == 1:
        col = x
    else:
        col = x[..., None] if x.ndim > 0 else x.reshape(*([1] * (len(batch) + 1)))
    if isinstance(ref, torch.Tensor):
        return col.expand(*batch, 1)
    return np.broadcast_to(col, batch + (1,))


def build_state(p_ee: Array, p_obj: Array, p_tgt_xy: Array, h_tgt, g) -> Array:
    """Build s of shape (..., 8) from absolute positions in one frame (T or W).

    p_ee     (..., 3) end-effector / pinch point
    p_obj    (..., 3) block centre, z measured from the support surface
    p_tgt_xy (..., 2) target centre, horizontal only
    h_tgt    scalar or (...,) target top height
    g        scalar or (...,) gripper closed (1) / open (0)
    """
    p_ee = p_ee if isinstance(p_ee, torch.Tensor) else np.asarray(p_ee, dtype=float)
    p_obj = _like(p_obj, p_ee)
    p_tgt_xy = _like(p_tgt_xy, p_ee)
    assert p_ee.shape[-1] == 3, p_ee.shape
    assert p_obj.shape == p_ee.shape, (p_obj.shape, p_ee.shape)
    assert p_tgt_xy.shape[-1] == 2, p_tgt_xy.shape

    batch = p_ee.shape[:-1]
    d_eo = p_ee - p_obj
    d_to = p_tgt_xy - p_obj[..., :2]
    z_obj = p_obj[..., 2:3]
    s = _cat([d_eo, d_to, z_obj, _column(h_tgt, batch, p_ee), _column(g, batch, p_ee)])
    assert s.shape == batch + (STATE_DIM,), s.shape
    return s


def build_state_v2(p_ee: Array, p_obj: Array, p_tgt: Array, h_tgt, g) -> Array:
    """State of shape (..., 11): the 8-D state, then the target position (3,).

    The target is a condition. next_state copies it through and does not predict it.
    p_tgt (..., 3) is the target point the block should reach, in the same frame as p_ee.
    """
    p_tgt = p_tgt if isinstance(p_tgt, torch.Tensor) else np.asarray(p_tgt, dtype=float)
    base = build_state(p_ee, p_obj, p_tgt[..., :2], h_tgt, g)
    tgt = _like(p_tgt, base)
    assert tgt.shape[-1] == 3, tgt.shape
    s = _cat([base, tgt])
    assert s.shape[-1] == STATE_DIM_V2, s.shape
    return s


def build_action(dp_ee: Array, g_next) -> Array:
    """Build a of shape (..., 4) from the end-effector displacement and the next gripper command."""
    dp_ee = dp_ee if isinstance(dp_ee, torch.Tensor) else np.asarray(dp_ee, dtype=float)
    assert dp_ee.shape[-1] == 3, dp_ee.shape
    batch = dp_ee.shape[:-1]
    a = _cat([dp_ee, _column(g_next, batch, dp_ee)])
    assert a.shape == batch + (ACTION_DIM,), a.shape
    return a


def decode_state(s: Array) -> StateParts:
    """Split s (..., 8) into its named parts. Scalar parts lose the trailing axis."""
    assert s.shape[-1] == STATE_DIM, s.shape
    return StateParts(
        d_eo=s[..., I_D_EO],
        d_to=s[..., I_D_TO],
        z_obj=s[..., I_Z_OBJ],
        h_tgt=s[..., I_H_TGT],
        g=s[..., I_G],
    )


def next_state(s: Array, a: Array, dp_obj: Array) -> Array:
    """Apply the EXPLAINER §2 update equations.

    s      (..., 8) current state
    a      (..., 4) action, a[..., :3] moves the end-effector, a[..., 3] is the next gripper
    dp_obj (..., 3) block displacement over the step (measured, or predicted by the world model)

    returns (..., 8)
    """
    assert s.shape[-1] in (STATE_DIM, STATE_DIM_V2), s.shape
    assert a.shape[-1] == ACTION_DIM, a.shape
    assert dp_obj.shape[-1] == 3, dp_obj.shape
    assert s.shape[:-1] == a.shape[:-1] == dp_obj.shape[:-1], (s.shape, a.shape, dp_obj.shape)

    d_eo = s[..., I_D_EO] + a[..., 0:3] - dp_obj
    d_to = s[..., I_D_TO] - dp_obj[..., 0:2]
    z_obj = s[..., I_Z_OBJ : I_Z_OBJ + 1] + dp_obj[..., 2:3]
    h_tgt = s[..., I_H_TGT : I_H_TGT + 1]
    g = a[..., 3:4]
    parts = [d_eo, d_to, z_obj, h_tgt, g]
    if s.shape[-1] == STATE_DIM_V2:
        parts.append(s[..., STATE_DIM:])
    s_next = _cat(parts)
    assert s_next.shape == s.shape, (s_next.shape, s.shape)
    return s_next


def attached(g: Array, z_obj: Array, block_size: float, margin: float) -> Array:
    """The attach label of EXPLAINER Module 5, used identically for human and robot data.

    attached = (g == 1) and (z_obj > block_size / 2 + margin). Returns a float array of 0/1.
    """
    lifted = z_obj > block_size / 2.0 + margin
    closed = g > 0.5
    if isinstance(g, torch.Tensor):
        return (closed & lifted).to(dtype=g.dtype)
    return np.logical_and(closed, lifted).astype(float)
