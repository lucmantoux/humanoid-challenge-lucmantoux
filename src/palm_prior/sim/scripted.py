"""A hand-written pick-and-place, used as the simulator's reference behaviour.

This is not a policy anyone learns from. It exists to answer one question: can this arm,
this gripper and these friction settings pick the block up and put it down at all? If
the scripted version cannot, nothing built on top of the simulator means anything.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Waypoint(NamedTuple):
    """Hold the tcp at p_W with the gripper at g for n_steps control steps.

    p_W (3,) world frame; g is 1 closed, 0 open; n_steps at 10 Hz.
    """

    p_W: np.ndarray
    g: float
    n_steps: int
    label: str


def pick_place(
    block_xy: np.ndarray, target_xy: np.ndarray, h_tgt: float, p_ee_start: np.ndarray, cfg
) -> list[Waypoint]:
    """The eight waypoints of a pick and place. All arguments in the world frame W.

    block_xy (2,), target_xy (2,), h_tgt the top height of the goal object, p_ee_start
    (3,) where the tcp is now, used only to budget steps for the first move.

    Each leg gets as many steps as the tcp needs at sim.max_step, plus settle_steps so
    the tracker can converge before the next leg starts pulling it somewhere else.
    """
    block_xy = np.asarray(block_xy, float)
    target_xy = np.asarray(target_xy, float)
    p_ee_start = np.asarray(p_ee_start, float)
    assert block_xy.shape == (2,) and target_xy.shape == (2,), (block_xy.shape, target_xy.shape)
    assert p_ee_start.shape == (3,), p_ee_start.shape

    sc = cfg.scripted
    b = float(cfg.objects.block_size)
    grasp_z = b / 2 + float(sc.grasp_offset)
    hover_z = b + float(sc.approach_height)
    carry_z = float(sc.carry_height)
    release_z = h_tgt + b / 2 + float(sc.release_clearance)

    legs = [
        (np.array([*block_xy, hover_z]), 0.0, "approach"),
        (np.array([*block_xy, grasp_z]), 0.0, "descend"),
        (np.array([*block_xy, grasp_z]), 1.0, "close"),
        (np.array([*block_xy, carry_z]), 1.0, "lift"),
        (np.array([*target_xy, carry_z]), 1.0, "traverse"),
        (np.array([*target_xy, release_z]), 1.0, "place"),
        (np.array([*target_xy, release_z]), 0.0, "open"),
        (np.array([*target_xy, hover_z + h_tgt]), 0.0, "retreat"),
    ]

    max_step = float(cfg.sim.max_step)
    settle = int(sc.settle_steps)
    dwell = int(sc.dwell_steps)

    waypoints = []
    previous = p_ee_start
    for p_W, g, label in legs:
        travel = math.ceil(float(np.linalg.norm(p_W - previous)) / max_step)
        n_steps = dwell if label in ("close", "open") else travel + settle
        waypoints.append(Waypoint(p_W=p_W, g=g, n_steps=max(n_steps, 1), label=label))
        previous = p_W
    return waypoints


def run(env, waypoints: list[Waypoint], on_step=None) -> int:
    """Execute the waypoints with proportional tracking. Returns the steps taken.

    The action is simply the vector to the waypoint; Env.step clips it to sim.max_step,
    so a far waypoint becomes a straight run at full speed and a near one a gentle
    approach, with no gain to tune.
    """
    total = 0
    for wp in waypoints:
        for _ in range(wp.n_steps):
            env.step(np.concatenate([wp.p_W - env.p_ee_W(), [wp.g]]))
            total += 1
            if on_step is not None:
                on_step(wp, total)
    return total
