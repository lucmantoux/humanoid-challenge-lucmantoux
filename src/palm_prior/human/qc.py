"""QC grasp rule that keeps deliberate re-grasps (FIX.md §6)."""

from __future__ import annotations


def grasp_passes(clip_type: str, n_grasp: int, allow: list[str], max_grasps: int) -> bool:
    """Success needs exactly one grasp. Labels in `allow` accept 1..max_grasps.

    F4 is a push. It is in `allow` so a spread hand is not rejected, and it still
    passes when the count is zero.
    """
    n_grasp = int(n_grasp)
    if clip_type == "success":
        return n_grasp == 1
    if clip_type in allow:
        if clip_type == "F4" and n_grasp == 0:
            return True
        return 1 <= n_grasp <= int(max_grasps)
    return False
