"""Draw a world-model rollout back onto a held-out phone clip.

The model (human embedding) is rolled from the first state with the real hand
actions. Blue is the measured block, orange is the prediction.

Usage:
    uv run python scripts/fig_human_overlay.py
    uv run python scripts/fig_human_overlay.py clip=10
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

import _bootstrap  # noqa: F401

from palm_prior.human.transitions import _interp_rows, _nearest, sample_times  # noqa: E402
from palm_prior.perception.aruco import board_pose, build_board, make_detector, project, static_pose  # noqa: E402
from palm_prior.state import next_state  # noqa: E402
from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_torch_threads, video_frames, video_info  # noqa: E402
from palm_prior.wm.model import Ensemble, NormStats, rollout  # noqa: E402
from extract_human import _file_map  # noqa: E402
from exp_e2 import _heldout_clips  # noqa: E402


def _kept_objects(data, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured block positions and times for the transitions that were saved.

    returns t (K,), p_obj (K, 3), actions (K, 4).
    """
    t_frame = np.asarray(data["t_frame"], float)
    duration = float(t_frame[-1] + np.median(np.diff(t_frame)))
    t_q = sample_times(duration, float(cfg.human.rate_hz), float(cfg.human.tau))
    ee_i = _interp_rows(t_frame, np.asarray(data["p_ee"], float), t_q)
    obj_i = _interp_rows(t_frame, np.asarray(data["p_obj"], float), t_q)
    gg = _nearest(t_frame, np.asarray(data["g"], float), t_q)
    good = np.isfinite(ee_i).all(1) & np.isfinite(obj_i).all(1) & np.isfinite(gg)
    pair = good[:-1] & good[1:]
    obj_k = obj_i[:-1][pair]
    assert len(obj_k) == len(data["s"]), (len(obj_k), len(data["s"]))
    return t_q[:-1][pair], obj_k, np.asarray(data["a"], np.float32)


@torch.no_grad()
def _predict(model, stats, s0, actions) -> np.ndarray:
    """Per-member open-loop positions, averaged. returns (K+1, 3) starting at s0's block.

    The caller passes the measured block at step 0 separately; this returns dp sums.
    returns (K, 3) cumulative predicted displacement.
    """
    device = next(model.parameters()).device
    m = model.n_members
    state = torch.as_tensor(s0, dtype=torch.float32, device=device).view(1, 1, 8).expand(m, 1, 8).clone()
    cum = []
    total = torch.zeros(m, 1, 3, device=device)
    for k in range(len(actions)):
        act = torch.as_tensor(actions[k], dtype=torch.float32, device=device).view(1, 1, 1, 4).expand(m, 1, 1, 4)
        dp, _ = rollout(model, state, act, 0.0, stats)
        total = total + dp[:, :, 0, :]
        cum.append(total.mean(dim=0)[0].cpu().numpy())
        state = next_state(state, act[:, :, 0, :], dp[:, :, 0, :])
    return np.stack(cum)


def _board(video, cfg, K, dist):
    tb = build_board(cfg)
    detector = make_detector(tb)
    poses = []
    for i, frame in video_frames(video):
        if i % 10 != 0:
            continue
        pose = board_pose(frame, tb, detector, K, dist, int(cfg.aruco.min_markers))
        if pose is not None:
            poses.append(pose)
        if len(poses) >= 20:
            break
    if not poses:
        raise RuntimeError(f"board never detected in {video.name}")
    return static_pose(poses)


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_torch_threads()
    ckpt = resolve(cfg.paths.checkpoints)
    path = ckpt / "wm_pretrain_ft_N0_seed0.pt"
    if not path.exists():
        path = ckpt / "_human_pretrain_seed0.pt"
    if not path.exists():
        raise SystemExit("missing the human-only checkpoint; run scripts/exp_e2.py first")
    blob = torch.load(path, weights_only=False)
    model = Ensemble(int(cfg.wm.n_members), tuple(int(h) for h in blob.get("cfg_hidden", cfg.wm.hidden)))
    model.load_state_dict(blob["model"])
    model.eval()
    stats = NormStats.load(blob["stats"])

    human_z = np.load(resolve(cfg.paths.human_dir) / "transitions.npz")
    hold = _heldout_clips(human_z["clip_id"], human_z["is_failure"], int(cfg.human.n_heldout_clips), int(cfg.seed))
    chosen = cfg.get("clip")
    clip_id = int(chosen) if chosen is not None else int(sorted(hold)[0])
    raw = resolve(cfg.paths.raw_dir)
    video = raw / _file_map(raw).get(clip_id, f"demo_{clip_id:03d}.mp4")
    data = np.load(resolve(cfg.paths.human_dir) / f"clip_{clip_id:03d}.npz")

    cam = OmegaConf.load(resolve(cfg.paths.camera_yaml))
    K = np.array(cam.K, float)
    dist = np.array(cam.dist, float)
    with Timer("fig_human_overlay"):
        times, measured, actions = _kept_objects(data, cfg)
        cum = _predict(model, stats, np.asarray(data["s"][0], np.float32), actions)
        predicted = measured[0] + cum
        # The displacement at step k lands on the block at the next sample.
        pred_at = np.vstack([measured[0], predicted])
        meas_at = np.vstack([measured, measured[-1]])
        t_at = np.concatenate([times, [times[-1] + (times[-1] - times[-2])]])
        R_CT, t_CT = _board(video, cfg, K, dist)

        info = video_info(video)
        out = ensure_dir(resolve(cfg.paths.results)) / f"human_overlay_clip{clip_id:03d}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
        try:
            for i, frame in video_frames(video):
                t = i / info.fps
                k = int(np.argmin(np.abs(t_at - t)))
                for point, colour in ((meas_at[k], (255, 80, 40)), (pred_at[k], (40, 140, 255))):
                    if not np.isfinite(point).all():
                        continue
                    if np.linalg.norm(point) > 5.0:
                        continue
                    uv = np.asarray(project(point, R_CT, t_CT, K, dist), float).reshape(-1)
                    u, v = float(uv[0]), float(uv[1])
                    if not (np.isfinite(u) and np.isfinite(v) and -2000 <= u <= info.width + 2000 and -2000 <= v <= info.height + 2000):
                        continue
                    cv2.circle(frame, (int(round(u)), int(round(v))), 14, colour, 2)
                cv2.putText(frame, "blue measured   orange predicted", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                writer.write(frame)
        finally:
            writer.release()
    print(f"clip {clip_id}  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
