"""Config loading, seeding, paths, video writing and timing."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(overrides: Sequence[str] | None = None, path: str | Path = DEFAULT_CONFIG) -> DictConfig:
    """Load configs/default.yaml and apply key=value overrides (omegaconf dotlist)."""
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (torch imported lazily so non-torch scripts stay fast)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def set_torch_threads(n: int | None = None) -> int:
    """Limit torch to n threads (default: half the logical cores, at least 1).

    The models are tiny; more threads cost more than they save.
    """
    import torch

    if n is None:
        n = max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(n)
    return n


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (parents included) and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def should_skip(output: str | Path, overwrite: bool) -> bool:
    """True when `output` already exists and we are not overwriting (scripts are resumable)."""
    exists = Path(output).exists()
    if exists and not overwrite:
        print(f"skip: {output} exists (pass overwrite=true to redo)")
    return exists and not overwrite


def write_video(path: str | Path, frames: Sequence[np.ndarray], fps: float) -> Path:
    """Write RGB uint8 frames (H, W, 3) to an mp4. Returns the path."""
    import imageio.v2 as imageio

    assert len(frames) > 0, "no frames to write"
    h, w = frames[0].shape[:2]
    assert all(f.shape == (h, w, 3) and f.dtype == np.uint8 for f in frames), "frames must be uint8 (H, W, 3)"
    out = Path(path)
    ensure_dir(out.parent)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w_:
        for f in frames:
            w_.append_data(f)
    return out


class VideoInfo(NamedTuple):
    """n_frames may be 0 if the container does not report it."""

    n_frames: int
    fps: float
    width: int
    height: int


def video_info(path: str | Path) -> VideoInfo:
    """Read frame count, fps and size without decoding the whole file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def video_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) from a video, keeping every `stride`-th frame."""
    import cv2

    assert stride >= 1, stride
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if i % stride == 0:
                yield i, frame
            i += 1
    finally:
        cap.release()


class Timer:
    """Context manager printing the wall-clock runtime of a block.

    with Timer("extract"):
        ...
    """

    def __init__(self, label: str):
        self.label = label
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        print(f"[{self.label}] {self.seconds:.3f} s")
