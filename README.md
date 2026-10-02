# palm-prior

An object-centric world model of how a block moves when a hand or a gripper acts on it,
pretrained on phone clips of my own hand (including clips where the grasp fails on purpose),
adapted to a simulated Franka Panda with few robot episodes, and used for CEM-MPC.

Specification: [docs/EXPLAINER.md](docs/EXPLAINER.md). Plan: [docs/TODO.md](docs/TODO.md).
Recording protocol: [docs/FILMING_GUIDE.md](docs/FILMING_GUIDE.md).

Results and the write-up are filled in as the phases land; nothing below is a measured
claim yet.

## Setup (macOS, Apple Silicon, CPU only)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
brew install ffmpeg
uv sync --extra dev
uv run pytest
```

`uv sync` installs Python 3.11 and the dependencies into `.venv`. The Panda model comes
from MuJoCo Menagerie through the `robot_descriptions` package, which git-clones the
repository into `~/.cache/robot_descriptions` the first time it is used (about 3 minutes,
needs network once).

## What runs today

```bash
uv run python scripts/smoke_mujoco.py    # loads the Menagerie Panda, prints steps/s, writes results/smoke.png
uv run python scripts/make_board.py      # writes results/print/board_a4.pdf (ArUco board + checkerboard)
```

Every script takes `key=value` overrides for anything in `configs/default.yaml`, plus
`seed=<n>` and `overwrite=true` (scripts skip work whose output already exists).

## Layout

```
configs/default.yaml   every parameter, grouped by EXPLAINER module
src/palm_prior/        state.py (the shared 8-D state), perception/, human/, retarget/, sim/, wm/, plan/, viz/
scripts/               one entry point per pipeline stage
assets/                MJCF scene
tests/                 pytest
results/               csv, png, mp4 (committed); data/ and checkpoints/ are not
```
