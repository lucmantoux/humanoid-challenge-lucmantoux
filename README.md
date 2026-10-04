# palm-prior

An object-centric world model of how a block moves when a hand or a gripper acts on it,
pretrained on phone clips of my own hand (including clips where the grasp fails on purpose),
adapted to a simulated Franka Panda with few robot episodes, and used for CEM-MPC.

Specification: [docs/EXPLAINER.md](docs/EXPLAINER.md). Plan: [docs/TODO.md](docs/TODO.md).
Recording protocol: [docs/FILMING_GUIDE.md](docs/FILMING_GUIDE.md).

Results and the write-up are filled in as the phases land; nothing below is a measured
claim yet.

## Setup

CPU only, no GPU anywhere. Developed and tested on macOS 15 / Apple Silicon; the same
two commands work on Windows and Linux.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is missing
uv sync --extra dev
uv run pytest
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --extra dev
uv run pytest
```

`uv sync` downloads Python 3.11 and every dependency into `.venv`; nothing else needs to
be installed. FFmpeg is not a system requirement — OpenCV and `imageio-ffmpeg` both ship
their own. On Linux, torch resolves to the `+cpu` build so the install stays under a
gigabyte instead of pulling the CUDA stack.

The Panda model comes from MuJoCo Menagerie through the `robot_descriptions` package,
which git-clones the repository into `~/.cache/robot_descriptions` the first time it runs
(about 3 minutes, needs network once).

The one platform difference: the interactive MuJoCo viewer must be launched with
`mjpython` on macOS and with plain `python` on Windows and Linux. Offscreen rendering,
which is what every script in this repo uses, is identical everywhere.

`uv sync` installs the package in editable mode through a `.pth` file in `.venv`. Python
ignores a `.pth` file that the filesystem marks hidden, which macOS does to everything
under an iCloud-synced `.venv`, so neither the tests nor the scripts lean on it: pytest
reads `pythonpath = ["src"]` and the scripts import `scripts/_bootstrap.py`.

## What runs today

Print pack and environment:

```bash
uv run python scripts/smoke_mujoco.py    # loads the Menagerie Panda, prints steps/s, writes results/smoke.png
uv run python scripts/make_board.py      # writes results/print/board_a4.pdf (ArUco board + checkerboard)
```

Perception, once the clips from [docs/FILMING_GUIDE.md](docs/FILMING_GUIDE.md) are in `data/raw/`:

```bash
uv run python scripts/calibrate.py       # calib.mp4 -> data/calib/camera.yaml + results/calib_check.png
uv run python scripts/debug_aruco.py clip=1    # table-frame axes over a clip, prints the origin jitter
uv run python scripts/preview_hand.py clip=1   # hand landmarks + block mask over a clip
```

Simulator, which needs no footage:

```bash
uv run python scripts/render_layouts.py  # 6 sampled layouts -> results/layouts.png
uv run mjpython scripts/view_scene.py    # interactive viewer; plain `python` on Windows and Linux
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
