# palm-prior

Forty-five phone clips of one hand putting a lid on a plate, turned into a state a simulated Franka Panda can use.

The numbers are in [RESULTS.md](RESULTS.md). Equations and file formats are in [docs/EXPLAINER.md](docs/EXPLAINER.md). How the clips were filmed is in [docs/FILMING_GUIDE.md](docs/FILMING_GUIDE.md).

## Setup

CPU only. Python 3.11, with [uv](https://docs.astral.sh/uv/). macOS, Linux and Windows.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --extra dev
```

On Linux, uv takes the CPU build of PyTorch. The published numbers are already in `results/v2/`. To check the install without the phone videos:

```bash
uv run pytest
uv run python scripts/smoke_mujoco.py overwrite=true
```

`pytest` checks the state, the palm-size fit, the retarget, the IK, and CEM on a quadratic. `smoke_mujoco.py` renders the Panda and writes `results/smoke.png`. Rebuilding the tables is `uv run python scripts/fix_all.py`. On the machine that produced these numbers that was about three quarters of an hour, and it skips clips already in `results/v2/clips/`. It does not replace `RESULTS.md`.

Five clips are in the repo so a reviewer can see the camera, the table, a success, a miss and a drop: `data/raw/calib.mp4`, `empty.mp4`, `demo_001.mp4`, `demo_031.mp4`, `demo_035.mp4`. The other 42 stay local. `calib.mp4` is 52 MB, so the push warns and still goes through. Labels are in `data/raw/recording_log.csv`, not in the filenames. Every row has `goal=plate`.

The first extraction downloads the MediaPipe hand model into `third_party/hand_landmarker.task` from the URL in `src/palm_prior/perception/hand.py`. The first MuJoCo run downloads the Menagerie Panda into `~/.cache/robot_descriptions`. On macOS the interactive viewer needs `mjpython`. The offscreen renderer used here runs under `python` on macOS, Windows and Linux. MediaPipe 1.x crashes in the hand landmarker on macOS, so the pin is `mediapipe>=0.10.14,<1.0`.

## The height was wrong. Measuring the hand fixed the trace

A single camera sees the fingers in the image and does not see their distance. The first estimate used a generic hand and none of the lengths measured on this hand.

### Demo 2. Finger height before and after measuring the hand

This is one success clip, `demo_002.mp4`, a high reach onto a plate. Time runs left to right. The vertical lines mark the grasp and the release. Gray is the old height, with no palm measurement: it runs from −1 cm to 30 cm, and the spike to 30 cm is at the release. Blue is the same frames after fitting the eight measured lengths on the back of the hand. That reach stays between 1 cm and 13 cm, about 8 cm at the grasp and 9 cm at the release.

![Demo 2. Gray has no measured hand and spikes to 30 cm at the release. Blue uses palm size](results/v2/depth/plots/clip002_height.png)

## What that changed

Every clip aims the lid at a plate. The pad and the box exist only in the simulator. The arm is tested by replaying one plate video in three ways. The names below are the labels on the next figure.

**Naive.** One map, built once from clip 1 and a single plate layout, then reused on every layout. It never looks at where the lid and the target are on that try.

**Two-anchor.** For each layout, the grasp in the video is sent to the lid and the release is sent to the target. Those two points fix the whole path, and the height is still the height from the video.

**Hybrid.** The same two pins across the table. The height of the grasp comes from the lid, and the height of the release comes from the target. A smooth curve joins them, so the gripper arrives stopped. While the fingers close and lift, the gripper stays over the lid. While they open, it stays over the target.

| | Result |
|---|---|
| Naive | 0/150 |
| Two-anchor | 56/150. Plate 23/50, pad 25/50, box 8/50 |
| Hybrid | **148/150.** Plate 50/50, pad 49/50, box 49/50 |
| Predict the lid, clips plus 10 robot tries | **7.9 cm**. The same 10 tries alone are 15.7 cm. "The lid never moves" is 14.7 cm |
| Clips alone, no robot tries | 23.5 cm, worse than guessing the lid stays still |

### Where the copied reach lands

Each bar is a success rate: the lid finished within 3 cm of the target, at the right height, with the fingers open. There are 50 layouts on a plate, 50 on a pad, and 50 on a box. The videos are all plate placements. The pad and the box are simulator targets the hand never touched. Naive is absent from the top of the chart because it lands on none of them. Hybrid is the right-hand bar in each group.

![Success rate of naive, two-anchor, and hybrid, split by simulator target](results/v2/e1/plots/success_by_goal.png)

### Whether the clips make the prediction better

Two panels, same horizontal axis: how many robot tries the model was trained on. The left panel is how many centimetres the model is wrong about where the lid will be, about one second ahead. Lower is better. The dashed line is the guess that the lid never moves. The right panel is whether the model can tell that the lid is in the hand. A score of 0.5 is a coin toss, and 1 is perfect. The band around each line is the spread across three training starts.

![Position error and attach score against the number of robot tries](results/v2/e2/plots/learning_curves.png)

### One frame of demo 2, while the lid is in the air

Frame 196 of `demo_002.mp4`, about 6.5 seconds in, with the fingers closed. White lines are the hand. The green dot is the pinch in the image. The red circle is that pinch estimated in centimetres and drawn back onto the frame. The blue box is the lid. The plate is on the table below the hand. The plot beside the frame is the old height, the gray line from the graph above: it spikes near the release. The blue line of the palm-size height is the separate graph.

![Demo 2, hand holding the lid above the plate](results/extract_demo_002.jpg)

## How it works

```mermaid
flowchart LR
  phone[Phone video of a plate placement] --> hand[Palm-size hand height]
  phone --> lid[Lid on the table]
  hand --> state[Shared state]
  lid --> state
  state --> copy[Copy the reach onto the Panda]
  state --> model[Ensemble world model]
  copy --> panda[MuJoCo Panda]
  model --> panda
```

**Hand height.** Eight measured segments on the back of the hand are fitted to the knuckles. A scale lines that fit up with the lid at the grasp and at the release. The gray line in the figure is the previous estimate, which never used those lengths.

**Copying the reach.** Hybrid, above, takes its timing from the video. The grasp is pinned to the lid and the release to the target. Near those two moments the height comes from the object, and a minimum-jerk curve joins them so the gripper arrives stopped. Two points still fix the horizontal map,

$$
T(x) = \alpha x + \beta,\quad
\alpha = \frac{P - G}{x_r - x_g},\quad
\beta = G - \alpha x_g.
$$

Worked check, in `tests/test_retarget.py`: $x_g = (0.10, 0.05)$, $x_r = (0.30, 0.15)$, lid $G = (0.50, 0.10)$, target $P = (0.60, -0.12)$ give $\alpha = -0.04 - 1.08i$ and $\beta = 0.45 + 0.21i$.

**State.** The same vector is written for the hand and for the gripper. v2 appends the target position, which stays constant inside a clip:

$$
s = [\,d_{eo},\; d_{to},\; z_{obj},\; h_{tgt},\; g,\; p_{tgt}\,],\quad
d_{eo} = p_{ee} - p_{obj},\quad
d_{to} = p_{tgt}^{xy} - p_{obj}^{xy}.
$$

That is 11 numbers. The action is the change in finger position plus the next grip, at 10 Hz. Built only through `src/palm_prior/state.py`.

**World model.** Five small networks. Input is the 11 numbers, the action, and a bit that says hand or robot. Output is how the lid moves, and whether it is held. Trained three ways: robot tries only, clips first then robot, or both from the start. The recipe that works at a small budget is both from the start.

**Simulator.** Plain MuJoCo and the Menagerie `franka_emika_panda`. The scene is `assets/scene.xml`. The gripper tendon is `actuator8`, 255 open and 0 closed. There are no sites in that model. The tool centre is injected at hand-local `[0, 0, 0.1034]`. The filmed lid is 1.7 cm tall. The simulated block is a 4 cm cube, because a 1.7 cm cube cannot be pinched by this gripper. The fingers count as open when the thumb and index are together, and closed when the lid spreads them.

Closed-loop planning was not run.

## What the files do

| File | What it does |
|---|---|
| `configs/default.yaml` | Every parameter: camera paths, the eight hand lengths, object sizes, the three copy methods, training |
| `data/raw/recording_log.csv` | Which clip is a success, a miss, a drop, a short placement, or a push, and which dots it uses. Every row is `goal=plate` |
| `data/raw/dots.yaml` | The printed plan of the nine marks, in centimetres |
| `data/calib/camera.yaml` | The phone's intrinsics, from the checkerboard |
| `data/calib/block_hsv.yaml` | The colour range of the lid |
| `src/palm_prior/perception/` | Checkerboard, ArUco table frame, MediaPipe hand, colour of the lid |
| `src/palm_prior/vision/hand_depth.py` | Finger height from the eight measured lengths. This is the blue line |
| `src/palm_prior/vision/objects.py` | Lid and plate position by intersecting a pixel ray with the table |
| `src/palm_prior/state.py` | Builds the shared state. 8 numbers, or 11 when the target position is appended |
| `src/palm_prior/retarget/naive.py` | One fixed map from clip 1, reused on every layout |
| `src/palm_prior/retarget/two_anchor.py` | Grasp sent to the lid, release sent to the target, height still from the video |
| `src/palm_prior/retarget/hybrid.py` | The same two pins, with grasp and release height taken from the objects |
| `src/palm_prior/sim/` | The MuJoCo Panda, the scene, and the inverse kinematics |
| `src/palm_prior/wm/` | The five small networks and their training |
| `src/palm_prior/fix_all.py` | The run that writes `results/v2/`. Start it with `uv run python scripts/fix_all.py` |
| `assets/scene.xml` | The table, the block, the plate, the pad, and the box |
| `docs/EXPLAINER.md` | Equations, frames, and file formats |
| `tests/` | Checks on the state, the palm-size fit, the three copies, the IK, and CEM |

`scripts/extract_human.py`, `scripts/replay.py`, and `scripts/exp_e2.py` are the 7 October path. They write `results/*.csv`, not `results/v2/`.

## What the outputs are

The numbers in [RESULTS.md](RESULTS.md) are the `results/v2/` column. `results/v1/` is the same kind of table from 7 October, before palm size and hybrid.

| Output | What is in it |
|---|---|
| `results/v2/clips/clip_NNN.npz` | One phone clip in centimetres. `p_ee` is the palm-size pinch. `p_ee_old` is the gray line. Also the lid path, the gripper bit, the grasp and release frames |
| `results/v2/depth/depth_check.csv` | Per clip: scale factors `kappa_g` and `kappa_r`, fraction of frames with a usable fit, release error in centimetres |
| `results/v2/depth/plots/clip002_height.png` | Demo 2. Gray is the old height. Blue is palm size |
| `results/v2/qc/extraction_report.csv` | One row per clip: visibility, grasp count, pass or fail. 45 of 45 pass |
| `results/v2/e1/e1.csv` | One row per layout. Columns `method` (naive, two_anchor, hybrid), `goal`, `seed`, `success`, `xy_err_m` |
| `results/v2/e1/plots/success_by_goal.png` | Those success rates, split by plate, pad, and box |
| `results/v2/robot_data/episodes.npz` | The 400 robot tries made by replaying hybrid. This is what the world model trains on |
| `results/v2/e2/e2.csv` | One row per training. Columns `variant`, `N` (robot tries), `seed`, `pos_err_cm`, `attach_auroc` |
| `results/v2/e2/plots/learning_curves.png` | Those two scores against the number of robot tries |
| `RESULTS.md` | The written account of the tables above. The pipeline does not replace this file |

`data/human/`, `data/robot/`, `checkpoints/`, `.venv/` and `third_party/` are gitignored, as are the phone videos other than the five named above.

License: MIT. Copyright Luc Mantoux.
