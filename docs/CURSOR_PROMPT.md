# palm-prior — Cursor Prompt Pack (Mac-native, novel version)

## How to use
1. Put `EXPLAINER.md`, `FILMING_GUIDE.md` and `TODO.md` in `docs/`. EXPLAINER is the spec.
2. Paste the **Master Rule** into `.cursor/rules/project.mdc`.
3. Agent mode, **one phase at a time**; move on only when its acceptance checks pass on your Mac.
4. Strongest model for Phases 3, 4a, 6, 7.
5. Bugs → **Debug Template**.

---

## Master Rule (`.cursor/rules/project.mdc`)

```
---
description: palm-prior rules
alwaysApply: true
---
# Project
palm-prior: I learn an object-centric world model of "how a block moves when a hand/gripper acts" from 45 phone clips of my own hand (30 successes + 15 deliberate failures), adapt it to a simulated Franka Panda with few robot episodes, and use it for CEM-MPC seeded by a two-anchor retargeted path from my clips.
docs/EXPLAINER.md is the source of truth for state/action definitions, equations, frames, file formats and defaults. If code must deviate, STOP and explain first.

# Platform
macOS on Apple Silicon, CPU only. Python 3.11 managed by uv. No CUDA, no Colab, no robosuite, no LIBERO.
Sim = plain MuJoCo (pip `mujoco`) + MuJoCo Menagerie franka_emika_panda (via the `robot_descriptions` package, or a git clone into third_party/ — pick one and document it). Offscreen rendering via mujoco.Renderer; the interactive viewer runs under `mjpython`.
torch on CPU (models are tiny). Use torch.set_num_threads sensibly.

# Stack (ask + justify before adding anything)
numpy, scipy, opencv-contrib-python, mediapipe, mujoco, robot_descriptions, torch, omegaconf, matplotlib, imageio[ffmpeg], tqdm, pytest.

# Code rules
- Layout: src/palm_prior/{perception,human,retarget,sim,wm,plan,viz}/, scripts/, assets/ (MJCF), configs/default.yaml, tests/.
- All parameters in configs/default.yaml (omegaconf), overridable by key=value.
- Units m, rad, s. Frames C/T/W. Name variables with their frame (p_T, R_CT). State s is ALWAYS the 8-D vector in EXPLAINER §2, in that order; build/decode it only through src/palm_prior/state.py.
- Docstrings state shapes and frames; assert shapes at function boundaries.
- Every script: seed=, resumable (skip existing outputs unless overwrite=true), saves a debug plot or mp4, prints its runtime.
- Pure small functions for math; pytest for state.py, geometry, retarget, IK, CEM.
- No placeholder code, no TODO stubs, no invented numbers anywhere. No abstraction for one-off use.

# After every task reply with
1) files changed 2) exact commands 3) the expected output and how to spot a wrong result 4) API/asset names you're unsure about that I must verify (especially Menagerie actuator/site names and MediaPipe Tasks API).
```

---

## Phase 0 — Scaffold + Mac environment
```
Read docs/EXPLAINER.md fully. Create:
- pyproject.toml for uv (python 3.11, deps from the rules), package src/palm_prior with the subpackages, scripts/, assets/, tests/, results/.gitkeep, data/.gitkeep, .gitignore (data/, checkpoints/, third_party/, large mp4s except results/ and docs/media/).
- configs/default.yaml with EVERY default in EXPLAINER, grouped by module, one comment per key; include block_size, plate_radius, dot file path, the target heights.
- src/palm_prior/utils.py (load_config, set_seed, ensure_dir, write_video, Timer).
- src/palm_prior/state.py: build_state(p_ee, p_obj, p_tgt_xy, h_tgt, g) -> (8,), next_state(s, a, dp_obj) per the EXPLAINER §2 update equations, batched (works on (...,8)), numpy and torch versions. tests/test_state.py reproducing the EXPLAINER §2 worked example exactly, plus a translation-invariance test (shifting all positions by the same xy leaves s unchanged).
- scripts/smoke_mujoco.py: load the Menagerie Panda scene, step 500 times, render one 480x360 frame to results/smoke.png, print the steps/s and the actuator names.
Acceptance: `uv sync`, `uv run pytest` green, smoke.png shows the Panda, steps/s printed.
```

## Phase 1 — Calibration, ArUco, previews
```
EXPLAINER Modules 1–2 plus preview tools.
- scripts/make_board.py: A4 PDF at 300 dpi: ArUco GridBoard (DICT_4X4_50, 2x2, 60 mm, 15 mm gap) with "ORIGIN" + axis arrows (x right, y up) printed at the table-frame origin corner; second page: checkerboard 9x6 / 25 mm. Print the expected sizes.
- scripts/calibrate.py: video → ~40 well-spread frames → calibrateCamera → data/calib/camera.yaml + results/calib_check.png.
- perception/aruco.py: board_pose, static_pose (median, scipy Rotation mean), ray(u,v,K,dist) with undistortion, intersect_plane_z(ray_C, R_CT, t_CT, h) -> X_T. tests/test_geometry.py with synthetic poses (round-trip error < 1e-6).
- scripts/debug_aruco.py: board axes over a video.
- perception/block.py: fit_hsv(frame, roi) from empty.mp4 (I click the block once; OpenCV window), mask(frame) → largest blob centroid + area.
- scripts/preview_hand.py: on one video, overlay MediaPipe HandLandmarker landmarks (Tasks API, VIDEO mode; auto-download the .task model into third_party/) with the thumb/index tips highlighted + the block mask contour. Save the mp4.
Acceptance: tests green; RMS < 0.5 px; stable axes; landmarks and block mask clean on my test clip.
```

## Phase 2 — Human state extraction
```
EXPLAINER Modules 3–5.
- perception/hand.py: per frame image + world landmarks, solvePnP → pinch point; scale k from the rest frames (first and last 2 s); aperture + adaptive hysteresis; gap fill ≤ 10 frames; Savitzky-Golay(9,2).
- perception/block_track.py: per frame block position in T: on the table → intersect the ray with z = block_size/2; closed and lifted → depth-matched to the pinch point (EXPLAINER Module 4); occlusion rule; gap fill; smoothing.
- human/transitions.py + scripts/extract_human.py: reads data/raw/recording_log.csv and dots.yaml; target xy from the plate dot, h_tgt from the goal; time stretch tau, 10 Hz sampling; builds s, a, dp_obj, attach labels (same rule as the robot), clip_id, is_failure, clip type → data/human/transitions.npz. Also a per-clip raw npz.
- QC → results/extraction_report.csv: valid ratios (hand, block), n grasp events, the reprojection error, the block start vs dots.yaml error, pass flag. Rules from EXPLAINER + require exactly one grasp for success/F1/F2/F3 clips and zero for F4.
- Debug mp4 per clip: landmarks, green pinch, red reprojection, blue block box, OPEN/CLOSED, side plot of hand z, block z, aperture with a moving cursor.
Acceptance on 3 clips (1 success, 1 F1, 1 F2): the checks listed in docs/TODO.md Day 2 all hold.
```

## Phase 3 — Two-anchor retarget (tests first)
```
EXPLAINER Module 6. Write tests/test_retarget.py FIRST:
- anchors exact (atol 1e-9); the worked example: alpha = -0.04-1.08i, beta = 0.45+0.21i (atol 1e-3), T(x_r) = P
- degenerate → alpha = 1; kappa clipped; h_tgt added after release
- naive_global takes no object positions
Then retarget/two_anchor.py (complex numbers), retarget/naive.py, retarget/build_plan.py: build_plan(clip, block_W, tgt_W, h_tgt, residual_knots (8,3) or None, timing_shift (2,) or None, cfg, ee_start_W) -> dict(p_star (K,3) at 10 Hz, grip (K,), k_close, k_open). Residual = linear interpolation of the knots (knot 0 fixed at 0). scripts/plot_retarget.py: top view, human (T) vs robot (W).
Acceptance: tests green; plot looks right.
```

## Phase 4 — MuJoCo sim
### 4a. Scene + IK
```
EXPLAINER Module 7.
- assets/scene.xml: include the Menagerie Panda; floor; block (free joint, size from config, 50 g, friction 1.5); plate (static cylinder, radius from config, 1.5 cm tall); pad (static 10x10x0.5 cm); box (static 8 cm cube); option timestep 0.002, implicitfast, cone elliptic, impratio 10; a `tcp` site between the fingertips (verify the offset by rendering a small sphere at the site); an agentview-like camera facing the robot from the front, ~45° down.
- sim/env.py: Env(goal, seed) → reset() samples the layout (EXPLAINER ranges, ≥ 15 cm apart, objects not overlapping), moves the static targets by editing model body positions before mj_forward; get_state() → p_ee_W, p_obj_W, tgt_xy, h_tgt, g; step(a) where a = 4-D action from EXPLAINER §2: the EE target = current + a_xyz (clipped to 2.5 cm/step), runs DLS IK every 10 substeps for 50 substeps, gripper ctrl from a_g; success(); render(cam, w, h); set_block_pose() for disturbance tests; save/restore full MjData state.
- sim/ik.py: DLS with nullspace posture term exactly as EXPLAINER, orientation error keeps the gripper pointing down with a fixed yaw. tests/test_ik.py: reaching 5 random targets in the workspace to < 5 mm within 30 control steps.
- scripts/view_scene.py (run with mjpython) and scripts/render_layouts.py (6 random layouts → a png grid).
Acceptance: IK test green; the layouts png looks right; the tcp sphere sits between the fingers.
```
### 4b. Scripted check
```
scripts/scripted_check.py: waypoint pick-place per goal using env.step, 10 seeds each, success printed, mp4s saved. Config: approach height, grasp height, lift height, dwell steps. Add --sweep over the grasp height and the max step size.
Acceptance: ≥ 9/10 per goal.
```
### 4c. Replay + E1
```
scripts/replay.py: methods {naive, two_anchor}, layout seeds 0–49 × goals, QC-passed SUCCESS clips round-robin, open-loop tracking of p_star via env.step (action = p_star[k+1] - p_ee, clipped). → results/e1.csv + printed summary. 3 side-by-side mp4s: phone debug clip (time-aligned using tau) | sim render.
Acceptance: CSV, summary, aligned videos.
```

## Phase 5 — Robot dataset
```
EXPLAINER Module 8. scripts/generate_robot.py: episodes with seeds 1000+ (train) and 900–999 (held-out), random goal + random success clip + smooth knot noise + failure injection (prob 0.2: early release at a random time OR close at a 2–3 cm xy offset), no rendering, multiprocessing. Log per step: s, a, dp_obj, attach (same rule as human), episode id, outcome. → data/robot/episodes.npz + nested subsets index for N ∈ {0,5,10,25,50,100,300}. Print the outcome mix and runtime.
Acceptance: train 300 + held-out 100 episodes in a few minutes; the success fraction printed.
```

## Phase 6 — World model + E2 + figures
```
EXPLAINER Module 9.
- wm/model.py: an ensemble of M=5 MLPs (13 → 128x3 SiLU → 4), vectorized in one module (batched weights, or a loop if simpler and fast enough). Input normalization stats fit on the training data and stored in the checkpoint. Outputs dp_obj (de-normalized) + attach logit. predict(s, a, emb) returns per-member outputs; rollout(s0, actions (B,H,4), emb) returns per-member state trajectories using state.next_state.
- wm/train.py: bootstrap per member; 5-step self-rollout loss (MSE on dp_obj + 0.5·BCE attach); variants: sim_only(N), pretrain_ft(N) (human pretrain → fine-tune lr x0.3 with 20% human batches), cotrain(N). Clips split: 5 held-out human clips (incl. ≥ 1 failure).
- scripts/train_wm.py and scripts/exp_e2.py: for every variant × N ∈ {0,5,10,25,50,100,300} × 3 seeds: 10-step open-loop block position error (cm) on held-out robot episodes + attach AUROC → results/e2.csv, results/e2_curve.png (log x, mean ± std band). Include the trivial baseline "block never moves".
- scripts/fig_gap_heatmap.py: grid of grasp xy offsets (±3 cm): start state above the block, actions = descend, close, lift 5 steps; predicted lift probability for human-only, pretrain_ft(25) and the TRUE sim (actually run it) → results/gap_heatmap.png (3 panels).
- scripts/fig_human_overlay.py: on a held-out human clip, roll the WM (human embedding) from the first state using the real hand actions; reproject the predicted block positions into the video frames (blue = measured, orange = predicted) → mp4.
- scripts/exp_e6.py: pretrain with vs without failure clips → attach AUROC on robot failure states → results/e6.csv.
Acceptance: training ≈ 1 min per model; the WM beats "never moves"; the plots render. Show me the e2 curve before any tuning.
```

## Phase 7 — CEM-MPC + control experiments
```
EXPLAINER Module 10.
- plan/cem.py: generic batched CEM with mixed continuous + integer dims (round the integers), warm start, sigma floor. tests/test_cem.py on a 2-D quadratic with a known optimum.
- plan/mpc.py: shrinking-horizon MPC: decision vars = 8x3 knots + 2 timing shifts; candidate → build_plan with the residual → actions → wm.rollout over all members → cost J (EXPLAINER, beta, w_s from config) → execute the first 5 actions in the env → re-plan with a warm start. Model options: wm checkpoint OR "true_sim" (rolls candidates out in a cloned MjData; fewer samples, from config). Prior options: human (retargeted clip) OR isotropic (straight-line scripted path with the same timing).
- scripts/eval_control.py: method spec (open_loop_naive, open_loop_two_anchor, mpc[model=..., prior=..., samples=...]), goals, seeds 0–49 x repeats, disturbance options (push the block 5 cm at step 15; Gaussian noise on the object positions given to the planner). Writes results/e3.csv, e4.csv, e5.csv, e6_control.csv, e7.csv; printed tables mean ± std; mean planning ms per re-plan.
- Before the big evals: scripts/watch_mpc.py renders 5 episodes with the imagined block path as small translucent spheres (add geoms to the renderer scene) → mp4.
Acceptance: watch_mpc videos make sense; the tables print; no hardcoded numbers.
```

## Phase 8 — Figures, reel, README
```
- scripts/make_figures.py: every figure from results/*.csv only; white background, large fonts, consistent colours (human-pretrained = one fixed colour everywhere).
- scripts/make_reel.py: ~60 s mp4 + GIF: phone clip → extracted 3D hand/block plot → robot replay → "imagination" spheres video → E2 curve → E3 table.
- scripts/fill_readme.py: README.md from a template with {{e2_summary}}, {{e3_table}} etc. read from the CSVs; leave "What worked / what didn't" as headings for me.
Style: short sentences, no marketing words, no emoji, every claim tied to a number or figure.
```

## Stretch — Distill MPC into a policy
```
Collect (s, executed MPC action) from 200 MPC episodes on seeds 2000+; train an MLP policy (8 → 256x2 → 4, gripper as a logit); evaluate on seeds 0–49; report success and ms per step vs MPC → results/distill.csv.
```

---

## Debug Template
```
Bug in Phase <N>, file <path>.
Ran: <command>
Expected (spec/acceptance): <...>
Got: <full traceback or wrong output>
Already checked: <...>
Don't touch unrelated code. Give the 2–3 most likely causes ranked, each with a minimal check. Fix only the confirmed one.
```

## Final review prompt
```
Review the repo against docs/EXPLAINER.md and docs/TODO.md: any equation or state definition implemented differently? README numbers not traceable to results/*.csv? dead code or debug prints? Does a fresh clone + `uv sync` + the README commands work on macOS (verify every path)? Any marketing language? List findings only, change nothing.
```
