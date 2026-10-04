# Task list — do these in order

Deadline: Friday 9 October 2026, 23:59 BST.

"Ask Cursor: Phase N" means open Agent mode and paste the Phase N block from
`docs/CURSOR_PROMPT.md`. Do one phase at a time.

Every task has an **Expect** line: the thing you should be looking at before you move to
the next one. If you do not see it, stop and fix it there. Carrying a broken step forward
is what costs days.

✂️ marks a fallback: if you hit it, take the fallback and keep moving.

**Stop point.** Tasks 1–29 are a complete, submittable project on their own: human state
extraction, simulator, E1, E2. Everything after task 29 is upside. Protect tasks 1–29.

---

## Done

- [x] uv installed, `uv sync --extra dev` works, `uv run pytest` green
- [x] Phase 0: package scaffold, `configs/default.yaml`, the shared 8-D state, MuJoCo
      smoke test (`results/smoke.png` shows the Panda)
- [x] `results/print/board_a4.pdf` generated
- [x] `data/raw/recording_log.csv` and `data/raw/dots.yaml` filled in

---

## Perception scripts

1. [x] Ask Cursor: **Phase 1**.
       **Expect:** `scripts/calibrate.py`, `scripts/debug_aruco.py`,
       `scripts/preview_hand.py`, `src/palm_prior/perception/block.py` all exist, and the
       agent reports what it could and could not verify without video.
2. [x] Run `uv run pytest`.
       **Expect:** 34 tests pass, including synthetic camera-pose and calibration tests.
       Nothing here has touched real video yet, so green tests mean the maths is right,
       not that your camera works.

## Film the dataset — follow `docs/FILMING_GUIDE.md` literally

3. [ ] Guide Step 1 — print the board, measure it, put your block size and plate radius
       into `configs/default.yaml`.
       **Expect:** a printed marker square measures exactly 60 mm and a checkerboard
       square exactly 25 mm. Anything else and every distance in the project is wrong.
4. [ ] Guide Step 2 — tape the dots and the board.
       **Expect:** 9 dots on a 15 cm grid, dot 7 at 22 cm right and 10 cm away from the
       board's ORIGIN corner, matching `data/raw/dots.yaml`.
5. [ ] Guide Step 3 — position the phone.
       **Expect:** the preview shows all 9 dots, the whole board, the rest mark and about
       10 cm of margin. The tripod is taped down.
6. [ ] Guide Step 4 — phone settings.
       **Expect:** a yellow **AE/AF LOCK** banner at the top of the Camera app, 1080p 30,
       1× lens, and the brightness no longer changing when you move your hand into frame.
7. [ ] Guide Step 5 — record `calib.mp4`.
       **Expect:** about 40 seconds of footage in which the whole checkerboard is visible
       in every frame, at many angles, sizes and screen positions.
8. [ ] Guide Step 6 — record `empty.mp4`.
       **Expect:** 5 seconds of the block on dot 5 and the plate on dot 9, no hands.
9. [ ] Guide Step 7 — the test clip and all five of its checks.
       **Expect:** calibration RMS below 0.5 px, board-origin jitter below 2 px, hand
       landmarks tracking, both fingertips visible at the grasp, and a clean block mask.
       **This is the single most important gate in the project.** Everything downstream
       assumes a fixed camera and a known scale.
       ✂️ Still failing after 45 minutes: switch to the Blackmagic Camera app
       (Guide Step 10) and re-record `calib.mp4` first.
10. [ ] Guide Step 8 — record clips 1–45.
        **Expect:** 45 clips of about 10 s each, 30 successes and 15 failures, with 2 s of
        a motionless hand at the start and end of every one.
11. [ ] Guide Step 9 — transfer, rename, back up `data/raw/` outside the repo.
        **Expect:** `demo_001.mp4` … `demo_045.mp4`, `calib.mp4`, `empty.mp4`, and a copy
        of the whole folder somewhere that is not this computer. Once the phone moves,
        none of this can be re-recorded.

## Simulator

12. [ ] Ask Cursor: **Phase 4a** (scene + IK).
        **Expect:** `assets/scene.xml` and `src/palm_prior/sim/{env,ik}.py` exist.
13. [ ] Look at the scene:
        macOS `uv run mjpython scripts/view_scene.py`,
        Windows and Linux `uv run python scripts/view_scene.py`.
        **Expect:** the `tcp` marker sphere sits **between the two fingertips**, not
        inside the wrist, and the block, plate, pad and box rest on the floor without
        sinking into it or overlapping each other.
14. [ ] Run `uv run pytest tests/test_ik.py`.
        **Expect:** 5 random targets reached to under 5 mm within 30 control steps. If the
        arm reaches them but slowly, the damping is too high; if it thrashes, too low.
15. [ ] Run `uv run python scripts/render_layouts.py`.
        **Expect:** a grid of 6 images, each with the block and the target at least 15 cm
        apart, both inside the arm's reach, nothing overlapping.
16. [ ] Ask Cursor: **Phase 4b** (scripted pick-and-place).
        **Expect:** `scripts/scripted_check.py` exists and takes a goal and a seed.
17. [ ] Run `uv run python scripts/scripted_check.py`.
        **Expect:** **at least 9 out of 10 succeed on each goal**, and in
        `results/scripted_check.mp4` the gripper closes on the block, lifts it cleanly and
        releases it on the target. If the block squirts out of the fingers, that is the
        friction problem below, not a planning problem.
        ✂️ Grasps slipping: `objects.block_friction=2.0`, `sim.impratio=20`,
        `sim.max_step=0.015`. Still failing after an hour: `objects.block_size=0.03`,
        `objects.block_mass=0.03`.

## Phone video to state

18. [ ] Ask Cursor: **Phase 2**.
        **Expect:** `perception/block_track.py`, `human/transitions.py` and
        `scripts/extract_human.py` exist, and `perception/hand.py` has grown the PnP
        pinch point, the metric scale and the grasp hysteresis on top of the raw
        landmarks Phase 1 gave it.
19. [ ] Run it on 3 clips first: one success, one F1 miss, one F2 drop.
        **Expect:** three debug mp4s in `results/`, each with a plot panel beside the video.
20. [ ] Open each debug video and check all eight:
        - [ ] green pinch point sits between your thumb and index tips
        - [ ] red reprojected point lands on the green one, under 15 px
        - [ ] blue box stays on the block for the whole clip
        - [ ] while the block is in your hand, it moves with your hand
        - [ ] the OPEN / CLOSED label flips at the right moments
        - [ ] side plot: hand z is 1–3 cm at rest and 8–15 cm mid-carry; block z is half
              the block size on the table and rises with your hand
        - [ ] **F1 clip: block z stays flat while your hand lifts.** This is the key check
        - [ ] F2 clip: block z drops after you open your fingers

        **Expect:** the F1 check in particular. If the block appears to lift on a miss
        clip, the tracker is following your hand instead of the block, and the entire
        failure-awareness claim of the project is dead until it is fixed.
21. [ ] Run the extraction on all 45 clips.
        **Expect:** `data/human/transitions.npz` with roughly 9,000 transitions, and one
        raw npz per clip. A couple of minutes of runtime.
22. [ ] Open `results/extraction_report.csv`.
        **Expect:** **at least 35 clips pass, including at least 10 failure clips**, and
        the vision-vs-`dots.yaml` block start error under 2 cm. A large start error means
        the board pose or the dot measurements are wrong, not the tracker.
        ✂️ Hand z too noisy: switch the hand to the fixed lift-profile fallback and keep
        the block measured. Block lost while held: set block = pinch point − (0, 0, b/2)
        while closed and lifted, and write that limitation into the README.

## Retargeting and robot data

23. [ ] Ask Cursor: **Phase 3** (tests first). Run `uv run pytest tests/test_retarget.py`.
        **Expect:** green, including the worked example from EXPLAINER §8
        (α = −0.04 − 1.08i, β = 0.45 + 0.21i) reproduced to 1e-3.
24. [ ] Run `uv run python scripts/plot_retarget.py`.
        **Expect:** a top view where the robot path has the same shape as your hand path,
        but starts exactly at the block and ends exactly at the target. If it starts
        somewhere else, the anchors are wrong.
25. [ ] Ask Cursor: **Phase 4c** (replay + E1). Run `uv run python scripts/replay.py`.
        **Expect:** `results/e1.csv` plus a printed summary in which **two-anchor beats
        naive**. If naive wins, the retargeting is mapping to the wrong frame. Keep the
        three side-by-side videos — one is the hero GIF.
26. [ ] Ask Cursor: **Phase 5**. Run `uv run python scripts/generate_robot.py`.
        **Expect:** 300 training and 100 held-out episodes in a few minutes, with a
        printed outcome mix of **50–70% success**. All-success means the failure injection
        is not firing; all-failure means the tracker or the scene is broken.

## World model

27. [ ] Ask Cursor: **Phase 6**.
        **Expect:** `wm/model.py`, `wm/train.py`, `scripts/train_wm.py`,
        `scripts/exp_e2.py`. One model trains in about a minute on CPU.
28. [ ] Run `uv run python scripts/exp_e2.py`.
        **Expect:** `results/e2.csv` and `results/e2_curve.png`, covering 3 variants × 7
        values of N × 3 seeds.
29. [ ] Check all three before trusting any curve:
        - [ ] the world model beats "the block never moves" on held-out data
        - [ ] on F1-style states (fingers closed next to the block) the predicted lift is
              near zero
        - [ ] ensemble spread is larger on robot-only states for the human-only model

        **Expect:** all three. If the model cannot beat a constant zero prediction, nothing
        downstream is meaningful and no amount of planning will rescue it.
30. [ ] Look at `results/e2_curve.png` **before** tuning anything.
        **Expect:** error falling as N grows, and the human-pretrained curve sitting to
        the left of sim-only. This is the headline result of the submission.
        ✂️ Pretraining does not help: that is a valid result. Use the heatmap to explain
        why, try co-training once, then report it honestly and move on.
31. [ ] Run `uv run python scripts/fig_gap_heatmap.py` and
        `uv run python scripts/fig_human_overlay.py`.
        **Expect:** three heatmap panels where the true-sim panel has a visibly narrower
        high-probability region than the human-only panel — that is the embodiment gap
        made visible — and an overlay video where the predicted block tracks the measured
        one for the first second or two before drifting.
32. [ ] Commit and push. Write a 10-line README draft.
        **Expect:** a repo that someone else could clone and run. **MVP reached.**

## Planning

33. [ ] Ask Cursor: **Phase 7**. Run `uv run pytest tests/test_cem.py`.
        **Expect:** green — CEM finds the optimum of a 2-D quadratic.
34. [ ] Run `uv run python scripts/watch_mpc.py` and **watch all 5 episodes**.
        **Expect:** the translucent "imagination" spheres trace a sensible path from the
        block to the target. If they fly off the table, the cost or the rollout is wrong —
        find out now, not after a four-hour evaluation.
35. [ ] Run `uv run python scripts/eval_control.py` for **E3**, the main table.
        **Expect:** `results/e3.csv`, mean ± std per method, and a printed planning time
        of roughly 50–100 ms per re-plan.
        ✂️ MPC worse than open loop: lower `mpc.sigma_knot`, raise `mpc.beta`, check the
        cost uses the right target height. Still worse after an hour: report it, analyse
        one failure in detail, move on.
36. [ ] Run **E4**, **E5**, **E6**, **E7**.
        **Expect:** `results/e4.csv`, `e5.csv`, `e6_control.csv`, `e7.csv`, and in E6 a
        lower attach AUROC for the model trained without the failure clips. That number is
        the direct evidence for the failure-clip idea.

## Ship

37. [ ] Ask Cursor: **Phase 8** (figures, reel, README).
        **Expect:** every figure regenerated from `results/*.csv` only.
38. [ ] Write **"What worked / what didn't"** yourself. Do not let the agent write this.
        **Expect:** at least one thing that genuinely did not work, with the number.
39. [ ] Check every number in the README traces to a file in `results/`.
        **Expect:** zero numbers you cannot point at a CSV for.
40. [ ] Fresh-clone test in a new folder: `git clone`, `uv sync --extra dev`, then run
        every command in the README.
        **Expect:** it works on a machine with none of your local state. This is what the
        reviewer will do.
41. [ ] Add LICENSE (MIT). Confirm `uv.lock` is committed, no secrets, no dead files.
42. [ ] Upload 3 sample clips, the processed human transitions and the trained world model
        to a GitHub release. Link them from the README.
        **Expect:** the links work from a logged-out browser.
43. [ ] Make the repo **public**. Open it in a private window to confirm.
44. [ ] Submit the application form with the repo URL, your name and your CV.

---

## Submission requirements, straight from the brief

- [ ] Public GitHub repository
- [ ] README with **instructions to run the system**
- [ ] README with **example outputs**
- [ ] README with **a note on design choices, what worked and what didn't**
- [ ] The data you personally collected plays a central role — say exactly where
- [ ] No AI slop: no "seamless", "revolutionary", "cutting-edge", "leverage"; no emoji;
      no sentence that could describe any other project; no claim without a number beside it

---

## If you run short of time, drop in this order

1. The stretch policy distillation. It was never required.
2. The 60-second reel. Keep the hero GIF and the static figures.
3. E4, E5, E6, E7. Keep E1, E2, E3.
4. The bonus pad and box clips. Run everything on the plate goal only.
