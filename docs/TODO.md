# Task list — do these in order

Deadline: Friday 9 October 2026, 23:59 BST.

"Ask Cursor: Phase N" means open Agent mode and paste the Phase N block from
`docs/CURSOR_PROMPT.md`. Do one phase at a time. Do not start the next task until the
current one's checks pass on your machine.

✂️ marks a fallback: if you hit it, take the fallback and keep moving.

**Stop point.** Tasks 1–26 are a complete, submittable project on their own: human state
extraction, simulator, E1, E2. Everything after task 26 is upside. Protect tasks 1–26.

---

## Done

- [x] uv installed, `uv sync --extra dev` works, `uv run pytest` green
- [x] Phase 0: package scaffold, `configs/default.yaml`, the shared 8-D state, MuJoCo
      smoke test (`results/smoke.png` shows the Panda)
- [x] `results/print/board_a4.pdf` generated
- [x] `data/raw/recording_log.csv` and `data/raw/dots.yaml` filled in

---

## Perception scripts

1. [ ] Ask Cursor: **Phase 1**
2. [ ] Run `uv run pytest`. Green.
3. [ ] Confirm `scripts/calibrate.py`, `scripts/debug_aruco.py` and
       `scripts/preview_hand.py` exist.

## Film the dataset — follow `docs/FILMING_GUIDE.md` literally

4. [ ] Guide Step 1 — print the board, measure 60 mm and 25 mm, put your block size and
       plate radius into `configs/default.yaml`.
5. [ ] Guide Step 2 — tape the dots and the board.
6. [ ] Guide Step 3 — position the phone.
7. [ ] Guide Step 4 — phone settings.
8. [ ] Guide Step 5 — record `calib.mp4`.
9. [ ] Guide Step 6 — record `empty.mp4`.
10. [ ] Guide Step 7 — the test clip, and all five of its checks.
        ✂️ Still failing after 45 minutes: switch to the Blackmagic Camera app
        (Guide Step 10) and re-record `calib.mp4` first.
11. [ ] Guide Step 8 — record clips 1–45.
12. [ ] Guide Step 9 — transfer, rename, back up `data/raw/` outside the repo.

## Simulator

13. [ ] Ask Cursor: **Phase 4a** (scene + IK).
14. [ ] Look at the scene:
        macOS `uv run mjpython scripts/view_scene.py`,
        Windows and Linux `uv run python scripts/view_scene.py`.
        Check the `tcp` marker sphere sits between the two fingertips, and the block,
        plate, pad and box rest on the floor without intersecting anything.
15. [ ] Run `uv run pytest tests/test_ik.py`. IK must reach 5 random targets to under
        5 mm within 30 control steps.
16. [ ] Run `uv run python scripts/render_layouts.py` and look at the 6-layout grid.
17. [ ] Ask Cursor: **Phase 4b** (scripted pick-and-place).
18. [ ] Run `uv run python scripts/scripted_check.py`. At least **9 out of 10 succeed on
        each goal**, and `results/scripted_check.mp4` looks sane.
        ✂️ Grasps slipping: `objects.block_friction=2.0`, `sim.impratio=20`,
        `sim.max_step=0.015`. Still failing after an hour: `objects.block_size=0.03`,
        `objects.block_mass=0.03`.

## Phone video to state

19. [ ] Ask Cursor: **Phase 2**.
20. [ ] Run it on 3 clips first: one success, one F1 miss, one F2 drop.
21. [ ] Open each debug video and check all eight:
        - [ ] green pinch point sits between your thumb and index tips
        - [ ] red reprojected point lands on the green one, under 15 px
        - [ ] blue box stays on the block for the whole clip
        - [ ] while the block is in your hand, it moves with your hand
        - [ ] the OPEN / CLOSED label flips at the right moments
        - [ ] side plot: hand z is 1–3 cm at rest and 8–15 cm mid-carry; block z is half
              the block size on the table and rises with your hand
        - [ ] **F1 clip: block z stays flat while your hand lifts.** This is the key check
        - [ ] F2 clip: block z drops after you open your fingers
22. [ ] Run the extraction on all 45 clips.
23. [ ] Open `results/extraction_report.csv`. **At least 35 clips pass, including at
        least 10 failure clips.** The vision-vs-`dots.yaml` block start error is under 2 cm.
        ✂️ Hand z too noisy: switch the hand to the fixed lift-profile fallback and keep
        the block measured. Block lost while held: set block = pinch point − (0, 0, b/2)
        while closed and lifted, and write that limitation into the README.

## Retargeting and robot data

24. [ ] Ask Cursor: **Phase 3** (tests first). `uv run pytest tests/test_retarget.py` green.
25. [ ] Run `uv run python scripts/plot_retarget.py`. The robot path has the same shape as
        your hand path, starts at the block and ends at the target.
26. [ ] Ask Cursor: **Phase 4c** (replay + E1). Run `uv run python scripts/replay.py`.
        Check `results/e1.csv`: two-anchor beats naive. Keep the three side-by-side
        videos — one is the hero GIF.
27. [ ] Ask Cursor: **Phase 5** (robot dataset). Run
        `uv run python scripts/generate_robot.py`. Printed outcome mix **50–70% success**,
        300 training and 100 held-out episodes, a few minutes.

## World model

28. [ ] Ask Cursor: **Phase 6**.
29. [ ] Run `uv run python scripts/exp_e2.py`.
30. [ ] Check all three before trusting any curve:
        - [ ] the world model beats "the block never moves" on held-out data
        - [ ] on F1-style states (fingers closed next to the block) the predicted lift is
              near zero
        - [ ] ensemble spread is larger on robot-only states for the human-only model
31. [ ] Look at `results/e2_curve.png` **before** tuning anything.
        ✂️ Pretraining does not help: that is a valid result. Use the heatmap to explain
        why, try co-training once, then report it honestly and move on.
32. [ ] Run `uv run python scripts/fig_gap_heatmap.py` and
        `uv run python scripts/fig_human_overlay.py`.
33. [ ] Commit and push. Write a 10-line README draft.

## Planning

34. [ ] Ask Cursor: **Phase 7**. `uv run pytest tests/test_cem.py` green.
35. [ ] Run `uv run python scripts/watch_mpc.py` and **watch all 5 episodes** before
        launching anything long. A bug here costs hours.
36. [ ] Run `uv run python scripts/eval_control.py` for **E3**, the main table.
        ✂️ MPC worse than open loop: lower `mpc.sigma_knot`, raise `mpc.beta`, check the
        cost uses the right target height. Still worse after an hour: report it, analyse
        one failure in detail, move on.
37. [ ] Run **E4** (your path vs a straight line), **E5** (push and noise), **E6**
        (failure-clip ablation), **E7** (three goals). Every table lands in `results/*.csv`.

## Ship

38. [ ] Ask Cursor: **Phase 8** (figures, reel, README).
39. [ ] Write **"What worked / what didn't"** yourself. Do not let the agent write this.
40. [ ] Check every number in the README traces to a file in `results/`.
41. [ ] Fresh-clone test in a new folder: `git clone`, `uv sync --extra dev`, then run
        every command in the README.
42. [ ] Add LICENSE (MIT). Confirm `uv.lock` is committed, no secrets, no dead files.
43. [ ] Upload 3 sample clips, the processed human transitions and the trained world model
        to a GitHub release. Link them from the README.
44. [ ] Make the repo **public**. Open it in a private window to confirm.
45. [ ] Submit the application form with the repo URL, your name and your CV.

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
