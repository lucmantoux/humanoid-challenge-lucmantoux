# palm-prior — Your Action Plan (Mac-only)

**Deadline:** 23:59 BST Fri 9 Oct = 00:59 Paris Sat 10 Oct. **Your deadline: Fri 9 Oct, 20:00 Paris.**
**Everything runs on your Mac (CPU).** No Colab, no GPU.
**MVP** (a valid submission by itself) = human state extraction + sim + E1 + E2 curve. Ready by **Tue night**.
✂️ = cut line: hit it → take the fallback, move on.

Read `docs/EXPLAINER.md` sections 0–2 before starting. They explain *why* every step exists.

---

## Day 0 — Fri 2 Oct (tonight, ~1 h)

### Mac setup (15 min)
- [ ] Check your chip: Apple menu → About This Mac (M1/M2/M3/M4 = Apple Silicon, assumed throughout).
- [ ] Install `uv` (fast Python manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] `brew install ffmpeg` (for videos). Install Homebrew first if you don't have it.
- [ ] Create the public GitHub repo `palm-prior`, clone it, open it in Cursor.
- [ ] Put `EXPLAINER.md` → `docs/`, `FILMING_GUIDE.md` → `docs/`, this file → `docs/TODO.md`.
- [ ] Cursor: add the Master Rule, run **Phase 0**, then the MuJoCo smoke test it creates. **You should see a Panda image saved to `results/`.**

### Shopping / printing
- [ ] Find a **blue or green 3–5 cm block**, a small plain plate, tape, a ruler; optionally a coaster and an ~8 cm box.
- [ ] Cursor **Phase 1** first part: `make_board.py` → print the ArUco board + checkerboard → **measure them**.

**Theory to know for tonight (EXPLAINER §2):** the whole project uses one 8-number state describing *where the hand is relative to the block, and the block relative to the target*. Because it only uses differences, your table and the robot's floor never need to be aligned.

---

## Day 1 — Sat 3 Oct: film + sim scene

### Morning: film (≈ 1.5 h) — follow `FILMING_GUIDE.md` exactly
- [ ] Finish Cursor **Phase 1** (calibration, ArUco debug, hand + block preview).
- [ ] Set up the scene, measure the dots → `dots.yaml`.
- [ ] Calibration video → `scripts/calibrate.py` → RMS < 0.5 px.
- [ ] `empty.mp4` + **test clip** → check board / hand / block mask.
- [ ] Record 30 successes + 15 failures. Fill in the log.

**Why the failure clips (EXPLAINER §5, "Why deliberate failure clips matter"):** a planner hunts for the model's mistakes. Without failures in the data, the model never learns that a missed grasp leaves the block behind, and the planner will happily "grasp" next to the block.

### Afternoon: sim (≈ 3 h)
- [ ] Cursor **Phase 4a**: scene (Panda + block + plate + pad + box) + DLS IK tracker.
- [ ] Watch it with `mjpython scripts/view_scene.py` (on Mac the viewer needs `mjpython`).
- [ ] Cursor **Phase 4b**: scripted pick-place ≥ 9/10 on each goal. Save `results/scripted_check.mp4`.
- ✂️ Grasp slips: raise the friction to 2.0, set `impratio` to 20, slow the lift (smaller max step). Still failing after 1 h → make the block 3 cm and lighter (30 g).

**Theory (EXPLAINER Module 7):** the IK formula $\Delta q = J^\top(JJ^\top + \lambda^2 I)^{-1}e$ turns "move the gripper 2 cm" into joint motions; λ stops wild motions near awkward poses.

---

## Day 2 — Sun 4 Oct: phone → human state

- [ ] Cursor **Phase 2**: hand PnP + block tracking + transitions.
- [ ] Run on 3 clips first. In each debug video:
  - [ ] green pinch point between thumb and index
  - [ ] red reprojected pinch on green (< 15 px)
  - [ ] blue box on the block the whole time; in-hand block follows the hand
  - [ ] OPEN/CLOSED correct
  - [ ] plot: hand z ≈ 1–3 cm at rest, 8–15 cm mid-carry; block z = b/2 on the table, rises with the hand
  - [ ] F1 clip: block z stays flat while the hand lifts ← **the key check**
  - [ ] F2 clip: block z drops after release
- [ ] Run all → `results/extraction_report.csv` → **≥ 35 clips must pass** (incl. ≥ 10 failures).
- [ ] Cross-check: the vision-measured block start vs `dots.yaml` (the report shows the error; < 2 cm is fine).
- ✂️ Hand z noisy: use the fixed lift profile fallback for the hand (keep the block measured). Block lost in hand: while closed and lifted, set block = pinch point − (0, 0, b/2) and **say so in the README** (it weakens the WM claim a bit).

**Theory (EXPLAINER Modules 3–5):** PnP gets the hand's 3D pose from its 2D landmarks; the 2 s rest on the table fixes the scale; the block's 3D position comes from intersecting its pixel ray with a known height.

---

## Day 3 — Mon 5 Oct: retarget + robot data

- [ ] Cursor **Phase 3** (tests first, all green).
- [ ] Cursor **Phase 4c**: replay → **E1** + 3 side-by-side videos (phone | sim). Hero GIF candidate.
- [ ] Cursor **Phase 5**: generate 300 robot episodes (+ 100 held-out, seeds 900–999). Check the outcome mix printed: aim for ~50–70% success, the rest failures.

**Theory (Module 6):** two points fix a 2D rotation + scale + shift exactly, so forcing "my grasp → robot grasp" and "my release → robot release" defines the whole mapping while keeping my motion's shape.

---

## Day 4 — Tue 6 Oct: world model + E2 (MVP done tonight)

- [ ] Cursor **Phase 6**: ensemble WM, the 3 training variants, the E2 curve, the embodiment heatmap, the human-video overlay.
- [ ] **Sanity checks before trusting any curve:**
  - [ ] the WM beats "block doesn't move" (Δ = 0) on held-out data
  - [ ] the F1 states (closed next to the block): predicted lift ≈ 0
  - [ ] ensemble std is higher on robot-only states for the human-only model
- [ ] Commit, push, write a 10-line README draft. **MVP done.**
- ✂️ Pretraining doesn't help on the curve: that's a valid finding. Check the heatmap to explain why (usually grasp tolerance), try co-training, then report honestly.

**Theory (Module 9):** predict the *change* of the block, train 5 models so their disagreement shows ignorance, train on 5-step self-rollouts to limit drift.

---

## Day 5 — Wed 7 Oct: planning + control experiments

- [ ] Cursor **Phase 7**: CEM-MPC.
- [ ] Watch 5 episodes as video *before* running the big evals (a bug there wastes hours).
- [ ] Run **E3** (main table), **E4** (prior vs isotropic), **E5** (push + noise), **E6** (failure ablation), **E7** (3 goals). Everything → `results/*.csv`.
- ✂️ MPC worse than open loop: lower the CEM σ, raise β (more pessimism), check that the cost uses the right target height. Still worse by 18:00 → report it, analyse one failure in detail, move on.

**Theory (Module 10):** CEM = sample, keep the best, refit, repeat. MPC = plan to the end, execute 0.5 s, re-plan. The β·std term makes the robot avoid plans the model is unsure about.

---

## Day 6 — Thu 8 Oct: visuals + README

- [ ] Cursor **Phase 8**: figures from CSVs, "imagination" video (predicted block path drawn as ghost spheres in the MuJoCo render), human-video prediction overlay, 60 s reel.
- [ ] README draft from the skeleton below.
- [ ] Stretch only if done by 15:00: distill MPC into a policy (speed comparison).

## Day 7 — Fri 9 Oct: ship by 20:00 Paris

- [ ] Write **"What worked / what didn't"** yourself.
- [ ] Fresh-clone test on your Mac in a new folder: `uv sync` → README commands on the sample data.
- [ ] Sample clips (3) + processed human transitions + trained WM in a GitHub release or on Drive; link them.
- [ ] `.gitignore`, LICENSE (MIT), `uv.lock`, no secrets, no dead files.
- [ ] Public → check in incognito → submit the form + CV.

---

## README skeleton
1. **Title + one-line claim + hero GIF** (phone clip | robot doing the same)
2. **Key result**: E2 curve + E3 table (measured only)
3. **Idea in 4 bullets**: physics from my hands; failures on purpose; one shared state; MPC seeded by my path
4. **My data**: setup photo, 30 + 15 clips, protocol, QC pass rate
5. **How it works**: pipeline diagram + one paragraph per stage
6. **Design choices + options rejected** (EXPLAINER §0 table)
7. **What worked / what didn't**
8. **Run it** (Mac, CPU): one command per stage, with timings
9. **Limitations**: sim object positions are given to the robot; top-down grasps only; one object type; human block position is approximate while in hand
10. **Next steps**

**No slop:** no "seamless/revolutionary/cutting-edge/leverage", no emoji, no claim without a number, delete any sentence that could describe any project.

---

## Risks
| Risk | Fallback |
|---|---|
| Grasp physics flaky in MuJoCo | friction/impratio up, lighter smaller block (Day 1 cut) |
| Block lost while in hand | geometric fallback (Day 2 cut), stated in README |
| Pretraining shows no gain | report + heatmap explanation (Day 4 cut) |
| MPC slow | fewer samples (128), fewer ensemble members (3) |
| MediaPipe won't install | use Python 3.11 exactly (wheels exist for it) |
| Running late | MVP is a full submission; skip E5–E7 first |
