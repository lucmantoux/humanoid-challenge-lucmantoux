# palm-prior — Filming Guide (45 clips: 30 success + 15 deliberate failures)

Follow this literally. Total time: about **1 h 30 min** (20 min setup, 10 min calibration, 55 min recording, 5 min transfer).

---

## 1. What you need

| Item | Spec | Notes |
|---|---|---|
| Phone + tripod | tripod or books + tape | must not move during a clip |
| Camera app | **Blackmagic Camera** (free, iOS + Android) or **Open Camera** (Android) | The stock iPhone app can't turn stabilization off |
| ArUco board | printed by `scripts/make_board.py`, 2×2 markers of 60 mm, with "ORIGIN" printed on one corner | measure with a ruler |
| Checkerboard | 9×6 inner corners, 25 mm squares, taped flat on cardboard | must be rigid |
| **Block** | 3–5 cm cube, **saturated blue or green** | Not red, orange or beige: those confuse the colour mask with skin. The only object of that colour in view. |
| Plate | small, ~18 cm, plain | measure its radius (the sim copies it) |
| Tape / stickers | 9 dots + 1 rest mark | |
| Ruler or tape measure | | to measure dot positions |
| Optional | coaster (pad), ~8 cm box | bonus goal clips |

**Measure and write down:** block size b (cm), plate radius (cm). These go in `configs/default.yaml`.

---

## 2. Scene layout

The camera faces **you** across the table, like the sim camera faces the robot. Your fingertips point toward the camera, so they stay visible.

```
                 YOU
        ┌───────────────────────────────────────────┐
        │                                   [REST]  │  ← hand rest mark
        │      (1)          (2)          (3)        │
        │                                           │
        │      (4)          (5)          (6)        │   dots 15 cm apart
        │                                           │
        │      (7)          (8)          (9)        │
        │ ┌─────┐                                   │
        │ │ArUco│  ← near-left corner, flat,        │
        │ └─────┘    arm never crosses it           │
        └───────────────────────────────────────────┘
                          ▲  PHONE (landscape)
```

1. Clear a 50×50 cm matte area.
2. Tape 9 dots in a 3×3 grid, **15 cm apart**, numbered as shown from the phone's point of view.
3. Tape the ArUco board flat at the near-left corner, ~8 cm outside dot 7.
4. Tape a rest mark ~10 cm outside dot 3.
5. **Measure every dot** relative to the board's ORIGIN corner: x = along the board's bottom edge (to the right as printed), y = along its left edge, in cm. Write the 9 values into `data/raw/dots.yaml`:
   ```yaml
   dots:   # cm, table frame (origin = board ORIGIN corner)
     1: [x, y]
     2: [x, y]
     # ...
     9: [x, y]
   ```
   Accuracy ±0.5 cm is enough. The pipeline cross-checks these against the vision measurements.
6. **Phone:** opposite you, landscape, lens ~55 cm above the table, ~40 cm horizontally from the near row, aimed at dot 5 (≈ 45° down).
7. **Framing:** the grid, board, rest mark and ~10 cm margin all visible.
8. **Light:** side daylight or two side lamps. No window behind you. No hard shadow on the board or the block.

---

## 3. Phone settings (set once, never change)

| Setting | Value |
|---|---|
| Resolution / fps | 1920×1080, 30 fps |
| Lens | main 1× only; auto lens/macro switching off |
| Stabilization | **OFF** |
| Focus | manual, locked on dot 5 |
| Exposure | locked; shutter 1/250 s or faster if the light allows |
| White balance | locked (fixed preset) |
| HDR / Dolby Vision | off |

Screenshot the settings screen.

---

## 4. Calibration video (10 min, record FIRST)

Same app and settings. Phone on the tripod, you move the checkerboard, about 1 s per position:
- centre, flat-on
- the 4 corners of the frame
- tilted ~30° left / right / up / down
- near (fills ~60% of the frame) and far (~20%)
- lying flat on the grid

40 s total, slow, whole board visible, no fingers on corners. Name it `calib.mp4`.

Also record **`empty.mp4`**: 5 s of the scene with the block on dot 5, plate on dot 9, no hand. It's used to fit the block colour and the camera pose.

---

## 5. Test clip (10 min) — don't skip

1. Record clip #1 from the plan.
2. On the Mac: `python scripts/debug_aruco.py` and `python scripts/preview_hand.py` on it.
3. Check:
   - [ ] board axes stable on every frame
   - [ ] hand landmarks on your hand, **thumb and index tips visible at the grasp**
   - [ ] the block mask (shown by `preview_hand.py`) is a clean blob on the block only, including while it's in your hand
4. Everything good → record the rest. Otherwise → section 10.

---

## 6. One SUCCESS clip (≈ 10 s)

Before each clip: block centred on its dot, plate centred on its dot (see the plan).

| Time | Do | Why |
|---|---|---|
| 0 s | Tap record. Right hand **flat on the rest mark**. | |
| 0–2 s | **Completely still.** | metric depth scale (mandatory) |
| 2–4 s | Reach; **pinch the block from the top with thumb + index**, other fingers loosely curled, fingertips facing the camera side | clean grasp signal |
| 4–7 s | Lift (arc per the plan), carry, set it down **centred on the plate** | |
| 7 s | **Open fingers wide**, lift your hand straight up ~5 cm | clean release signal |
| 7–8 s | Hand back to the rest mark, flat | |
| 8–10 s | **Still, 2 s.** Stop. | second scale reference |

**Rules:** right hand, sleeves up, no watch, natural speed, exactly one grasp and one release, the arm never covers the board, the other hand out of frame. Fumbled → delete and redo the same number.

---

## 7. One FAILURE clip (≈ 10 s) — same structure, the middle changes

These teach the world model what failure looks like. Do them **on purpose and cleanly**: the failure should be obvious and happen only once.

| Type | What you do in the middle | What the model learns |
|---|---|---|
| **F1 Miss** | Close your pinch **2–3 cm beside** the block (left, right or behind, as in the plan), lift your hand 10 cm, move toward the plate, open. The block **stays on its dot**. | closing near ≠ holding |
| **F2 Early drop** | Grasp properly, lift ~10 cm, carry halfway, **open your fingers in the air**. Let the block fall to the table. Don't catch it. | release → fall |
| **F3 Wrong place** | Grasp properly, carry, set it down **on the table 8–10 cm short of the plate**, release. | where the block ends up = where it's released |
| **F4 Push** | Hand open, move it low and **push the block sideways ~10 cm** with your fingers along the table, no grasp. Then rest. | contact without a grasp moves the block differently |

Rest 2 s at the start and end, as always.

---

## 8. Recording plan (45 clips)

**Arc:** low = lift ~5 cm, high = ~15 cm. **Side** (failures): where you close/push relative to the block, from the camera's view.

### Successes (#1–30)
| # | Block | Plate | Arc | | # | Block | Plate | Arc |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3 | low | | 16 | 7 | 3 | high |
| 2 | 3 | 1 | high | | 17 | 5 | 1 | low |
| 3 | 4 | 6 | low | | 18 | 5 | 3 | high |
| 4 | 6 | 4 | high | | 19 | 5 | 7 | low |
| 5 | 7 | 9 | low | | 20 | 5 | 9 | high |
| 6 | 9 | 7 | high | | 21 | 2 | 4 | low |
| 7 | 1 | 7 | low | | 22 | 4 | 8 | high |
| 8 | 7 | 1 | high | | 23 | 8 | 6 | low |
| 9 | 2 | 8 | low | | 24 | 6 | 2 | high |
| 10 | 8 | 2 | high | | 25 | 1 | 5 | low |
| 11 | 3 | 9 | low | | 26 | 9 | 5 | high |
| 12 | 9 | 3 | high | | 27 | 2 | 6 | low |
| 13 | 1 | 9 | low | | 28 | 4 | 2 | high |
| 14 | 9 | 1 | high | | 29 | 8 | 4 | low |
| 15 | 3 | 7 | low | | 30 | 6 | 8 | high |

### Failures (#31–45)
| # | Type | Block | Plate | Detail |
|---|---|---|---|---|
| 31 | F1 Miss | 1 | 3 | close 3 cm to the left |
| 32 | F1 Miss | 5 | 9 | close 3 cm to the right |
| 33 | F1 Miss | 7 | 2 | close 3 cm behind (toward you) |
| 34 | F1 Miss | 6 | 4 | close 2 cm to the left |
| 35 | F2 Drop | 1 | 9 | drop halfway, from ~10 cm |
| 36 | F2 Drop | 3 | 7 | drop halfway, from ~15 cm |
| 37 | F2 Drop | 8 | 2 | drop 1/3 of the way |
| 38 | F2 Drop | 4 | 6 | drop 2/3 of the way |
| 39 | F3 Short | 1 | 3 | place 8 cm short |
| 40 | F3 Short | 9 | 5 | place 10 cm short |
| 41 | F3 Short | 7 | 1 | place 8 cm short |
| 42 | F3 Short | 2 | 8 | place 10 cm short |
| 43 | F4 Push | 5 | 1 | push toward dot 6 |
| 44 | F4 Push | 4 | 9 | push toward dot 5 |
| 45 | F4 Push | 8 | 3 | push toward dot 9 |

**Every 10 clips:** play back the last one. Board in frame? Fingertips visible? Block mask still fine (quick look)?

### Bonus (optional)
- #46–50: block → **pad** (put the coaster on the "plate" dot; reuse pairs 1, 5, 9, 13, 17)
- #51–55: block **stacked on the box** (reuse pairs 3, 7, 11, 15, 19)

---

## 9. After recording

1. Transfer **originals** (USB, AirDrop, or Drive upload of the original file). Never WhatsApp/Messenger/email: they compress the video.
2. Rename: `calib.mp4`, `empty.mp4`, `demo_001.mp4` … `demo_045.mp4`.
3. Fill in `data/raw/recording_log.csv`:
   ```
   clip,type,block_dot,plate_dot,goal,arc,detail,redo
   1,success,1,3,plate,low,,0
   31,F1,1,3,plate,,close 3cm left,0
   ```
4. Check `data/raw/dots.yaml` is filled, plus the block size and plate radius in the config.
5. Photo of the setup from the side, for the README.
6. Back up the raw folder (Drive or an external disk). Never commit it to git; put 3 sample clips in the repo release or link them.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Board not detected | more light, flatten it, keep it fully in frame |
| Fingertips missing at the grasp | rotate your wrist so thumb + index face the camera; pinch from the camera side |
| Block mask also catches skin or background | use a more saturated blue/green block; remove same-coloured objects |
| Block mask disappears in hand | pinch near the top edge so more of the block shows |
| Landmarks jitter | faster shutter, more light |
| Image wobbles / crops | stabilization still on |
| Exposure pumps when the hand enters | exposure not locked |

---

## 11. Final checklist
- [ ] `calib.mp4`, `empty.mp4`
- [ ] test clip passed (board, hand, block mask)
- [ ] 30 success + 15 failure clips
- [ ] `recording_log.csv`, `dots.yaml`, block size, plate radius
- [ ] originals transferred and backed up
- [ ] setup photo
