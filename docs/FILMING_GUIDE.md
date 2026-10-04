# Filming Guide — do exactly this, in this order

Recording 45 clips with an iPhone and the stock Camera app. No decisions to make: every
number and every tap is written out. Total time about 2 h.

Why each step exists is in `EXPLAINER.md`. Do not read it now.

Tick the boxes as you go.

---

## Step 0 — Put these on the table before you start

- [ ] iPhone
- [ ] Tripod, or a stack of books plus tape (the phone must not move for 2 h)
- [ ] The printed pack: page 1 ArUco board, page 2 checkerboard (Step 1)
- [ ] A rigid piece of cardboard, at least A4, to tape the checkerboard onto
- [ ] **A cube 3–5 cm on a side, saturated blue or green.** Not red, orange, beige or
      white: those get confused with skin. It must be the only object of that colour in view.
- [ ] A plain plate, about 18 cm across
- [ ] Masking tape or 10 small round stickers
- [ ] A tape measure or a 50 cm ruler
- [ ] A marker pen
- [ ] Optional: a coaster, and a box about 8 cm on a side

---

## Step 1 — Print and measure the board

1. [ ] In the repo folder run:
       ```
       uv run python scripts/make_board.py
       ```
2. [ ] Open `results/print/board_a4.pdf`.
3. [ ] Press Cmd-P. In the print dialog set **Paper Size: A4**, **Scale: 100%**.
       If you see "Scale to Fit", switch it off. Print both pages.
4. [ ] Take a ruler to page 1. Measure one black marker square edge to edge.
       **It must read 60 mm.** Measure the printed scale bar: **100 mm**.
       - Not 60 mm → reprint with scaling off. Do not continue, and do not edit the config.
5. [ ] Measure one square on page 2: **25 mm**.
6. [ ] Tape page 2 flat onto the cardboard. No bubbles, no curl. This is the
       **checkerboard**. Set it aside.
7. [ ] Measure your cube's edge with the ruler, in centimetres, to the nearest millimetre.
       Write it here: block = ______ cm
8. [ ] Measure your plate's radius (edge to centre). Write it here: plate radius = ______ cm
9. [ ] Open `configs/default.yaml` and set:
       - `objects.block_size:` your block in **metres** (4.0 cm → `0.04`)
       - `objects.plate_radius:` your plate radius in **metres** (9.0 cm → `0.09`)

---

## Step 2 — Build the scene

You are taping a 3×3 grid of dots, the ArUco board, and a hand rest mark onto a table.
All positions below are measured from one corner, so they come out right by construction
and you never have to measure the dots individually.

1. [ ] Clear a flat, matte area about **75 cm wide by 55 cm deep**. No shiny tablecloth,
       no patterned surface. Plain and light-coloured is best.
2. [ ] Decide which side you will **sit** on. The phone goes on the **opposite** side,
       facing you. From here on, "right" means **your** right as you sit at the table.
3. [ ] Tape page 1 (the ArUco board) flat on the table, to your **far left**,
       with the printed text the right way up **as seen from the phone's side**, i.e.
       upside-down from where you sit. Press all four edges down. It must not lift.
4. [ ] Find the **ORIGIN corner** of the board: it is labelled, with an x arrow pointing
       right and a y arrow pointing away from the phone. Everything below is measured
       from that corner.
5. [ ] Lay the tape measure along the table with 0 at the ORIGIN corner, running to your
       right. Place the 9 dots at these positions. **x** = centimetres to the right of the
       ORIGIN corner, **y** = centimetres away from the phone, i.e. toward you.

       | Dot | x (cm) | y (cm) |     | Dot | x (cm) | y (cm) |
       |---|---|---|---|---|---|---|
       | 7 | 22 | 10 |  | 4 | 22 | 25 |
       | 8 | 37 | 10 |  | 5 | 37 | 25 |
       | 9 | 52 | 10 |  | 6 | 52 | 25 |
       | 1 | 22 | 40 |  | 2 | 37 | 40 |
       | 3 | 52 | 40 |  |   |    |    |

       Accuracy of ±0.5 cm is fine. Write the dot number next to each one with the marker.
6. [ ] Check the pattern looks like this from where you sit (dots 1–3 nearest you,
       7–9 nearest the phone, board at the far left):

       ```
                             YOU
         ┌──────────────────────────────────────────────┐
         │   (1)        (2)        (3)        [REST]    │   y = 40, rest at x = 64
         │                                              │
         │   (4)        (5)        (6)                  │   y = 25
         │                                              │
         │   (7)        (8)        (9)                  │   y = 10
         │ ┌───────┐                                    │
         │ │ ArUco │ ORIGIN at its lower-left corner    │
         │ └───────┘                                    │
         └──────────────────────────────────────────────┘
                            ▲ PHONE (landscape)
         x = 0 at ORIGIN, increasing to the right  ─────────►
       ```
7. [ ] Tape the **rest mark** — a strip of tape about 8 cm long — at x = 64 cm, y = 40 cm.
       That is 12 cm to the right of dot 3. This is where your hand starts and ends.
8. [ ] The dot positions are already written into `data/raw/dots.yaml`. Open it and check
       the numbers match what you taped. **If you had to move anything, edit that file now.**

---

## Step 3 — Position the phone

1. [ ] Mount the phone **landscape** (long edge horizontal) on the tripod.
2. [ ] Place it on the far side of the table, opposite you, centred roughly on dot 8.
3. [ ] Set the lens height to **55 cm above the table top**. Measure it.
4. [ ] Set it **40 cm back** from the near row of dots (the y = 10 row), measured
       horizontally along the table.
5. [ ] Tilt it down so it points at dot 5, roughly 45°.
6. [ ] Open the Camera app and check the preview shows **all 9 dots, the whole ArUco
       board, the rest mark, and about 10 cm of table margin around everything**.
       Adjust height and tilt until it does.
7. [ ] Tape the tripod legs to the floor or table. From now on **nothing may move the
       phone** until all 45 clips are recorded. If it moves, you must redo Step 4.

---

## Step 4 — Phone settings (do all of these once, then never touch them)

### In the Settings app

1. [ ] Settings → **Display & Brightness** → **Auto-Lock** → **Never**
2. [ ] Swipe down from the top-right → tap the **crescent moon** (Focus / Do Not Disturb) to turn it on
3. [ ] Settings → **Camera** → **Formats** → tap **Most Compatible**
       (this records H.264, which the pipeline reads reliably; "High Efficiency" can fail)
4. [ ] Settings → **Camera** → **Record Video** → tap **1080p at 30 fps**
5. [ ] Settings → **Camera** → **Record Video** → **HDR Video** → **OFF**
       (if there is no such switch, Settings → **Camera** → **Formats** → turn off
       **Apple ProRes** and any **ProRes RAW / HDR** switch)
6. [ ] Settings → **Camera** → **Record Video** → **Enhanced Stabilization** → **OFF**
       (not on every model; skip if absent)
7. [ ] Settings → **Camera** → **Record Video** → **Lock Camera** → **ON**
       (stops the phone switching lenses mid-clip)
8. [ ] Settings → **Camera** → **Record Video** → **Lock White Balance** → **ON**
9. [ ] Settings → **Camera** → **Macro Control** → **ON**
10. [ ] Settings → **Camera** → **Preserve Settings** → **Camera Mode** → **ON**
11. [ ] Settings → **Camera** → **Grid** → **ON**

### In the Camera app

12. [ ] Swipe to **VIDEO**.
13. [ ] Tap **1×**. Do not use 0.5× or 2× or 3× at any point.
14. [ ] Look at the top of the screen. If there is a **running-man** icon and it is
        highlighted yellow, tap it to turn **Action mode OFF**.
15. [ ] If a yellow **flower** icon appears at the bottom left, tap it so it is **not**
        highlighted (macro off).
16. [ ] **Tap and hold** on dot 5 in the preview for about 2 seconds, until a yellow
        **AE/AF LOCK** banner appears at the top. Let go.
17. [ ] Check the banner says AE/AF LOCK. **It must be visible before every single clip.**
        If you ever tap the screen again, repeat step 16.
18. [ ] Take a screenshot of the Settings → Camera → Record Video screen. You will want it
        for the README.

### Lighting

19. [ ] Turn on two lamps, one on each side of the table, or work in side daylight.
20. [ ] Make sure there is **no window directly behind you**.
21. [ ] Check that no hard shadow falls across the ArUco board or the block.

---

## Step 5 — Record `calib.mp4` (about 2 minutes)

You hold the checkerboard cardboard and move it. The phone stays on the tripod.

1. [ ] Tap the red button to start.
2. [ ] Hold the checkerboard so it fills the middle of the frame, flat-on, for 2 seconds.
3. [ ] Move it to the **top-left** of the frame, hold 2 seconds.
4. [ ] **Top-right**, hold 2 seconds.
5. [ ] **Bottom-left**, hold 2 seconds.
6. [ ] **Bottom-right**, hold 2 seconds.
7. [ ] Back to the middle, **tilt the top edge away from you** about 30°, hold 2 seconds.
8. [ ] **Tilt the bottom edge away** about 30°, hold 2 seconds.
9. [ ] **Tilt the left edge away** about 30°, hold 2 seconds.
10. [ ] **Tilt the right edge away** about 30°, hold 2 seconds.
11. [ ] Bring it **close**, filling about 60% of the frame, hold 2 seconds.
12. [ ] Move it **far**, filling about 20% of the frame, hold 2 seconds.
13. [ ] Lay it **flat on the table** over the dots, hold 2 seconds.
14. [ ] Tap the red button to stop.

Rules while filming: move slowly, keep the **whole** board visible at every moment, and
keep your fingers off the printed squares.

---

## Step 6 — Record `empty.mp4` (10 seconds)

1. [ ] Put the block on dot 5. Put the plate on dot 9.
2. [ ] Take your hands completely out of the frame.
3. [ ] Record 5 seconds of the still scene. Stop.

---

## Step 7 — The test clip. Do not skip this.

The three scripts below are built in Cursor Phase 1. Make sure that phase is finished
before you get here (`docs/TODO.md` tells you when).

1. [ ] Record clip number 1 exactly as described in Step 8, using block dot 1 and
       plate dot 3.
2. [ ] Transfer `calib.mp4`, `empty.mp4` and this clip to the computer now (Step 9,
       items 1–3), naming the clip `demo_001.mp4`.
3. [ ] Run, one after the other:
       ```
       uv run python scripts/calibrate.py
       uv run python scripts/debug_aruco.py clip=1
       uv run python scripts/preview_hand.py clip=1
       ```
4. [ ] Check all five of these. **Every one must pass before you record anything else.**
       - [ ] `calibrate.py` prints an RMS below **0.5 px**
       - [ ] `debug_aruco.py` prints board-origin jitter below **2 px** and the drawn axes
             sit still on the board for the whole clip
       - [ ] in the preview video, hand landmarks track your hand, and the **thumb tip and
             index tip are both visible at the moment you grasp**
       - [ ] the block mask is a clean blob covering the block only — not your skin, not
             the background — including while the block is in your hand
       - [ ] the board is never covered by your arm
5. [ ] Anything fails → go to Step 10, fix it, and redo the test clip.

---

## Step 8 — Record the 45 clips

### Before every single clip

- [ ] Block centred on its dot, correct side up
- [ ] Plate centred on its dot
- [ ] Yellow **AE/AF LOCK** banner visible
- [ ] Your right hand flat on the rest mark, left hand out of the frame
- [ ] Sleeves above the elbow, no watch, no rings on the index or thumb

### A SUCCESS clip — the exact 10 seconds

Count the seconds out loud in your head: "one thousand one, one thousand two, ...".

| Count | Do exactly this |
|---|---|
| — | Tap the red button |
| 1, 2 | **Do not move.** Hand flat on the rest mark, completely still |
| 3, 4 | Reach to the block and **pinch it from above with thumb and index only**. Curl your other fingers loosely. Turn your wrist so the thumb and index tips face the phone |
| 5, 6, 7 | Lift, carry, and set the block down **centred on the plate**. Low arc = peak about 5 cm up. High arc = peak about 15 cm up |
| 8 | **Open your fingers wide**, then lift your hand straight up about 5 cm |
| 9 | Hand back to the rest mark, lay it flat |
| 10, 11 | **Do not move.** Still for a full 2 seconds |
| — | Tap the red button |

Rules: right hand only, natural speed, **exactly one grasp and one release**, your arm
never passes over the ArUco board. Fumbled it? Delete the clip and redo the same number.

### A FAILURE clip

Identical, except counts 5–7 are replaced by one of these. Do the failure **clearly and
only once**. Keep the 2 seconds of stillness at both ends.

| Type | Replace counts 5–7 with |
|---|---|
| **F1 Miss** | Close your pinch **in mid-air, 2–3 cm to the side of the block** as listed. Do not touch it. Lift your hand 10 cm, move it to above the plate, open your fingers. The block stays on its dot and never moves |
| **F2 Drop** | Grasp the block properly, lift about 10 cm, carry it part-way to the plate, then **open your fingers in the air**. Let it fall and bounce. Do not catch it |
| **F3 Short** | Grasp properly, carry, and put the block down **on the bare table, 8–10 cm short of the plate**, then release |
| **F4 Push** | Keep your hand **open** the whole time. Move it low along the table and **push the block sideways about 10 cm** with your fingers. Never pinch. Then go back to rest |

### The list — record them in this order

Successes 1–30:

| # | Block | Plate | Arc |  | # | Block | Plate | Arc |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3 | low |  | 16 | 7 | 3 | high |
| 2 | 3 | 1 | high |  | 17 | 5 | 1 | low |
| 3 | 4 | 6 | low |  | 18 | 5 | 3 | high |
| 4 | 6 | 4 | high |  | 19 | 5 | 7 | low |
| 5 | 7 | 9 | low |  | 20 | 5 | 9 | high |
| 6 | 9 | 7 | high |  | 21 | 2 | 4 | low |
| 7 | 1 | 7 | low |  | 22 | 4 | 8 | high |
| 8 | 7 | 1 | high |  | 23 | 8 | 6 | low |
| 9 | 2 | 8 | low |  | 24 | 6 | 2 | high |
| 10 | 8 | 2 | high |  | 25 | 1 | 5 | low |
| 11 | 3 | 9 | low |  | 26 | 9 | 5 | high |
| 12 | 9 | 3 | high |  | 27 | 2 | 6 | low |
| 13 | 1 | 9 | low |  | 28 | 4 | 2 | high |
| 14 | 9 | 1 | high |  | 29 | 8 | 4 | low |
| 15 | 3 | 7 | low |  | 30 | 6 | 8 | high |

Failures 31–45:

| # | Type | Block | Plate | Exactly what to do |
|---|---|---|---|---|
| 31 | F1 | 1 | 3 | close 3 cm to the left of the block |
| 32 | F1 | 5 | 9 | close 3 cm to the right of the block |
| 33 | F1 | 7 | 2 | close 3 cm in front of the block, on your side |
| 34 | F1 | 6 | 4 | close 2 cm to the left of the block |
| 35 | F2 | 1 | 9 | drop it halfway, from about 10 cm up |
| 36 | F2 | 3 | 7 | drop it halfway, from about 15 cm up |
| 37 | F2 | 8 | 2 | drop it one third of the way |
| 38 | F2 | 4 | 6 | drop it two thirds of the way |
| 39 | F3 | 1 | 3 | put it down 8 cm short of the plate |
| 40 | F3 | 9 | 5 | put it down 10 cm short of the plate |
| 41 | F3 | 7 | 1 | put it down 8 cm short of the plate |
| 42 | F3 | 2 | 8 | put it down 10 cm short of the plate |
| 43 | F4 | 5 | 1 | push the block toward dot 6 |
| 44 | F4 | 4 | 9 | push the block toward dot 5 |
| 45 | F4 | 8 | 3 | push the block toward dot 9 |

### Every 10 clips

- [ ] Play back the last clip and check: the whole board is in frame, your thumb and index
      tips are visible at the grasp, the phone has not shifted.

---

## Step 9 — Get the files onto the computer

1. [ ] Connect the iPhone by cable, or use AirDrop. **Never WhatsApp, Messages or email** —
       they re-compress the video and the calibration becomes worthless.
2. [ ] Copy everything into `data/raw/`.
3. [ ] Rename the files exactly:
       - the calibration video → `calib.mp4`
       - the empty-scene video → `empty.mp4`
       - clip 1 → `demo_001.mp4`, clip 2 → `demo_002.mp4`, … clip 45 → `demo_045.mp4`
         (three digits, always)
       - iPhone files arrive as `.MOV`. Rename the extension to `.mp4`; the contents are
         the same container and the pipeline reads them.
4. [ ] Open `data/raw/recording_log.csv`. It is already filled in for all 45 clips.
       Change the `redo` column to `1` for any clip you had to repeat or are unhappy with.
5. [ ] Open `data/raw/dots.yaml` and confirm it matches the table you taped.
6. [ ] Take one photo of the whole setup from the side and save it as
       `docs/media/setup.jpg`. This goes in the README.
7. [ ] Copy the whole `data/raw/` folder to an external disk or Drive. It is not in git
       and it cannot be re-recorded once you move the phone.

---

## Step 10 — If something fails

| What you see | Do this |
|---|---|
| `calibrate.py` RMS above 0.5 px | Re-record `calib.mp4`, moving more slowly, with the whole board visible and better light |
| Board not detected, or the axes jump around | More light on the board; flatten it; check your arm is not crossing it; check the board is fully in frame |
| Board origin jitter above 2 px | Stabilization is interfering. Install the free **Blackmagic Camera** app, set 1080p 30 fps, stabilization **off**, focus/exposure/white balance locked, and re-record everything including `calib.mp4` |
| Thumb or index tip missing at the grasp | Rotate your wrist so both tips face the phone; pinch from the phone's side of the block |
| Block mask also catches your skin or the background | Use a more saturated blue or green block; remove anything else of that colour from the frame |
| Block mask disappears while you hold it | Pinch nearer the top edge of the block so more of it stays visible |
| Landmarks jitter | More light |
| The image wobbles or crops during a clip | Stabilization is still on — use Blackmagic Camera |
| Brightness pumps when your hand enters the frame | AE/AF LOCK is not on. Redo Step 4 item 16 |

---

## Final checklist

- [ ] `data/raw/calib.mp4`
- [ ] `data/raw/empty.mp4`
- [ ] `data/raw/demo_001.mp4` … `data/raw/demo_045.mp4` — 45 files
- [ ] The test clip passed all five checks in Step 7
- [ ] `data/raw/recording_log.csv` redo column updated
- [ ] `data/raw/dots.yaml` matches the table
- [ ] `objects.block_size` and `objects.plate_radius` set in `configs/default.yaml`
- [ ] `docs/media/setup.jpg`
- [ ] `data/raw/` backed up somewhere outside the repo
