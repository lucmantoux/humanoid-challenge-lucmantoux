# Filming Guide — do exactly this, in this order

You are recording 45 clips of your own right hand moving a coloured cube onto a plate.
An iPhone on a tripod films all of them. The stock Camera app is enough. Total time
about 2 hours. Tick the boxes as you go.

Every distance below is already the number the code expects. Do not round them, swap
them, or "improve" the layout. If a printed square is the wrong size, reprint. Do not
edit the config to match a wrong print.

---

## What the camera is actually looking for

Four different things go on or near the table. They are not interchangeable.

| Thing | What it looks like | Where it goes | What it is for |
|---|---|---|---|
| **Page 1, the ArUco board** | One A4 sheet with **four** QR-like squares in a 2×2 block, plus the word ORIGIN | Taped flat on the table, phone side, your left | The camera's only landmark. It tells the code where the table is, which way is right, and which way is toward you. |
| **Page 2, the checkerboard** | One A4 sheet of chess squares | Taped flat onto cardboard. You **hold it in your hand** for one video, `calib.mp4` | Teaches the code the phone's lens. It never stays on the table. |
| **Nine dots** | Masking tape or round stickers, numbered 1–9 in marker | A 3×3 grid on the table | Parking spots for the cube and the plate. The camera does **not** read the dots. You use them so every clip starts and ends in a known place. |
| **Rest mark** | One strip of tape, about 8 cm long | To the right of dot 3 | Where your right hand lies flat and still at the start and end of every clip. |

The camera finds the **cube by its colour**. It finds your **hand by the thumb and the index finger**. It finds the **table by the four squares**. A glossy marble table, a second phone lying in frame, or a cube the same colour as the table all break that.

The 5 October session (clips, calibration, colour range, and extracted states) is in `data/archive/session_2026-10-05/`. `data/raw/` now holds only `dots.yaml` and `recording_log.csv`. Drop the new files there under the Step 9 names: `calib.mp4`, `empty.mp4`, `demo_001.mp4` … `demo_045.mp4`. Do not put the archived files back, and do not add `clip_files.csv`: that map belonged to the old numbering, and a new one would send the wrong file to each clip id.

---

## Step 0 — Put these on the table before you start

- [x] iPhone
- [x] Tripod, or a stack of books plus tape. The phone must not move at all for the whole session.
- [x] The printed pack from Step 1: page 1 (four squares) and page 2 (chessboard)
- [x] A rigid piece of cardboard, at least A4, for the chessboard
- [x] A **block 3–5 cm across, saturated blue or green**. A cube is the default. A cylinder is fine if you always stand it on a flat end and never lay it on its side: the camera finds it by colour, not by shape. Not red, orange, pink, beige, white, or wood. Red wraps around the colour wheel and the code rejects it. Skin-coloured and white objects disappear into your hand and the table. It must be the only object of that colour anywhere the phone can see, including your shirt. Everywhere below, "cube" means this object.
- [x] A plain plate, about 18 cm across, **not** the cube's colour
- [x] Masking tape, or 10 small round stickers
- [x] A tape measure or a ruler at least 50 cm long
- [x] A marker pen
- [x] Optional, not used in these 45 clips: a coaster and a box about 8 cm on a side

The 45 clips all use the plate. The coaster and the box are for the simulator later. Do not put them in these videos.

---

## Step 1 — Print the two sheets and measure them

The code assumes the black squares have a real-world size. If the printer shrinks the page, every centimetre in the project is silently wrong.

1. [x] In the repo folder run:
       ```
       uv run python scripts/make_board.py
       ```
       It writes `results/print/board_a4.pdf`, `results/print/table_layout.png`
       and `results/print/setup_schematic.png`. The schematic is the dimensioned drawing.
       If it prints `skip: board_a4.pdf exists`, the file is already there. Open that file. Do not pass `overwrite=true` unless you want to regenerate it.
2. [x] Open `results/print/table_layout.png` and leave it on screen. It is the map for Step 2.
3. [x] Open `results/print/board_a4.pdf`. It has two pages.
4. [x] Press Cmd-P. Set **Paper Size: A4** and **Scale: 100%**. If you see "Scale to Fit" or "Fit to Page", turn it off. Print **both** pages.
5. [x] Look at what came out of the printer before you tape anything:
       - **Page 1** has the title "palm-prior — ArUco table board", **four** black-and-white squares in a 2×2, the word ORIGIN, an x arrow and a y arrow, and a scale bar labelled "100 mm — measure this". This sheet goes on the table. Do not cut the four squares apart. If you can only see three squares, one was cut off the edge of the paper: reprint page 1.
       - **Page 2** has the title "palm-prior — calibration checkerboard" and a chess pattern. This sheet is held in your hand. It does not go on the table.
6. [x] With a ruler, measure one black square on page 1, outer edge to outer edge. **It must read 60 mm.** Measure the printed scale bar. **It must read 100 mm.**
       - Not 60 mm, or not 100 mm → the printer scaled the page. Reprint at 100%. Do not continue, and do not change any number in the config to compensate.
7. [x] Measure one square of the chessboard, edge to edge. **It must read 25 mm.** Same rule: reprint if it is anything else.
8. [x] Tape page 2 onto the cardboard so the whole chess pattern is flat. No curl, no bubble, no tape crossing a black square. Set this card aside. You only pick it up again in Step 5.
9. [ ] Measure the block's **height**, table to top, with it standing the way you will place it in every clip. A cylinder must stand on a flat end for this measurement and for all 45 clips. Read it in centimetres to the nearest millimetre. Write it here: block height = ______ cm
       Also write the width across: ______ cm. The code uses the height only. The simulator block is a cube of that same height, so a cylinder whose height and width are both about 3–5 cm is the case this project was written for. If those two numbers differ by more than about 1 cm, stop and say so before recording the 45 clips.
10. [ ] Measure the plate. Measure the full width from rim to rim, then divide by 2. That half-width is the radius. Write it here: plate radius = ______ cm
        Example: a plate 18.0 cm across has radius 9.0 cm.
11. [ ] Open `configs/default.yaml`. Set these two lines, in **metres** (divide your centimetres by 100):
        - `objects.block_size:` — 4.0 cm is `0.04`
        - `objects.plate_radius:` — 9.0 cm is `0.09`
        Change only those two numbers. Leave everything else.

---

## Step 2 — Tape the board, the nine dots, and the rest mark

Open `results/print/setup_schematic.png` and keep it beside the table. It has the centimetre dimensions. `results/print/table_layout.png` only shows which printed sheet is which. Both pictures are drawn as you see the table from your chair: **YOU at the top, the PHONE at the bottom.**

All positions are measured from one corner of the ArUco sheet, so you measure from that corner. You do not measure the gap between dots and hope.

### Words used from here on

Sit down first. Everything below uses **your** left and **your** right as you sit in that chair.

- **Toward you** means toward your chair.
- **Toward the phone** means toward the opposite side of the table, where the tripod will stand.
- **x** is centimetres to your right of the ORIGIN corner.
- **y** is centimetres toward you from the ORIGIN corner.

### Clear the surface

1. [x] Clear a flat area about **75 cm wide and 55 cm deep**.
2. [x] The surface must be **matte and plain**: no gloss, no marble veins, no patterned cloth, no second phone, no mug. A glossy table makes bright streaks that the camera treats as part of the board or the cube. If the table is shiny, cover the working area with a large sheet of plain matte paper or card, light coloured, and tape that down. The cover must not be the cube's colour.
3. [x] Light the table from the sides, or with daylight from the side. Do not sit with a window directly behind you: the camera then looks into the bright background and your hand becomes a silhouette.

### Tape page 1

4. [ ] Sit in the chair you will use for all 45 clips. The phone will stand on the opposite side, facing you.
5. [ ] Take page 1 to the **phone's side of the table, on your left**. Hold it where the phone will be and check all four of these before you tape it. They are one orientation; if one is wrong, the sheet is rotated.
       - Standing on the phone's side, looking toward your chair, the title "palm-prior — ArUco table board" reads normally, not upside down.
       - The word **ORIGIN** is at the corner **nearest the phone**, on the **left** as you look from the phone toward your chair.
       - The **x** arrow points to your right.
       - The **y** arrow points toward your chair.
       From the chair, the title is upside down. That is correct.
6. [ ] Tape all four edges of the sheet flat. Press the middle down. The paper must not lift, curl, or catch glare. Do not put tape across the four squares.

The four squares together are 135 mm by 135 mm. The ORIGIN corner is the zero of every measurement below.

### Place the nine dots

7. [ ] Lay the tape measure on the table with the **0 mark on the ORIGIN corner**, running along the **x** arrow, to your right. Mark these four distances with a pencil dot or a bit of tape, still on that line: **22 cm, 37 cm, 52 cm, 64 cm**.
8. [ ] From the **22 cm** mark, measure **toward your chair** (the direction of the y arrow) and stick a dot at each of these:
       - 10 cm toward you → write **7** next to it
       - 25 cm toward you → write **4**
       - 40 cm toward you → write **1**
9. [ ] From the **37 cm** mark, same direction:
       - 10 cm → **8**
       - 25 cm → **5**
       - 40 cm → **2**
10. [ ] From the **52 cm** mark, same direction:
        - 10 cm → **9**
        - 25 cm → **6**
        - 40 cm → **3**
11. [ ] Write the number **beside** the dot, not on top of it, so the dot itself stays a small clear mark. Half a centimetre of error is fine. A swapped number is not.

Read as you sit, the grid is:

```
                          YOU, sitting here
        (1)            (2)            (3)         [REST]
        y = 40 cm, nearest you

        (4)            (5)            (6)
        y = 25 cm

        (7)            (8)            (9)
        y = 10 cm, nearest the phone

      ┌─────────┐
      │ 4 squares│   ORIGIN = the corner of this sheet nearest the phone
      └─────────┘
                         PHONE, landscape, looking at you
      x = 0 at ORIGIN, increasing to your right
```

Dots 1, 2, 3 are the row nearest you. Dots 7, 8, 9 are the row nearest the phone. Dot 5 is the centre. The ArUco sheet is to the left of dots 7, 4 and 1, not underneath them.

12. [ ] Tape the **rest mark**: a strip of tape about **8 cm long**, at **64 cm to the right of ORIGIN** and **40 cm toward you**. That is 12 cm to the right of dot 3, in the same row as dots 1, 2 and 3. This is where your right hand starts and ends. It is not an extra dot and it has no number.

### Check the file

13. [ ] Open `data/raw/dots.yaml`. It already contains the nine positions in centimetres. They must match what you taped:

        | Dot | x | y |   | Dot | x | y |
        |---|---|---|---|---|---|---|
        | 1 | 22 | 40 |   | 6 | 52 | 25 |
        | 2 | 37 | 40 |   | 7 | 22 | 10 |
        | 3 | 52 | 40 |   | 8 | 37 | 10 |
        | 4 | 22 | 25 |   | 9 | 52 | 10 |
        | 5 | 37 | 25 |   |   |    |    |

        If you had to shift the whole grid, measure the new positions from ORIGIN and edit those numbers now. If you taped the grid as written, do not touch the file.

---

## Step 3 — Put the phone where it will stay

The code assumes one frozen camera. If the phone moves by a few millimetres after this step, the calibration and every clip after it are wrong, and you redo from Step 4.

1. [ ] Mount the phone **landscape**: the long edge of the phone is horizontal, the camera lens on the side facing the table.
2. [ ] Stand it on the opposite side of the table from your chair, centred on **dot 8**.
3. [ ] Measure from the table top up to the **lens** (the glass dot), not to the top of the phone. Set that height to **55 cm**.
4. [ ] The row of dots 7, 8 and 9 is the row nearest the phone. From that row, measure **40 cm further away from your chair**, along the table, and put the tripod there. The phone is beyond the dots, on its own side. It is not sitting on the dots.
5. [ ] Tilt the phone down so the lens points at **dot 5**. The tilt is about 45° from horizontal. You will correct it by looking at the screen, not by measuring the angle.
6. [ ] Open the Camera app. The preview must show, all at once:
        - all **9** numbered dots
        - the **whole** ArUco sheet, including the white paper around the four squares
        - the **rest mark**
        - about **10 cm of bare table** around the outside of all of that
        If a dot or the board is cut off, change the height or the tilt. Do not zoom. Do not switch to 0.5×.
7. [ ] Tape the tripod feet to the floor, or tape the book stack to the table. From this moment until clip 45 is transferred to the computer, do not bump the phone, the table, or the tripod. If it moves, redo Step 4 and re-record `calib.mp4` before any new clip.

---

## Step 4 — Lock the phone settings, then never touch them

The pipeline needs one resolution, one lens, and a picture whose brightness does not pump when your hand enters. Do every item once.

### In the Settings app

1. [ ] Settings → **Display & Brightness** → **Auto-Lock** → **Never**. The screen must not sleep mid-clip.
2. [ ] Swipe down from the top-right corner. Tap the **crescent moon** so Focus / Do Not Disturb is on. A notification must not cover the record button.
3. [ ] Settings → **Camera** → **Formats** → **Most Compatible**. This records H.264. "High Efficiency" (HEVC) sometimes fails to open later.
4. [ ] Settings → **Camera** → **Record Video** → **1080p at 30 fps**. Not 4K, not 60 fps.
5. [ ] Settings → **Camera** → **Record Video** → **HDR Video** → **OFF**. If that switch is missing: Settings → **Camera** → **Formats**, and turn off **Apple ProRes** and any **ProRes RAW / HDR** switch.
6. [ ] Settings → **Camera** → **Record Video** → **Enhanced Stabilization** → **OFF**. Skip this item if your phone has no such switch. Stabilisation crops and warps the frame, which makes the lens measurement impossible.
7. [ ] Settings → **Camera** → **Record Video** → **Lock Camera** → **ON**. This stops the phone changing lenses in the middle of a clip.
8. [ ] Settings → **Camera** → **Record Video** → **Lock White Balance** → **ON**.
9. [ ] Settings → **Camera** → **Macro Control** → **ON**. This only adds a switch in the Camera app. You will turn macro itself off in item 15.
10. [ ] Settings → **Camera** → **Preserve Settings** → **Camera Mode** → **ON**. So it stays on Video.
11. [ ] Settings → **Camera** → **Grid** → **ON**. The grid is only a guide for you.

### In the Camera app

12. [ ] Open Camera and swipe to **VIDEO**. The red record button is on this mode. Photo mode is the wrong mode.
13. [ ] Tap **1×**. Never tap 0.5×, 2×, or 3×. The wide lens and the tele lens are different cameras; the calibration would not match the clips.
14. [ ] If a **running-man** icon at the top is yellow, tap it so Action mode is **off**.
15. [ ] If a yellow **flower** icon is showing at the bottom left, tap it so it is **not** yellow. That is macro, and it must stay off.
16. [ ] **Tap and hold** on **dot 5** in the preview for about 2 seconds, until a yellow banner **AE/AF LOCK** appears at the top. Let go. Do not tap the picture again.
        That lock freezes focus and brightness. Without it, the picture brightens and darkens every time your hand enters, and the cube's colour stops matching.
17. [ ] The banner must say AE/AF LOCK. It must be visible before every clip. If you tap the screen, the banner disappears and you repeat item 16.
18. [ ] Check the lock: wave your hand over the table. The brightness of the table must **not** change. If it does, the banner is gone. Repeat item 16.
19. [ ] Take a screenshot of Settings → Camera → Record Video and keep it. It is the record of which switches were off.

---

## Step 5 — Record `calib.mp4`

This is the only video in which you hold the chessboard. The phone stays on the tripod. You move the cardboard.

Why: the code finds the corners of the 25 mm squares in many positions and computes the lens from them. It needs the **entire** chessboard inside the frame, sharp, and not hidden by your fingers, in each of the poses below.

1. [ ] Confirm the yellow **AE/AF LOCK** banner is showing.
2. [ ] Pick up the cardboard. Keep both hands on the **edges** or the **back**. No finger on a black or white square.
3. [ ] Tap the red button to start.
4. [ ] Hold the board in the **middle** of the frame, facing the phone flat, for 2 seconds. The whole pattern is visible.
5. [ ] Move it slowly to the **top-left** of the frame. Hold 2 seconds. Whole pattern still visible.
6. [ ] **Top-right**. Hold 2 seconds.
7. [ ] **Bottom-left**. Hold 2 seconds.
8. [ ] **Bottom-right**. Hold 2 seconds.
9. [ ] Back to the middle. Tilt the **top edge away from the phone** about 30°. Hold 2 seconds.
10. [ ] Tilt the **bottom edge away from the phone** about 30°. Hold 2 seconds.
11. [ ] Tilt the **left edge away** about 30°. Hold 2 seconds.
12. [ ] Tilt the **right edge away** about 30°. Hold 2 seconds.
13. [ ] Bring it **closer**, until the pattern fills about 60% of the frame. Hold 2 seconds. Do not let it leave the frame.
14. [ ] Move it **farther**, until the pattern fills about 20% of the frame. Hold 2 seconds.
15. [ ] Lay it **flat on the table**, on top of the dots, the whole pattern visible. Hold 2 seconds.
16. [ ] Tap the red button to stop.

Move slowly between holds. If the board goes out of frame, or a finger covers a square, delete this video and do Step 5 again. The file on the phone can stay with its default name for now. You will rename it to `calib.mp4` in Step 9.

---

## Step 6 — Record `empty.mp4`

One short still shot. No hands. This is the picture the code uses to learn the cube's colour. Without this file, `preview_hand.py` stops immediately and does not look at your hand.

1. [ ] Put the cube on **dot 5**, centred, the coloured face up.
2. [ ] Put the plate on **dot 9**, centred.
3. [ ] Take **both hands** out of the frame. Step back so your shirt is out of the frame too.
4. [ ] Confirm AE/AF LOCK is still showing.
5. [ ] Tap the red button. Wait **5 seconds** without touching anything. Tap the red button again.

The cube, the plate, all nine dots, and the whole ArUco sheet are in frame, and nothing moves. You will rename this file to `empty.mp4` in Step 9.

---

## Step 7 — Record one test clip and run the three checks

Do not record clips 2–45 until all five checks at the end of this step pass. This is the gate. A bad test clip means the other 44 would be filmed into the same mistake.

`data/raw/` should contain no video yet. The first file you copy in as `demo_001.mp4` is this test clip.

### Record it

1. [ ] This is clip **1** from the list in Step 8: cube on **dot 1**, plate on **dot 3**, **low** arc (the hand peaks about 5 cm above the table).
2. [ ] Follow the success script in Step 8 exactly, including 2 seconds of a flat still hand at the start and 2 seconds at the end.
3. [ ] Copy three videos onto the computer now, using Step 9 items 1–3 only:
        - the chessboard video → `data/raw/calib.mp4`
        - the still shot → `data/raw/empty.mp4`
        - this clip → `data/raw/demo_001.mp4`
        Cable or AirDrop. Not WhatsApp, Messages, or email.

### Run the three scripts

4. [ ] In the repo folder, one after another:
        ```
        uv run python scripts/calibrate.py
        uv run python scripts/debug_aruco.py clip=1
        uv run python scripts/preview_hand.py clip=1
        ```
5. [ ] The third script opens a window on the first frame of `empty.mp4`. **Click once on the centre of the cube**, then press any key. It asks only the first time. The colour is saved to `data/calib/block_hsv.yaml`. If you click the table instead of the cube, delete that yaml file and run the third script again.

### Five checks — every one must pass

6. [ ] `calibrate.py` prints an RMS **below 0.5 px**. Above 0.5 means the lens measurement is too sloppy to use. Re-record `calib.mp4` more slowly, with the whole chessboard visible and more light.
7. [ ] `debug_aruco.py` prints board-origin jitter **below 2 px**. Then watch `results/debug_aruco_demo_001.mp4`:
        - a yellow dot sits on the **ORIGIN** corner of the printed sheet for the whole clip
        - the **red** axis runs from that corner along the bottom edge of the squares, to the right
        - the **green** axis runs from that corner up the left edge, toward your chair
        - the **blue** axis points **up off the table**, toward the camera
        Blue pointing down into the table means the sheet is rotated. Rotate page 1 until the checks in Step 2 item 5 are true, then redo this test clip. Axes that jump mean glare, a curled sheet, or your arm covering the squares.
8. [ ] Watch `results/preview_hand_demo_001.mp4`. A stick skeleton stays on your hand. At the moment you pinch, a yellow dot is on the **thumb tip** and a yellow dot is on the **index tip**, and a green dot sits between them. If a tip has no dot, the camera could not see that fingertip: turn your wrist so both tips face the phone and record the test clip again.
9. [ ] In that same preview, an orange outline covers the **cube only**. It does not cover your skin, the plate, or the table. The outline is still there while the cube is in your hand. If it grabs your skin, the cube is not a strong enough blue or green, or something else of that colour is in frame. If the outline vanishes while you hold the cube, pinch nearer the top of the cube so more of the colour stays visible.
10. [ ] Your arm never passes over the ArUco sheet. If it does, the table landmark disappears for those frames. Reach to the cube from the rest mark, which is to the right, and stay to the right of the sheet.

Anything fails → Step 10, fix that one thing, delete the bad test clip, and do Step 7 again. Do not start clip 2.

---

## Step 8 — Record clips 1 to 45

Clip 1 is the test clip you already recorded, if it passed Step 7. Continue at clip 2. If you re-recorded clip 1 during the test, that new file is clip 1.

### Before you press record, every time

- [ ] Cube centred on its numbered dot, coloured face up
- [ ] Plate centred on its numbered dot
- [ ] Yellow **AE/AF LOCK** banner visible. Wave a hand: brightness must not change.
- [ ] Right hand flat on the rest mark. Left hand out of the frame, in your lap or behind your back.
- [ ] Sleeve above the elbow. No watch. No ring on the thumb or the index finger. Those hide the two points the tracker needs.

### How to pinch

Right hand only. Pinch **from above** with **thumb and index only**. Curl the other three fingers loosely into the palm. Turn your wrist so the **thumb tip and the index tip both face the phone**. The camera has to see both tips at the moment they meet.

When you let go, open the thumb and index **wide**, then lift the hand straight up about 5 cm, then return to the rest mark.

Count the seconds in your head: "one thousand one, one thousand two, …". Natural speed. Exactly **one** pinch and **one** opening per success clip. If you fumble, delete that clip on the phone and record the same number again.

Your forearm stays to the right of the ArUco sheet. It never crosses the four squares.

### A success clip — the 11 seconds

| Count | Do exactly this |
|---|---|
| — | Tap the red button. |
| 1, 2 | Do not move. Hand flat on the rest mark, completely still. These two seconds are how the code learns the size of your hand. A moving hand here makes every height in the clip wrong. |
| 3, 4 | Reach to the cube and pinch it from above, thumb and index, both tips facing the phone. |
| 5, 6, 7 | Lift, carry, and set the cube down **centred on the plate**. **Low** arc: the hand peaks about **5 cm** above the table. **High** arc: the hand peaks about **15 cm** above the table. The list below says which arc. |
| 8 | Open thumb and index wide. Then lift the empty hand straight up about 5 cm. |
| 9 | Hand back onto the rest mark, laid flat. |
| 10, 11 | Do not move. Still for a full 2 seconds. |
| — | Tap the red button. |

### A failure clip

Same opening (counts 1–4) and same ending (counts 8–11: open or return, then two still seconds). Replace only counts 5, 6 and 7 with the failure in the list. Do the failure **once**, clearly. The two still seconds at each end stay.

| Type | What replaces counts 5, 6 and 7 |
|---|---|
| **F1 Miss** | Close the pinch in mid-air, beside the cube, at the distance the list gives. Do not touch the cube. Lift the closed hand about 10 cm, move it until it is above the plate, then open the fingers. The cube stays on its dot and never moves. |
| **F2 Drop** | Pinch the cube properly, lift it, carry it part of the way to the plate, then **open your fingers in the air**. Let the cube fall and bounce. Do not catch it. |
| **F3 Short** | Pinch properly, carry it toward the plate, and put it down on the **bare table**, short of the plate, at the distance the list gives. Then open your hand. "Short" means along the line from the cube's dot to the plate's dot, that many centimetres **before** the centre of the plate. |
| **F4 Push** | Do not pinch at all. Keep the hand open the whole clip. Slide the open fingers along the table and push the cube about 10 cm toward the dot named in the list. Then go back to the rest mark. |

Directions in the F1 rows are from your chair: **left** is your left, **right** is your right, **toward you** is toward your chair.

### The list — record in this order

Successes 1–30. Block and plate columns are dot numbers.

| # | Block dot | Plate dot | Arc |  | # | Block dot | Plate dot | Arc |
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

| # | Type | Block dot | Plate dot | Do this |
|---|---|---|---|---|
| 31 | F1 | 1 | 3 | close the pinch 3 cm to **your left** of the cube |
| 32 | F1 | 5 | 9 | close the pinch 3 cm to **your right** of the cube |
| 33 | F1 | 7 | 2 | close the pinch 3 cm **toward you** from the cube |
| 34 | F1 | 6 | 4 | close the pinch 2 cm to **your left** of the cube |
| 35 | F2 | 1 | 9 | drop it halfway to the plate, from about 10 cm up |
| 36 | F2 | 3 | 7 | drop it halfway to the plate, from about 15 cm up |
| 37 | F2 | 8 | 2 | drop it about one third of the way to the plate |
| 38 | F2 | 4 | 6 | drop it about two thirds of the way to the plate |
| 39 | F3 | 1 | 3 | set it down 8 cm before the plate |
| 40 | F3 | 9 | 5 | set it down 10 cm before the plate |
| 41 | F3 | 7 | 1 | set it down 8 cm before the plate |
| 42 | F3 | 2 | 8 | set it down 10 cm before the plate |
| 43 | F4 | 5 | 1 | push the cube toward **dot 6** |
| 44 | F4 | 4 | 9 | push the cube toward **dot 5** |
| 45 | F4 | 8 | 3 | push the cube toward **dot 9** |

### Every 10 clips

- [ ] Play the clip you just shot, on the phone. Check three things: the whole ArUco sheet is in frame, both fingertips are visible at the pinch, and the framing matches the first clip. If the framing has shifted, the phone moved. Stop. Redo Step 4's lock, re-record `calib.mp4`, and redo the test in Step 7 before continuing.

---

## Step 9 — Copy the files onto the computer

1. [ ] Connect the iPhone with a cable, or use AirDrop. **Do not** send the videos through WhatsApp, Messages, Mail, or any app that compresses them. A recompressed video makes the calibration useless.
2. [ ] Copy all of them into `data/raw/`.
3. [ ] Rename exactly:
        - chessboard video → `calib.mp4`
        - still shot of cube and plate → `empty.mp4`
        - clip 1 → `demo_001.mp4`
        - clip 2 → `demo_002.mp4`
        - …
        - clip 45 → `demo_045.mp4`

        Always three digits. The phone usually names files `.MOV`. Change the extension to `.mp4`. The contents are the same, and the scripts open `.mp4`.
4. [ ] Open `data/raw/recording_log.csv`. One row per clip is already filled in, matching the list above. If you had to redo a clip, or you are unhappy with one, set that row's last column `redo` from `0` to `1`. Leave the other columns alone.
5. [ ] Open `data/raw/dots.yaml` and confirm the nine positions are the ones you taped.
6. [ ] Take one photo of the whole setup from the side, showing the phone, the board, and the dots. Save it as `docs/media/setup.jpg`.
7. [ ] Copy the whole `data/raw/` folder to an external disk or Drive. These files are not in git. Once you move the phone, you cannot re-record them and still use the same calibration.

---

## Step 10 — If a check fails

| What you see | What it means | Do this |
|---|---|---|
| `preview_hand.py` says `empty.mp4 not found` | The still shot was not copied or not renamed | Record Step 6 and copy it to `data/raw/empty.mp4`, then run the script again |
| `calibrate.py` RMS above 0.5 px | The chessboard views were too few, too fast, or partly hidden | Re-record `calib.mp4`. Slower, whole board in frame, more light |
| Board not found, or the coloured axes jump | The four squares were dark, curled, glaring, or covered | More light, flatten the sheet, keep your arm off it, keep the whole sheet in frame |
| Board-origin jitter above 2 px, or the picture wobbles or crops during a clip | Stabilisation is still on, so the lens changes every frame | Install the free **Blackmagic Camera** app. Set 1080p, 30 fps, stabilisation off, focus locked, exposure locked, white balance locked. Re-record `calib.mp4` first, then the test clip, then the rest |
| Blue axis points down into the table | Page 1 is rotated | Turn the sheet until ORIGIN is the corner nearest the phone and the y arrow points toward your chair. Redo the test clip |
| Thumb tip or index tip has no dot at the pinch | That fingertip was edge-on or hidden | Rotate your wrist so both tips face the phone. Pinch from the phone's side of the cube |
| Orange outline covers skin or the table | The cube is not a distinct saturated colour, or you clicked the wrong pixel | Use a stronger blue or green cube. Delete `data/calib/block_hsv.yaml`, run `preview_hand.py` again, and click the centre of the cube |
| Orange outline vanishes while you hold the cube | Your fingers hid the colour | Pinch nearer the top edge so a face of the cube stays visible |
| Skeleton jitters | The hand is too dark | Add a lamp from the side |
| Brightness changes when your hand enters | AE/AF LOCK is off | Tap and hold on dot 5 until the yellow banner returns. Do this before every clip |

Still failing after about 45 minutes on the test clip: switch to Blackmagic Camera as in the jitter row, re-record `calib.mp4`, and run Step 7 again. Do not film all 45 until the five checks pass.

---

## Final checklist

- [ ] Printed page-1 square measures 60 mm, scale bar 100 mm, chessboard square 25 mm
- [ ] `objects.block_size` and `objects.plate_radius` in `configs/default.yaml` match your cube and your plate, in metres
- [ ] Nine numbered dots and the rest mark match `data/raw/dots.yaml`
- [ ] `data/raw/calib.mp4`
- [ ] `data/raw/empty.mp4`
- [ ] `data/raw/demo_001.mp4` … `data/raw/demo_045.mp4` — 45 files
- [ ] The test clip passed all five checks in Step 7
- [ ] `recording_log.csv` column `redo` set to 1 on any clip you repeated
- [ ] `docs/media/setup.jpg`
- [ ] `data/raw/` copied somewhere that is not this computer
