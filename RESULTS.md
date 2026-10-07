# Results

Every phone clip puts a lid on a plate: 30 that land, and 15 misses, drops, short placements and pushes. A simulated Franka Panda is then scored on a plate, a pad and a box. The pad and the box were never filmed. Counts are in `results/v2/`. The 7 October run is in `results/v1/`.

| | Question | Score | Result |
|---|---|---|---|
| 1 | Was the finger height usable? | The trace on demo 2, in centimetres | **No, then yes.** Gray, with no measured hand, runs from −1 cm to 30 cm. Blue, with palm size, stays between 1 cm and 13 cm |
| 2 | Can the arm place the lid by copying a video? | Lid within 3 cm, right height, fingers open, settled | **148/150.** A fixed copy is 0/150 |
| 3 | Do the clips help predict the next second? | Centimetres of error. Lower is better | **7.9 cm** with the clips and 10 robot tries. **15.7 cm** with those tries alone |
| 4 | Can the model tell the lid is held? | 0.5 is a coin toss, 1 is perfect | **0.53** from the clips alone. **0.92** with five robot tries mixed in |
| 5 | Are the clips clear enough to use? | How many of 45 pass a checklist | **45/45** |

## 1. The height was not a reach. Palm size made it one

**In short.** One phone camera sees where the fingers are in the image and does not see how far away they are. The first estimate invented that distance from a generic hand, with none of the lengths measured on this hand. On demo 2 that estimate is the gray line. It dips to −1 cm and spikes to 30 cm, and the spike is at the release, where the old height is still 17 cm. That is not a hand setting a lid on a plate. The fix measures eight lengths on the back of the hand (90, 85, 80, 80, 28, 25, 25 and 72 mm, in `configs/default.yaml`) and fits them to the knuckles. The same frames become the blue line: one reach, between 1 cm and 13 cm, 8 cm at the grasp and 9 cm at the release.

### Demo 2. Finger height before and after measuring the hand

`demo_002.mp4` is a successful high reach onto a plate. Gray is the height with no palm measurement. Blue uses the eight measured lengths. The green line is the grasp. The red line is the release. The 30 cm spike sits on the red line.

![Demo 2. Gray has no measured hand and spikes to 30 cm at the release. Blue uses palm size](results/v2/depth/plots/clip002_height.png)

A stricter check still misses. It trusts only the grasp, then asks where the fingers are at the release, against where the lid is seen. On the 30 successes the typical miss is 4.6 cm in height and 11 cm in all three directions. The bar set beforehand was 1.5 cm and 4 cm. The spikes are gone. The last few centimetres are not. When the arm copies the video it does not use this height at the grasp. It uses the lid and the target.

Source: `results/v2/depth/depth_check.csv`. The gray line is `p_ee_old` in `results/v2/clips/clip_002.npz`. The blue line is `p_ee`.

| | Median on the 30 successes |
|---|---:|
| Height error at the release | 4.64 cm |
| Error in all three directions | 11.12 cm |

## 2. Can the arm place the lid by copying a video?

**In short.** The goal is to replay one plate video on the arm, once, with no learning. The arm is also asked to hit a pad and a box, which the hand never touched. A try succeeds when the lid finishes within 3 cm of the target, at the right height, fingers open, lid still. There are 150 tries, 50 on each target. A single stretch of clip 1, reused everywhere, places the lid on 0 of 150 (the range around that zero is 0% to 2.5%). Pinning the grasp to the lid and the release to the target, while still copying the video's height, places it on 56 of 150 (30% to 45%). The same two pins, with the height of the grasp and the release taken from the lid and the target, place it on 148 of 150 (95% to 100%): 50/50 on the plate, 49/50 on the pad, 49/50 on the box. On 7 October the pinned copy was 20/150. The palm-size path raised it to 56/150. The step from 56 to 148 is taking the height from the objects, which is what puts the lid on the box (8/50 when the height was copied from the video, 49/50 after).

Source: `results/v2/e1/e1.csv`. The ranges are Wilson intervals. The 7 October table is `results/v1/e1.csv`.

| Way of copying | Successes | Range |
|---|---:|---|
| One fixed stretch of clip 1 | 0/150 | 0.0% to 2.5% |
| Both ends pinned, height copied from the video | 56/150 | 30.0% to 45.3% |
| Both ends pinned, height taken from the objects | 148/150 | 95.3% to 99.6% |

### Where the copied reach lands

Each bar is the share of layouts where the lid finished within 3 cm of the target, at the right height, fingers open. Fifty layouts per target. The three bars in a group are naive, two-anchor, and hybrid. The videos are plate placements. The pad and the box were never filmed.

![Success rate of naive, two-anchor, and hybrid, split by simulator target](results/v2/e1/plots/success_by_goal.png)

Those reaches became 400 robot tries. The lid moves by more than 5 cm in 363 of them. On 7 October the copies mostly missed, so that set mostly showed a lid sitting still.

## 3. Do the clips help predict where the lid goes next?

**In short.** The goal is to see whether the plate clips teach a small model where the lid will be about one second from now, and whether that saves robot practice. The score is centimetres of error on robot tries the model was not trained on. Lower is better. Each cell is the mean of three training starts, and the ± is how far those three spread. With only three starts, that spread is a rough guide. Guessing that the lid never moves is wrong by 14.7 cm. With 10 robot tries and the clips trained in from the start, the error is 7.9 cm, against 15.7 cm for the same 10 tries without the clips. The clips help from 5 tries through 100. With no robot tries they make the guess worse, 23.5 cm. Training on the clips first and the robot afterwards jumps around from one start to the next. At 300 tries the three recipes meet near 12 cm, and the clips no longer help.

Source: `results/v2/e2/e2.csv`. The first cell, 14.71 cm, is "the lid never moves". The other two cells on that row are clips with no robot data.

| Robot tries | Robot tries only | Clips first, then robot | Both from the start |
|---:|---:|---:|---:|
| 0 | 14.71 ± 0.00 | 23.53 ± 3.76 | 23.53 ± 3.76 |
| 5 | 18.22 ± 0.60 | 15.80 ± 3.47 | 10.00 ± 0.23 |
| 10 | 15.73 ± 1.03 | 15.81 ± 4.35 | **7.91 ± 0.91** |
| 25 | 17.25 ± 0.50 | 18.98 ± 5.14 | 11.97 ± 2.39 |
| 50 | 19.92 ± 1.53 | 21.12 ± 3.91 | 10.23 ± 1.46 |
| 100 | 16.79 ± 1.55 | 17.28 ± 3.95 | 9.28 ± 0.88 |
| 300 | **11.52 ± 0.57** | 12.19 ± 2.00 | 12.52 ± 0.47 |

### Prediction error, and whether the lid is held

The left panel is centimetres of error about one second ahead. Lower is better. The dashed line is the guess that the lid never moves. The right panel asks whether a held lid is ranked above a free one. A score of 0.5 is a coin toss. The band is the spread across three training starts. The horizontal axis is the number of robot tries.

![Position error and attach score against the number of robot tries](results/v2/e2/plots/learning_curves.png)

### Is the lid in the hand?

**In short.** This score ignores the centimetres. It asks whether a held lid is ranked above a free one. The clips alone score 0.53, a coin toss. Five robot tries score about 0.87. Mixing the clips into those five tries scores about 0.92. At 300 tries every recipe is near 0.95.

| Robot tries | Robot tries only | Clips first, then robot | Both from the start |
|---:|---:|---:|---:|
| 0 | 0.50 ± 0.00 | 0.53 ± 0.04 | 0.53 ± 0.04 |
| 5 | 0.87 ± 0.01 | 0.92 ± 0.01 | 0.92 ± 0.01 |
| 10 | 0.90 ± 0.01 | 0.94 ± 0.01 | 0.94 ± 0.01 |
| 300 | 0.95 ± 0.00 | 0.95 ± 0.01 | 0.94 ± 0.00 |

## 4. Are the clips clear enough to use?

**In short.** A clip passes when the hand and the lid stay visible, the lid starts within 2 cm of its mark, and the fingers close about as often as that kind of clip should. A success must close once. A miss, a drop, a short placement or a push may close from once to eight times, and a push with no close still passes. All 45 pass. On 7 October, 36 passed, because a second close on a miss was thrown out. The palm-size fit itself is solid on about two thirds of the frames of a typical clip. Gaps are filled afterwards, so the visibility number looks cleaner than the raw measurement.

Source: `results/v2/qc/extraction_report.csv`. Column `depth_ok_frac` in the depth csv has median 0.65.

## What to remember

The gray height was not a reach: on demo 2 it ran from −1 cm to 30 cm, with the spike at the release. Measuring the palm replaced it with one reach between 1 cm and 13 cm.

Copying a plate video, with both ends pinned and the height taken from the objects, places the lid on 148 of 150 layouts, including a pad and a box the hand never touched. A fixed copy places it on none.

With 10 robot tries, training on the clips as well predicts the lid to 7.9 cm, against 15.7 cm without them. The clips alone do not replace the robot tries. At 300 tries they add nothing.

The release check is still 4.6 cm high. The arm does not trust that height at the grasp.
