# palm-prior — Technical Explainer

> **One line:** I teach a robot's *world model* how objects behave, using 45 phone clips of
> my own hand, **including clips where I fail on purpose**. Then I adapt it to a simulated
> Franka Panda with very little robot data, and the robot plans with it, starting its search
> from a path retargeted from my hand.
>
> **The claim I test:** *30 successful and 15 deliberately failed clips of my hand reduce how
> much robot data a world model needs, and make planning with it safer.*

Everything trains and runs on a laptop CPU. No GPU, no Colab, no CUDA.

This document is in three parts.

| Part | What it covers | Read it if |
|---|---|---|
| **I — Theory** | The ideas, the equations and small worked examples. No code, no file names. | You want to know *why* any of this should work |
| **II — The brief** | What Humanoid asked for, how this answers it, what I rejected and why | You are assessing the submission |
| **III — Implementation** | Architecture, module by module: inputs, outputs, defaults, file formats, failure modes | You are running or reading the code |

Part III is the source of truth for the code. If an implementation has to deviate from it,
the document gets fixed first.

Frames used throughout: **C** = camera (OpenCV convention), **T** = table (origin at the
printed board corner, z up), **W** = simulator world (z up, floor at 0).
Units: metres, radians, seconds.

---
---

# Part I — Theory

## 1. Model-based control, and what the model actually has to predict

A **policy** maps what the robot sees to what it does. A **world model** is different: it
maps a state and a candidate action to what would happen next. With a world model you can
pick actions by imagination — try a thousand plans inside the model, keep the best one,
execute a bit of it, repeat. Nothing needs to be differentiable and nothing needs a reward
signal collected by trial and error.

That matters here because of where the data comes from. A policy learned from my hand would
have to output *robot* actions, which my hand never produced. A world model learned from my
hand only has to answer a question that is the same for both bodies: **if something closes
around this block and moves, what does the block do?** Contact, lifting, dropping and
sliding are properties of the block and the table, not of the arm. That is the part worth
transferring.

Two consequences shape everything below:

- The model predicts the **object**, not the arm. Where the end-effector goes next is given
  by the action itself, so there is nothing to learn there.
- The model predicts a **change**, \( \Delta p_{obj} \), not the next absolute position. Most
  of the time the block does not move, so \( \Delta \approx 0 \) is the correct default
  and the network only has to learn departures from "nothing happens". Predicting absolute
  positions wastes capacity on re-encoding the input.

## 2. The shared state and action

This is the core design decision, and the reason human and robot data can be mixed at all.

**State** \( s \in \mathbb{R}^8 \):

| Index | Symbol | Meaning |
|---|---|---|
| 0–2 | \( d_{eo} = p_{ee} - p_{obj} \) | hand/gripper relative to the block (3) |
| 3–4 | \( d_{to} = p_{tgt}^{xy} - p_{obj}^{xy} \) | target relative to the block, horizontal (2) |
| 5 | \( z_{obj} \) | block centre height above the surface |
| 6 | \( h_{tgt} \) | target top height (plate 0.015, pad 0.005, box 0.08 m) |
| 7 | \( g \) | gripper / pinch closed (1) or open (0) |

**Action** \( a \in \mathbb{R}^4 \): \( a = [\,p_{ee}(t{+}1) - p_{ee}(t),\ g(t{+}1)\,] \),
i.e. where the hand or gripper goes next and whether it is closed. Control rate **10 Hz**.

**Why this transfers across bodies:**

- **Translation invariance.** Every horizontal quantity is a *difference*. Shift the whole
  scene by a vector \( c \) and \( s \) does not change. The model never learns "where on
  the table", only "how things relate" — so my kitchen table and the robot's floor need no
  alignment whatsoever. There is no hand-eye calibration step in this project because there
  is nothing to calibrate.
- **Gravity stays absolute.** \( z_{obj} \) and \( h_{tgt} \) are not differences. Falling
  and resting depend on true height, so those stay in absolute terms.
- **The actor is known, the object is not.** The only unknown is \( \Delta p_{obj} \), plus
  whether the block is **attached** to the hand. That is the physics: does it follow, stay
  put, fall, or slide.

**State update given a predicted block motion \( \Delta\hat p_{obj} \):**

$$d_{eo}' = d_{eo} + a_{xyz} - \Delta\hat p_{obj},\quad
d_{to}' = d_{to} - \Delta\hat p^{xy}_{obj},\quad
z_{obj}' = z_{obj} + \Delta\hat p^{z}_{obj},\quad
h_{tgt}' = h_{tgt},\quad g' = a_g$$

**Worked example.** Hand 1 cm above a 4 cm block, fingers closed, target 15 cm in x and
5 cm in y away, plate goal. The action lifts by 2 cm.

\( s = [0,0,0.01,\ 0.15,0.05,\ 0.02,\ 0.015,\ 1] \), \( a = [0,0,0.02,1] \).

The model predicts \( \Delta p_{obj} = [0,0,0.019] \) — the block came up with the hand.
Then \( s' = [0,0,0.011,\ 0.15,0.05,\ 0.039,\ 0.015,\ 1] \).

Had the grasp missed, the model should predict \( \Delta p_{obj} \approx 0 \), and the gap
\( d_{eo} \) would grow from 1 cm to 3 cm while \( z_{obj} \) stayed at 2 cm. Those two
outcomes are what section 4 is about.

## 3. Ensembles, and measuring what the model does not know

Train five copies of the same network, each from a different random initialisation and on a
different bootstrap resample of the data. Where training data was plentiful, the five agree.
Where there was none, nothing constrains them and they diverge. The spread
\( \sigma_{ens} \) is therefore an estimate of **epistemic** uncertainty — ignorance, the
reducible kind — as opposed to aleatoric noise.

Concretely, for a lift after a well-centred grasp the five members might predict
\( \Delta z_{obj} \in \{0.020, 0.021, 0.019, 0.020, 0.022\} \): mean 2.0 cm, spread 0.1 cm,
confident. For a grasp 2.5 cm off-centre, which never occurred in the human data:
\( \{0.020, 0.000, 0.017, 0.001, 0.000\} \): mean 0.8 cm, spread 0.9 cm. The mean is
meaningless; the spread is the useful output. A planner that is told about that spread can
refuse to rely on it.

## 4. Model exploitation, and why I film failures on purpose

A planner does not sample the model, it **optimises against** it. Anywhere the model is
wrong in an optimistic direction, the optimiser will find that spot and build its plan on
it. This is **model exploitation**, and it is the standard failure mode of model-based
control.

A model trained only on successful demonstrations has never observed fingers closing beside
a block. Asked what happens, it extrapolates from the only thing it has seen — closing on a
block, which lifts it — and happily predicts a lift. The planner then discovers that
"grasp 2 cm to the left" scores as well as a real grasp and is sometimes cheaper to reach.

Demonstration datasets are almost always success-only, because a demonstration is by
definition of someone succeeding. With a phone and ten seconds, filming a miss is as cheap
as filming a success. So I record 15 clips where I close beside the block, drop it in
mid-air, put it down short of the target, or shove it with an open hand. That puts real data
exactly where the planner wants to cheat, and the ensemble spread covers what is left.

Experiment E6 tests this directly by pretraining with and without those 15 clips.

## 5. Compounding error, and training on the model's own rollouts

A model that is 1 mm wrong per step is 2 cm wrong after 20 steps, because its own output
becomes its next input and the errors accumulate along a trajectory the model was never
trained on. Training on single-step transitions optimises the wrong thing.

The fix is to unroll the model against itself during training and penalise the whole
trajectory:

$$\mathcal{L} = \sum_{k=1}^{5}\Big(\big\lVert \Delta\hat p^{(k)}_{obj} - \Delta p^{(k)}_{obj}\big\rVert^2
+ w_a\,\operatorname{BCE}\big(\hat c^{(k)}, c^{(k)}\big)\Big),\qquad w_a = 0.5$$

where the input state at step \( k \) is built from the model's own predictions at all steps
before it, through the update equations of §2. The model is forced to see its own mistakes
and to correct back toward the truth.

## 6. Transferring between bodies: pretrain, fine-tune, and one extra input bit

Some of the physics is shared between my hand and the Panda: a held object follows the
gripper, a released object falls, an untouched object stays. Some is not: a human pinch
tolerates a centimetre of misalignment that a parallel jaw does not.

The model gets an **embodiment flag** \( e \in \{0 \text{ human}, 1 \text{ robot}\} \)
alongside the state and action. Shared structure is learned once from both sources; the
body-specific differences have somewhere to live. Then:

1. **Pretrain** on all the human data (\( e = 0 \)). This is cheap and plentiful.
2. **Fine-tune** on \( N \) robot episodes (\( e = 1 \)) at a third of the learning rate,
   mixing 20% human batches to stop it forgetting.

The quantity of interest is not the final accuracy, it is the **shape of the curve**: how
does the error depend on \( N \), and does pretraining shift that curve left? "How many robot
episodes do 45 phone clips buy me?" is a number, not an anecdote. That is experiment E2.

## 7. Getting metric 3-D out of one phone camera

No depth sensor, no learned depth network. Everything is classical geometry.

**Pinhole projection.** A 3-D point in the camera frame lands on a pixel:

$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim K\begin{bmatrix}X\\Y\\Z\end{bmatrix},\qquad
K = \begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix}$$

Calibration finds \( K \) and the lens distortion that minimise the reprojection error of a
checkerboard's known corners over many views. For a 1080p phone expect
\( f_x \approx 1450 \), \( c_x \approx 960 \), \( c_y \approx 540 \), with an RMS below
0.5 px. This is also why video stabilisation must be off: it changes the effective intrinsics
from frame to frame, and then there is no single \( K \) to find.

**A pixel gives a ray, not a point.** One image cannot tell you depth. But if you know the
3-D *plane* the point lies on, the ray meets it exactly once. A printed ArUco board defines
that plane: its 16 corners have known metric positions, so pose estimation gives
\( R_{CT}, t_{CT} \) with \( X_C = R_{CT}X_T + t_{CT} \). With normal \( n = R_{CT}e_z \),
offset \( c = n\cdot t_{CT} \) and ray \( r = K^{-1}[u,v,1]^\top \):

$$Z_{plane}(u,v) = \frac{c}{n\cdot r},\qquad X_C = Z_{plane}\,r,\qquad X_T = R_{CT}^\top(X_C - t_{CT})$$

For a point at a known height \( h \) above the table — a block centre sits at \( b/2 \) —
intersect with the plane \( z_T = h \) instead. A camera 55 cm above the table at 45° puts
the board centre at \( Z \approx 0.78 \) m.

**PnP for the hand.** Given 3-D points \( X_i \) in an object's own frame and their pixels
\( \ell_i \), find the pose minimising reprojection error:

$$\min_{R,t}\sum_i \big\lVert \ell_i - \pi(K(RX_i + t))\big\rVert^2,\qquad \pi([x,y,z]) = [x/z,\ y/z]$$

MediaPipe's hand tracker supplies both halves: 21 image landmarks and 21 "world" landmarks in
metres. The world landmarks have the right shape but are scaled to an *average* hand, so PnP
returns the correct pose up to one unknown constant \( k \).

**Fixing the scale with gravity and a flat table.** For two seconds at the start and end of
every clip my hand lies flat on the table, so its true depth is known from the plane
equation. One constant removes the ambiguity for the whole clip:

$$k = \operatorname{median}_{t\in rest}\frac{Z_{plane}(u_{pinch}(t))}{Z^{pnp}_{pinch}(t)},\qquad
p_C = k\,(R_h\,\tfrac12(X_4+X_8) + t_h),\qquad p_{ee} = R_{CT}^\top(p_C - t_{CT})$$

Example: PnP says 0.74 m at rest, the plane says 0.79 m, so \( k = 1.068 \). Mid-carry PnP
gives 0.66 m, hence a true 0.705 m, hence \( p_{ee} = (0.18, 0.12, 0.10) \) — 10 cm up.

**Open or closed.** The pinch aperture is the thumb-to-index distance divided by a length
that scales the same way with distance, so the ratio is distance-invariant:
\( a(t) = \lVert \ell_4 - \ell_8\rVert / \lVert \ell_0 - \ell_9\rVert \).
Thresholds are set per clip from its own distribution, with hysteresis so it cannot chatter:
with \( a_{lo} \) the 10th percentile and \( a_{hi} \) the 90th, close below
\( a_{lo} + 0.3(a_{hi}-a_{lo}) \) and open above \( a_{lo} + 0.6(a_{hi}-a_{lo}) \).

**The block.** A saturated colour makes the block a solid blob under an HSV threshold; its
centroid gives the pixel, and the plane intersection gives the metric position. Two caveats
with clean answers. First, a visible blob's centroid is biased toward the camera-facing face
of the cube — but the bias is constant at the 1–2 cm level and **cancels in differences**,
which is all the state uses. Second, once the block is lifted it is no longer on a known
plane; then it is at roughly the pinch point's depth, so
\( X_C = Z^{C}_{ee}\cdot K^{-1}[u_o,1]^\top \) recovers it.

## 8. Two-anchor retargeting: two points fix a plane transform exactly

A 2-D similarity transform — rotation, uniform scale, translation — has four degrees of
freedom. Two point correspondences give four equations. So *two* anchors determine it
uniquely, with nothing left to fit and nothing to tune.

The two anchors write themselves: **where I grasped must map to where the block is**, and
**where I released must map to where the target is**. Everything between them is carried
along, so the robot's path keeps the shape of my motion while starting and ending in the
right places for a layout I never filmed.

Writing 2-D points as complex numbers makes it one division:

$$T(x) = \alpha x + \beta,\qquad \alpha = \frac{P - G}{x_r - x_g},\qquad \beta = G - \alpha x_g$$

$$z^*(t) = z_{grasp} + \kappa\,(z_h(t) - z_h(t_g)) + h_{tgt}\,s(t),\qquad
\kappa = \operatorname{clip}(|\alpha|, 0.5, 1.5)$$

with \( G = b^{xy} \) the block, \( P = q^{xy} \) the target, and \( s(t) \) ramping 0 to 1
between grasp and release. Height is scaled by \( \kappa \) so a stretched path also lifts
higher, clipped so it cannot become absurd. If \( |x_r - x_g| < 5 \) cm the two anchors are
effectively one point and the problem is degenerate, so \( \alpha = 1 \).

**Worked example.** \( x_g = (0.10, 0.05) \), \( x_r = (0.30, 0.15) \), \( G = (0.50, 0.10) \),
\( P = (0.60, -0.12) \) give \( \alpha = -0.04 - 1.08i \) (scale 1.08, rotation −92°) and
\( \beta = 0.45 + 0.21i \). Check: \( T(x_r) = 0.60 - 0.12i = P \).

The baseline to beat is one fixed transform applied to every clip, ignoring where the objects
actually are.

## 9. Following a path: damped least-squares inverse kinematics

The Jacobian \( J \) (6×7 for a Panda) maps joint velocities to end-effector velocity.
Inverting it to turn "move the gripper 2 cm" into joint motion blows up near singular poses,
where a small Cartesian motion demands an enormous joint motion. Damping bounds it:

$$\Delta q = J^\top (JJ^\top + \lambda^2 I)^{-1} e,\qquad \lambda = 0.05$$

$$q_{cmd} = q + \Delta q + (I - J^{+}J)\,k_n (q_{home} - q)$$

The second term exploits the arm's seventh, redundant joint to drift back toward a
comfortable posture **without moving the end-effector at all**, since it acts only in the
null space of \( J \).

A one-dimensional illustration of why the damping matters. With \( J = 0.5 \) and
\( e = 0.02 \): the plain inverse gives \( \Delta q = 0.040 \), damped gives
\( 0.5\cdot0.02/(0.25 + 0.0025) = 0.0396 \) — essentially identical. Near a singularity,
\( J = 0.01 \): the plain inverse demands **2.0 rad**, a violent motion, while damped gives
\( 0.01\cdot0.02/(0.0001 + 0.0025) = 0.077 \) rad. The damping costs nothing when the arm is
well-conditioned and saves it when it is not.

## 10. Choosing actions: CEM inside a shrinking horizon

**Cross-entropy method.** Keep a Gaussian over the decision variables. Sample candidates,
score them all, keep the best few ("elites"), refit the Gaussian to the elites, repeat. It
needs no gradients, which matters because the score comes from rolling a neural network
forward sixty steps, and it copes with integer decision variables by rounding.

$$\theta_i \sim \mathcal{N}(\mu, \operatorname{diag}\sigma^2),\qquad
\mu \leftarrow \operatorname{mean}(\theta_{elite}),\qquad
\sigma \leftarrow \max(\operatorname{std}(\theta_{elite}),\ \sigma_{min})$$

In one dimension, with the true best grasp offset at +1.0 cm: iteration 1 starts at
\( \mu = 0, \sigma = 1.5 \) and its elites average +0.8; iteration 2 has
\( \mu = 0.8, \sigma = 0.6 \) and elites average +0.97; iteration 3 sits at
\( \mu = 0.97, \sigma = 0.3 \). Converged, in three rounds of pure sampling.

**Searching around my path rather than from scratch.** The decision variables are not raw
actions. They are a **residual** added to the retargeted path from §8: 8 knots × 3 axes,
linearly interpolated with the first knot pinned at zero, plus 2 integer shifts of the close
and open times. 26 numbers instead of a free trajectory. My clip supplies the shape of a
sensible pick-and-place, and the optimiser only has to correct it. Experiment E4 measures
what that is worth by replacing the prior with a straight line.

**Pessimism in the cost.** Of two plans with the same expected outcome, prefer the one the
model is confident about. That is one term:

$$J(\theta) = \underbrace{\frac{1}{M}\sum_m D_m(\theta)}_{\text{expected miss}}
+ \beta\,\underbrace{\operatorname{std}_m D_m(\theta)}_{\text{model doubt}}
+ w_s\sum_k\lVert r_k - r_{k-1}\rVert^2$$

$$D_m = \frac{\lVert \hat d^{\,xy}_{to}\rVert^2}{(2\,\text{cm})^2}
+ \frac{(\hat z_{obj} - h_{tgt} - b/2)^2}{(1\,\text{cm})^2},\qquad \beta = 1,\ w_s = 10$$

The \( \beta \) term is what turns the ensemble spread of §3 into behaviour, and it is the
direct countermeasure to the model exploitation of §4.

**Shrinking horizon.** Plan all the way to the end of the task, execute only the first
5 steps (0.5 s), look at what actually happened, and re-plan from there with the previous
solution as a warm start. The horizon shrinks as the task progresses. Re-planning is what
makes model error survivable: a block that slipped, or that someone pushed, is simply the
new starting state.

## 11. Glossary

| Term | Meaning |
|---|---|
| PnP | Find an object's pose from known 3-D points and their pixels |
| DLS IK | Damped least-squares inverse kinematics: joint motion for a desired gripper motion, stable near singularities |
| Ensemble | Several models trained differently; their disagreement estimates what the model does not know |
| Epistemic uncertainty | Uncertainty from lack of data (reducible), as opposed to aleatoric noise (irreducible) |
| Model exploitation | An optimiser finding and using the model's errors |
| CEM | Sample, keep the best, refit, repeat |
| MPC | Plan ahead, execute a little, re-plan |
| Pretrain → fine-tune | Learn general structure on big cheap data, adapt on small expensive data |
| AUROC | Chance a classifier ranks a random positive above a random negative; 0.5 is guessing |
| VLA | Vision-language-action model: a large policy mapping images and an instruction to actions |

---
---

# Part II — The brief, and how this answers it

## 12. What Humanoid asked for

The intern challenge, in its own terms:

> The goal of the challenge is to use real data collected by an applicant to drive a robotic
> manipulator in a simple simulation environment (e.g. Libero). The applicant is welcome to
> use a simple phone to record a small manipulation dataset and use it creatively showcasing
> their knowledge with VLA and/or World Models.

The illustration given is an egocentric hand-manipulation clip on the left, and a Panda arm
in Libero driven by a SmolVLA-based policy on the right. Among the suggested directions are
*"explore creative retargeting strategies, e.g. adapt your data to challenging embodiments"*
and *"use world modelling to showcase video/state prediction, less focusing on policy
performance"*. The binding constraint is stated plainly: **"you use the data that you
personally collected"**. Submissions are judged on creativity under that constraint, on
policy performance in simulation and/or quality of world-model predictions, and on
*"implementation simplicity and clear presentation of results without AI slop"*.

## 13. How this project maps onto it

| What the brief asks | What this project does |
|---|---|
| Real data collected by the applicant | 45 clips of my own hand, filmed on my phone, on my table. Protocol in `FILMING_GUIDE.md` |
| …that **drives** a manipulator in simulation | The world model the Panda plans with is pretrained on those clips, and the planner's search is seeded by a path retargeted from them. Both roles are measured, by E2 and E4 |
| A simple simulation environment | MuJoCo with the Menagerie Franka Panda — the same arm as the example, see §15 |
| VLA **and/or** World Models | World models. An ensemble dynamics model, trained from my video |
| Creative retargeting (suggested) | Two-anchor similarity retargeting, §8: exactly determined, no fitting, and measured against a fixed-transform baseline in E1 |
| World modelling for state prediction (suggested) | E2 measures 10-step block-position error on held-out robot episodes; a figure rolls the model forward on a held-out clip of my own video |
| Performance and/or WM prediction quality | Both. E3 is a control success table, E2 is a prediction-error curve |
| Implementation simplicity | ~35k parameters per ensemble member, five of them. Trains in about a minute on a laptop CPU. No pretrained checkpoints, no GPU, no Colab |
| No AI slop | Every number in the README comes from a file in `results/`. Figures are generated from the CSVs, never typed in |

**Where the data is load-bearing.** This is the constraint that matters, so it is worth being
precise. Delete my clips and three separate things stop working: the world model has no
pretraining corpus and E2 collapses to the sim-only curve; the planner loses its prior and
falls back to the straight-line baseline of E4; and the failure-awareness measured in E6
disappears entirely. The clips are not a demonstration set that seeds a policy and is then
discarded — they are where the model's physics comes from.

## 14. Why a world model rather than a VLA

The brief's own example is fine-tuning a VLA such as SmolVLA or π₀ on retargeted
demonstrations. I considered it and rejected it, for one disqualifying reason and one
interesting one.

The disqualifying reason: SmolVLA is 450M parameters. Fine-tuning it needs CUDA. My
constraint was a laptop CPU, and a submission I cannot iterate on in under a minute is a
submission I cannot debug.

The interesting reason: almost all work that uses human video uses it for **trajectories** —
what to do. Retarget the hand path, imitate it. The video teaches *intent*. But a video of a
hand moving a block also contains something else, which is usually thrown away: it contains
**consequences**. The block came with the hand, so it was held. The block stayed put, so the
grasp missed. The block fell, so it was released. That is physics, it is embodiment-agnostic
in the state space of §2, and it is exactly what a world model needs. Using my clips to teach
*what happens* rather than *what to do* is the part of this submission I would defend first.

Filming failures on purpose follows directly. If the video teaches consequences, then clips
where the consequence is "nothing happened" are as informative as clips where it worked — and
they are the ones no demonstration dataset contains.

## 15. Options I considered, and why I picked this one

Scored on: Laptop-feasible (does it train on a CPU in under an hour?), Novelty (against a
typical submission), Data role (how central my recordings are), Risk (chance of not working
in a week).

| # | Idea | Laptop | Novelty | Data role | Risk | Verdict |
|---|---|---|---|---|---|---|
| A | Fine-tune a VLA (SmolVLA / π₀) on retargeted demos | ✗ (450M+ params, needs CUDA) | low (it is the brief's own example) | medium | high | no |
| B | Pixel video world model (diffusion, Cosmos) | ✗ | medium | medium | very high | no |
| C | DINO-WM: plan in DINOv2 features of sim images | slow (encoding plus training) | medium | indirect: my data only seeds the search | medium | considered, replaced |
| D | Latent actions from video (LAPA / Genie style) | ✗ (needs thousands of clips) | medium | high | very high | no |
| E | Paint a robot over my hand in the video (Phantom style) | ✗ (segmentation plus inpainting models) | low (published) | high | high | no |
| F | Point-track world model (ATM, Track2Act) | ✗ (CoTracker needs a GPU) | medium | high | high | no |
| G | Residual RL (PPO) on top of the retargeted policy | ✓ | medium | medium | medium (RL tuning) | stretch |
| **H** | **Object-centric world model pretrained on my hand video including failures, adapted to the robot with few sim episodes, used for MPC seeded by my retargeted path** | **✓ (trains in ~1 min on CPU)** | **high** | **central: the model's physics comes from my video** | **low–medium** | **chosen** |

What is honest to say about novelty: every component exists. Probabilistic ensembles are
PETS (Chua et al., 2018); CEM-MPC is standard; object-centric state representations are old.
The contribution is the combination — a human-pretrained, failure-aware, embodiment-shared
dynamics model — and the measurement of how much it actually helps.

## 16. Three deliberate deviations from the brief's example

**Plain MuJoCo instead of Libero.** The brief says "e.g. Libero". Libero brings robosuite and
a large asset download, and is awkward to install on a CPU-only machine and on Windows. It
runs on MuJoCo, and the arm in the brief's own screenshot is a Franka Panda. So I use MuJoCo
directly with the Menagerie Panda: the same arm and the same physics engine, with a scene I
wrote and can explain in full, installable in one command on three operating systems. The
cost is that I do not inherit Libero's task suite, which this project does not need.

**A static third-person camera instead of an egocentric one.** The brief's example frame is
egocentric. I film from a tripod facing me. The reason is §7: a fixed camera plus a printed
board gives a fixed intrinsic matrix and a known metric plane, and those two things are what
turn pixels into centimetres. From a moving egocentric camera, every measurement would be
expressed in an unknown moving frame, and the state of §2 could not be built without
recovering camera motion first. The scene is also framed to mirror the simulator's camera,
which makes the side-by-side comparison videos honest.

**A laptop CPU instead of a Colab GPU.** The brief notes that many ideas fit on a Colab GPU
notebook. This one does not need one. Full world-model training is about a minute, and
generating 300 robot episodes is one to two minutes. Everything in the README can be
reproduced on the machine of whoever is reading it.

## 17. What would count as a result

Stated before running anything, so the answer cannot be chosen after the fact.

**Positive:** the pretrain-then-fine-tune curve reaches the sim-only error at
\( N \approx 25 \) with noticeably fewer robot episodes — the E2 curve shifts left — and in
E3, "ours at 25 episodes" approaches "sim-only at 300 episodes".

**Negative is still a result.** If human pretraining does not help, the embodiment-gap
heatmap is there to show *why* — the usual suspect being grasp tolerance, where a human pinch
forgives an offset that the Panda's jaws do not. That goes in the README as a finding, with
the figure next to it. A measured null result that explains itself is worth more than a demo
that only ever ran once.

---
---

# Part III — Implementation and architecture

## 18. Pipeline

```
 PHONE (45 clips: 30 success + 15 deliberate failures)
   │
   ├─[1 calib]  K
   ├─[2 ArUco]  table frame (metric)
   ├─[3 hand]   pinch point p_ee(t) via MediaPipe + PnP, gripper g(t)
   ├─[4 block]  block position p_obj(t) via colour mask + geometry
   └─[5 state]  shared state s_t ∈ R⁸, action a_t ∈ R⁴   ──► HUMAN transitions (~9k)
                                                                  │
 SIM (MuJoCo + Menagerie Panda, my own scene and IK)              │
   ├─[6 retarget]  two-anchor path from my clips ─┐               │
   ├─[7 IK tracker]  follow any EE path           │               │
   └─[8 generate]  N robot episodes (N = 0…300) ◄─┘──► ROBOT transitions
                                                                  │
                                                                  ▼
                        [9 WORLD MODEL] ensemble of 5 MLPs:  (s, a, embodiment) → Δp_obj, attach
                            pretrain on human  →  fine-tune on N robot episodes
                                                                  │
                                                                  ▼
                        [10 MPC] CEM over residuals around my retargeted path, re-plan every 0.5 s
                                                                  │
                                                                  ▼
                                                     Panda places the block (plate / pad / box top)
```

## 19. Platforms and environment

Runs on **macOS, Windows and Linux**, CPU only. Developed and tested on macOS on Apple
Silicon; the other two are kept working by using nothing platform-specific.

- Python 3.11, managed by `uv`. `uv sync --extra dev` downloads the interpreter and every
  dependency; nothing else is installed system-wide.
- FFmpeg is not a system requirement: OpenCV and `imageio-ffmpeg` each bundle their own.
- On Linux, torch resolves to the `+cpu` build so the install does not drag in the CUDA
  stack. On macOS and Windows the PyPI wheel is already CPU-only.
- The Panda comes from MuJoCo Menagerie via the `robot_descriptions` package, which caches a
  clone in `~/.cache/robot_descriptions` on first use.
- **The one platform difference:** the interactive MuJoCo viewer needs `mjpython` on macOS
  and plain `python` on Windows and Linux. Offscreen rendering with `mujoco.Renderer`, which
  every script in the repo uses, behaves identically everywhere.
- Code rules that keep it that way: `pathlib` for every path, utf-8 on every text read and
  write, no shelling out to platform tools, and any genuine difference branches on
  `sys.platform` with a note in the docstring.

## 20. Repository layout and conventions

```
configs/default.yaml   every parameter, grouped by module, one comment per key
src/palm_prior/
  state.py             the 8-D state of §2 — the only place it is built or decoded
  utils.py             config loading, seeding, paths, video writing, timing
  perception/          calibration, ArUco and the table frame, hand, block tracking
  human/               clip to transitions
  retarget/            two-anchor transform, naive baseline, plan building
  sim/                 MuJoCo scene, environment, IK
  wm/                  ensemble model and training
  plan/                CEM and MPC
  viz/                 figures and debug overlays
scripts/               one entry point per pipeline stage
assets/scene.xml       MJCF: Panda, floor, block, plate, pad, box
tests/                 pytest for state, geometry, retarget, IK, CEM
results/               csv, png, mp4 — committed; data/ and checkpoints/ are not
```

Conventions: every parameter lives in `configs/default.yaml` and is overridable as
`key=value` on any script's command line. Units are metres, radians and seconds. Variables
carry their frame in the name (`p_T`, `R_CT`). Every script takes `seed=`, skips work whose
output already exists unless given `overwrite=true`, saves a debug plot or video, and prints
its runtime. Docstrings state shapes and frames, and shapes are asserted at function
boundaries.

## 21. Perception: phone to human state

### Module 1 — Camera calibration
**In:** `calib.mp4`, a 9×6 inner-corner checkerboard with 25 mm squares.
**Out:** `K` (3×3), `dist` (5,), RMS → `data/calib/camera.yaml`.
About 40 well-spread frames are selected automatically. Theory in §7.
**Acceptance:** RMS < 0.5 px.
**Can go wrong:** stabilisation left on, so `K` changes frame to frame and no single
solution exists. The guide turns it off and the test clip checks for it.

### Module 2 — Table frame from ArUco
**In:** frame, `K`, board spec. **Out:** \( R_{CT}, t_{CT} \).
Board: `DICT_4X4_50`, 2×2 grid, 60 mm markers, 15 mm gaps, so 135 × 135 mm overall. The
phone is on a tripod, so the pose is the median over frames, with rotations averaged through
`scipy`'s rotation mean.
**Note on the frame.** OpenCV's `GridBoard` numbers markers from the top of the generated
image, which makes its own z axis point *into* the printed page. The board object points are
mirrored in y — a 180° rotation about x, so the printed picture is untouched — to give T with
its origin at the ORIGIN corner, x right along the bottom edge, y up the left edge and
**z out of the table**, as §2 requires.
**Can go wrong:** the arm crossing the board, or glare on it.

### Module 3 — Hand pinch point
**In:** frame, `K`. **Out:** \( p_{ee}(t) \) in T, aperture \( a(t) \), gripper \( g(t) \), `valid`.
MediaPipe HandLandmarker (Tasks API, VIDEO mode; the `.task` model downloads into
`third_party/`), then PnP, then the rest-frame scale \( k \) from the first and last 2 s.
Equations and thresholds in §7. Gaps of up to 10 frames are interpolated; the result is
smoothed with Savitzky-Golay (window 9, order 2).
**Can go wrong:** fingertips occluded at the moment of grasp.

### Module 4 — Block position
**In:** frame, HSV range fitted from `empty.mp4`, \( p_{ee} \), `K`, table pose.
**Out:** \( p_{obj}(t) \) in T, `valid_obj`.
1. HSV threshold, largest blob, centroid \( u_o \) and area \( A \).
2. **On the table** (gripper open, or the pinch point more than 3 cm above the block's last
   position): intersect the ray with \( z_T = b/2 \).
3. **In the hand** (closed and lifted): \( X_C = Z^{C}_{ee}\cdot K^{-1}[u_o,1]^\top \).
4. **Occluded** (\( A \) below 30% of the rest-time area): invalid; interpolate up to
   10 frames.

Worked example: centroid (1012, 640), ray meets \( z_T = 0.02 \) at
\( p_{obj} = (0.30, 0.15, 0.02) \); later, held at \( Z_C = 0.70 \), at
\( p_{obj} = (0.24, 0.13, 0.11) \).

### Module 5 — Human transitions
**In:** \( p_{ee}, g, p_{obj} \) per frame, plus the target dot from `recording_log.csv` and
`dots.yaml`, and the goal height. **Out:** `data/human/transitions.npz`.
1. Time stretch \( \tau = 2 \) (people move about twice as fast as this robot), then sample
   at 10 Hz: \( t_k = k \cdot 0.1 / \tau \) in video seconds.
2. Build \( s_k \) per §2, and \( a_k = [p_{ee}(k{+}1) - p_{ee}(k),\ g(k{+}1)] \).
3. Labels: \( \Delta p_{obj,k} = p_{obj}(k{+}1) - p_{obj}(k) \), and
   **attached** \( = g = 1 \wedge z_{obj} > b/2 + 0.01 \). **The same rule is used for the robot.**
4. Keep `is_failure_clip` and `clip_id` for the ablation and the train/held-out split.

**Size:** 45 clips × ~10 s × τ 2 × 10 Hz ≈ **9k transitions**.
**QC** → `results/extraction_report.csv`: valid fractions for hand and block, number of grasp
events, reprojection error, the disagreement between the vision-measured block start and
`dots.yaml`, and a pass flag. Success, F1, F2 and F3 clips must contain exactly one grasp;
F4 clips exactly zero.

## 22. Simulation

### Module 6 — Two-anchor retargeting
Theory and equations in §8. Defaults: 8 residual knots with knot 0 pinned at 0, degenerate
threshold 5 cm, \( \kappa \) clipped to [0.5, 1.5].
**Interface:** `build_plan(clip, block_W, tgt_W, h_tgt, residual_knots (8,3) or None,
timing_shift (2,) or None, cfg, ee_start_W)` → `p_star (K,3)` at 10 Hz, `grip (K,)`,
`k_close`, `k_open`. The residual is a linear interpolation of the knots.
**Baseline:** `naive` applies one fixed transform to every clip and takes no object
positions at all.

### Module 7 — Scene and IK tracker
**Scene** (`assets/scene.xml`): Menagerie `franka_emika_panda`; floor at z = 0; a 4 cm block
with a free joint, 50 g, friction 1.5; a plate as a static thin cylinder of the measured real
radius; a static 10 × 10 × 0.5 cm pad; a static 8 cm cube. Physics: `timestep=0.002`,
`integrator="implicitfast"`, `cone="elliptic"`, `impratio="10"` — the last two reduce grasp
slip. A `tcp` site sits between the fingertips, about 0.1034 m below the hand body; the
Menagerie Panda defines no sites of its own, so this is added here and checked visually by
rendering a small sphere at it. Layout sampling: block and target in x ∈ [0.40, 0.65],
y ∈ [−0.20, 0.20], at least 15 cm apart and not overlapping.

**Tracker:** DLS IK as in §9, \( \lambda = 0.05 \), with the orientation error keeping the
gripper pointing down at a fixed yaw. Each 10 Hz control step is 50 physics steps, with IK
recomputed every 10. The null-space posture gain \( k_n \) is set to 0.1; it is the one
constant not pinned down by the theory.

**Gripper:** Menagerie exposes 8 actuators, `actuator1`…`actuator8`; number 8 is the gripper,
with `ctrlrange` 0 to 255, where 255 is open and 0 is closed.

**Success:** block centre within the target radius in xy, \( |z_{obj} - (h_{tgt} + b/2)| < 1 \) cm,
gripper open, and block speed below 1 cm/s after 1 s of settling.

### Module 8 — Robot data generation
Per episode: random layout (seeds from 1000 for training, 900–999 held out; **seeds 0–49 are
reserved for control evaluation and never trained on**), random goal, random human clip,
retarget, add smooth noise (σ = 1.5 cm on the 8 knots), and with probability 0.2 inject a
failure — either an early release or closing 2–3 cm off-centre. Track it, and log state,
action, \( \Delta p_{obj} \) and attach at 10 Hz using exactly the Module 5 rule. No
rendering, so it is fast.
**Size:** 300 episodes × ~80 steps ≈ 24k transitions in **1–2 minutes** on a laptop CPU.
**Subsets:** nested, fixed-seed N ∈ {0, 5, 10, 25, 50, 100, 300}.

## 23. World model

### Module 9 — Ensemble dynamics model
**In:** \( x = [\bar s,\ \bar a,\ e] \), standardised state and action plus the embodiment
flag; dimension 8 + 4 + 1 = 13. Normalisation statistics are fitted on the training set and
stored inside the checkpoint.
**Out:** \( \Delta\hat p_{obj} \in \mathbb{R}^3 \) (standardised, then de-normalised) and one
attach logit.
**Architecture:** 5 independent MLPs, 13 → 128 → 128 → 128 → 4, SiLU, each with its own
random initialisation and bootstrap resample. About 35k parameters each.
**Loss:** the 5-step self-rollout of §5, with \( w_a = 0.5 \), where each step's input state
is rebuilt from the model's own predictions through the §2 update equations.
**Optimisation:** Adam, lr 1e-3, batch 256, 3k pretraining steps plus 1.5k fine-tuning steps
per member. Fine-tuning uses lr × 0.3 and 20% human batches. **About a minute in total on CPU.**

**Three training variants, compared in E2:**
1. *Sim-only:* train on N robot episodes.
2. *Human-pretrain → fine-tune (ours):* all human data, then N robot episodes.
3. *Co-train:* human and robot together from the start, robot batches upweighted.

Held out: 5 human clips, at least one of them a failure, and robot episodes from seeds
900–999.

## 24. Planning

### Module 10 — Shrinking-horizon MPC with CEM
Theory, cost and equations in §10.
**Decision variables** \( \theta \in \mathbb{R}^{26} \): 8 knots × 3 axes of xyz residual on
the prior path, linearly interpolated with the first knot fixed at 0, plus 2 integer shifts
of the close and open times, limited to ±5 steps.
**CEM settings:** 256 samples, 25 elites, 4 iterations; initial σ of 1.5 cm on the knots and
2 steps on the timings.
**Cost:** \( \beta = 1 \), \( w_s = 10 \), with the miss normalised by 2 cm in xy and 1 cm in
height.
**Execution:** the first 5 actions, then re-plan with μ shifted by 5 steps.
**Model options:** a world-model checkpoint, or `true_sim`, which rolls candidates out in a
cloned `MjData` with fewer samples (32) as an upper bound.
**Prior options:** `human` (the retargeted clip) or `isotropic` (a straight-line scripted path
with the same timing).
**Cost per re-plan:** 4 iterations × 256 candidates × ~60 steps × 5 tiny MLPs, batched in
torch on CPU, is about **50–100 ms**.

## 25. Experiments

Control evaluation uses **layout seeds 0–49** × 3 repeats; report mean ± standard deviation.
World-model prediction uses **held-out** robot episodes (seeds 900–999) and **held-out**
human clips (5 of them).

| ID | Question | Setup | Output |
|---|---|---|---|
| **E1** | Does object-centric retargeting beat naive? | open-loop replay, naive vs two-anchor | success % |
| **E2** ★ | **How much robot data do my clips save?** | 10-step block-position error on held-out sim vs N ∈ {0,5,10,25,50,100,300}, for sim-only / pretrain→FT / co-train | curve: error vs N, log x |
| **E3** ★ | Does it control better? | success on seeds 0–49: open-loop two-anchor; MPC+WM sim-only N=25; **MPC+WM pretrain→FT N=25**; MPC+WM human-only N=0 (zero-shot); MPC+WM sim-only N=300 (data-rich reference); MPC with the true simulator as the model, 32 samples (upper bound) | table |
| **E4** | Does my path help the search? | MPC with the human prior vs an isotropic prior, samples ∈ {16, 64, 256} | success vs samples |
| **E5** | Robustness | block pushed 5 cm at step 15; 2 cm noise on object positions; open loop vs MPC | success % |
| **E6** | Do failure clips matter? | pretrained with vs without the 15 failure clips: attach AUROC on robot failure states, plus E3 success | table |
| **E7** | One video, three goals | MPC on plate / pad / box-top | success % per goal |
| **Fig** | The embodiment gap, made visible | predicted P(lift succeeds) against grasp xy offset over ±3 cm: human-only WM, adapted WM, true sim | 3 heatmaps |
| **Fig** | The WM on my real video | roll the model on a held-out clip from the hand motion alone, and draw the predicted block onto the video frames | mp4 |

**Main table** — filled with measured numbers only:

| Method | plate | pad | box-top |
|---|---|---|---|
| Naive retarget, open loop | | | |
| Two-anchor retarget, open loop | | | |
| MPC + WM sim-only (25 ep) | | | |
| **MPC + WM human-pretrained (25 ep)** | | | |
| MPC + WM human-only (0 ep) | | | |
| MPC + WM sim-only (300 ep) | | | |
| MPC + true simulator (upper bound) | | | |

Criteria for a positive and for a negative result are in §17, and were written down before
anything was run.

## 26. Stretch, if the core is finished

- **Distill the MPC into a policy.** Collect (state, executed MPC action) pairs and train an
  MLP policy. Compare speed, roughly 0.05 ms against roughly 80 ms per step, and success.
- **Residual RL.** PPO on residual actions over the retargeted path, using world-model
  rollouts as extra imagined experience.
