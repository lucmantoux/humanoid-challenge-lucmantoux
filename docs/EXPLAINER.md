# palm-prior — Technical Explainer (Mac-native, novel version)

> **One line:** I teach a robot's *world model* how objects behave, using 45 phone clips of my own hand, **including clips where I fail on purpose**. Then I adapt it to a simulated Panda with very little robot data, and the robot plans with it, starting its search from a path retargeted from my hand.
>
> **The claim I test:** *30 successful + 15 deliberately failed clips of my hand reduce how much robot data a world model needs, and make planning with it safer.*

Everything here trains and runs on a **MacBook (Apple Silicon), CPU only**. No GPU, no Colab, no CUDA.

Each module follows the same layout: **Purpose → Inputs → Outputs → Theory → Equations → Example → What can go wrong**.

---

## 0. Options I considered (and why I picked this one)

Scores: Mac-feasible (does it train on a laptop in < 1 h?), Novelty (vs typical submissions), Data role (how central my recordings are), Risk (chance of not working in 1 week).

| # | Idea | Mac | Novelty | Data role | Risk | Verdict |
|---|---|---|---|---|---|---|
| A | Fine-tune a VLA (SmolVLA / π0) on retargeted demos | ✗ (450M+ params, needs CUDA) | low (it's the brief's own example) | medium | high | no |
| B | Pixel video world model (diffusion, Cosmos) | ✗ | medium | medium | very high | no |
| C | DINO-WM: plan in DINOv2 features of sim images | slow (encoding + training on MPS) | medium | indirect: my data only seeds the search | medium | previous plan, replaced |
| D | Latent actions from video (LAPA/Genie style) | ✗ (needs thousands of clips) | medium | high | very high | no |
| E | Paint a robot over my hand in the video (Phantom style) | ✗ (segmentation + inpainting models) | low (published) | high | high | no |
| F | Point-track world model (ATM, Track2Act) | ✗ (CoTracker needs a GPU) | medium | high | high | no |
| G | Residual RL (PPO) on top of the retargeted policy | ✓ | medium | medium | medium (RL tuning) | stretch |
| **H** | **Object-centric world model pretrained on my hand video (incl. failures), adapted to the robot with few sim episodes, used for MPC seeded by my retargeted path** | **✓ (trains in ~1 min on CPU)** | **high** | **central: the model's physics comes from my video** | **low–medium** | **chosen** |

**Why H is novel:**
1. **The world model learns physics from a human.** Most human-video work uses the video for *trajectories* (what to do). Here the video teaches *consequences* (what happens to the object when a hand does X).
2. **Deliberate failure clips.** Demonstrations only show success, but a world model must also know what failure looks like, or a planner will exploit its blind spots. I record misses, drops and off-target releases on purpose. That's cheap with a phone and almost never done.
3. **One state space for hand and gripper.** Everything is relative to the object, so hand data and robot data mix without any camera-to-robot alignment.
4. **A measurable claim, not just a demo.** "How many robot episodes do you save with 45 phone clips?" gives a curve, not an anecdote.

**What's honest to say:** the components exist (probabilistic ensembles, as in PETS, Chua et al. 2018; CEM-MPC; object-centric states). The contribution is the combination: a human-pretrained, failure-aware, embodiment-shared dynamics model, and the measurement of how much it helps.

---

## 1. Pipeline

```
 PHONE (45 clips: 30 success + 15 deliberate failures)
   │
   ├─[1 calib]  K
   ├─[2 ArUco]  table frame (metric)
   ├─[3 hand]   pinch point p_ee(t) via MediaPipe + PnP, gripper g(t)
   ├─[4 block]  block position p_obj(t) via colour mask + geometry
   └─[5 state]  shared state s_t ∈ R⁸, action a_t ∈ R⁴   ──► HUMAN transitions (~3–4k)
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

**Frames:** C = camera (OpenCV), T = table (ArUco corner, z up), W = sim world (z up, floor at 0). Units: metres, radians, seconds.

---

## 2. The shared state and action (the core design decision)

**State** $s \in \mathbb R^8$:

| Index | Symbol | Meaning |
|---|---|---|
| 0–2 | $d_{eo} = p_{ee} - p_{obj}$ | hand/gripper relative to the block (3) |
| 3–4 | $d_{to} = p_{tgt}^{xy} - p_{obj}^{xy}$ | target relative to the block, horizontal (2) |
| 5 | $z_{obj}$ | block centre height above the surface |
| 6 | $h_{tgt}$ | target top height (plate 0.015, pad 0.005, box 0.08 m) |
| 7 | $g$ | gripper / pinch closed (1) or open (0) |

**Action** $a \in \mathbb R^4$: $a = [\,p_{ee}(t{+}1) - p_{ee}(t),\ g(t{+}1)\,]$, i.e. where the hand/gripper goes next and whether it's closed. Control rate **10 Hz**.

**Theory — why this works across embodiments:**
- **Translation invariance.** Every horizontal quantity is a *difference*. Shift the whole scene by a vector c and s doesn't change. So the model never has to learn "where on the table", only "how things relate", and my kitchen table and the robot's floor don't need to be aligned at all.
- **Gravity is kept absolute** (z_obj, h_tgt). Falling and resting depend on height, so those stay absolute.
- **The hand is known, the object is the unknown.** The next end-effector position is given by the action itself. The only thing the world model must predict is **how the block moves**, $\Delta p_{obj}$, plus whether it's **attached** to the hand. That's the physics: does it follow the hand, stay put, fall, slide.

**State update given the predicted block motion $\Delta\hat p_{obj}$:**
$$d_{eo}' = d_{eo} + a_{xyz} - \Delta\hat p_{obj},\quad d_{to}' = d_{to} - \Delta\hat p^{xy}_{obj},\quad z_{obj}' = z_{obj} + \Delta\hat p^{z}_{obj},\quad h_{tgt}' = h_{tgt},\quad g' = a_g$$

**Example:** hand 1 cm above a 4 cm block, closed; action moves up 2 cm.
$s = [0,0,0.01,\ 0.15,0.05,\ 0.02,\ 0.015,\ 1]$, $a = [0,0,0.02,1]$. The WM predicts $\Delta p_{obj} = [0,0,0.019]$ (lifted with the hand). New $s' = [0,0,0.011,\ 0.15,0.05,\ 0.039,\ 0.015,\ 1]$. If instead the grasp had missed, the WM should predict $\Delta p_{obj} \approx 0$, and $d_{eo}$ grows to 3 cm.

---

## 3. Perception modules (phone → human state)

### Module 1 — Camera calibration
**Purpose:** convert pixels to rays.
**In:** `calib.mp4` (checkerboard 9×6, 25 mm). **Out:** `K` (3×3), `dist` (5,), RMS.
**Theory (pinhole camera):** a 3D point in the camera frame projects to pixel
$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim K\begin{bmatrix}X\\Y\\Z\end{bmatrix},\quad K = \begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix}$$
Calibration finds K (and the lens distortion) that minimizes the reprojection error of the known checkerboard corners over many views.
**Example:** 1080p phone: $f_x \approx 1450$, $c_x \approx 960$, $c_y \approx 540$. RMS < 0.5 px is good.
**Can go wrong:** stabilization on → K changes frame to frame → useless.

### Module 2 — Table frame from ArUco
**In:** frame, K, board spec. **Out:** $R_{CT}, t_{CT}$ with $X_C = R_{CT}X_T + t_{CT}$.
**Theory:** the board's 16 corners have known metric positions in T. PnP (Module 3) gives the camera pose relative to them. Phone on a tripod → median over frames.
**Table-plane depth of pixel (u, v):** with normal $n = R_{CT}e_z$, offset $c = n\cdot t_{CT}$, ray $r = K^{-1}[u,v,1]^\top$:
$$Z_{plane}(u,v) = \frac{c}{n\cdot r}$$
**Any table point from a pixel:** $X_C = Z_{plane}\,r$, then $X_T = R_{CT}^\top(X_C - t_{CT})$. For points at height h above the table (like the block centre), intersect the ray with the plane $z_T = h$ instead.
**Example:** camera 55 cm above the table at 45° → the board centre is at Z ≈ 0.78 m.

### Module 3 — Hand pinch point (MediaPipe + PnP, no depth network)
**In:** frame, K. **Out:** $p_{ee}(t)$ in T, aperture a(t), gripper g(t), `valid`.
**Theory (PnP, Perspective-n-Point):** given 3D points $X_i$ in an object's own frame and their 2D pixels $\ell_i$, find the rotation R and translation t that minimize the reprojection error
$$\min_{R,t}\sum_i \big\lVert \ell_i - \pi(K(RX_i + t))\big\rVert^2,\qquad \pi([x,y,z]) = [x/z,\ y/z]$$
MediaPipe HandLandmarker gives both: 21 image landmarks $\ell_i$, and 21 "world" landmarks $X_i$ in metres, with the right shape but scaled to an *average* hand. So PnP gives the hand pose, off by one constant factor k (your hand vs the average).
**Scale fix:** during the 2 s rest, the hand lies on the table, so the true depth is known from Module 2:
$$k = \operatorname{median}_{t\in rest}\frac{Z_{plane}(u_{pinch}(t))}{Z^{pnp}_{pinch}(t)},\qquad p_C = k\,(R_h\,\tfrac12(X_4+X_8) + t_h),\qquad p_{ee} = R_{CT}^\top(p_C - t_{CT})$$
**Gripper signal:** $a(t) = \lVert \ell_4 - \ell_8\rVert / \lVert \ell_0 - \ell_9\rVert$ (pinch size ÷ hand size, so distance cancels). Adaptive hysteresis per clip: $a_{lo}$ = 10th percentile, $a_{hi}$ = 90th percentile; close below $a_{lo} + 0.3(a_{hi}-a_{lo})$, open above $a_{lo} + 0.6(a_{hi}-a_{lo})$.
**Example:** PnP depth 0.74 m at rest, plane says 0.79 m → k = 1.068. Mid-carry PnP 0.66 → true 0.705 m → $p_{ee} = (0.18, 0.12, 0.10)$, i.e. 10 cm up.
**Can go wrong:** fingertips occluded at the grasp → interpolate gaps ≤ 10 frames. Jitter → Savitzky–Golay smoothing (window 9, order 2).

### Module 4 — Block position (colour + geometry)
**In:** frame, block colour HSV range (auto-fitted from the first frame), $p_{ee}$, K, table pose. **Out:** $p_{obj}(t)$ in T, `valid_obj`.
**Logic:**
1. HSV threshold → largest blob → centroid pixel $u_o$, area A.
2. **On the table** (gripper open, or the pinch point higher than 3 cm above the block's last position): intersect the ray through $u_o$ with the plane $z_T = b/2$, where b is the block size → metric block centre.
3. **In the hand** (closed and lifted): the block sits at roughly the pinch point's depth, so $X_C = Z^{C}_{ee}\cdot K^{-1}[u_o,1]^\top$ → transform to T.
4. Occluded (A < 30% of the rest-time area) → invalid; interpolate ≤ 10 frames.
**Theory:** the visible blob centroid is biased toward the camera-facing side of the block. This bias is constant at the 1–2 cm level and cancels in differences, which is all the state uses.
**Example:** blob centroid (1012, 640), ray hits $z_T = 0.02$ at $p_{obj} = (0.30, 0.15, 0.02)$; later, in hand at $Z_C = 0.70$ → $p_{obj} = (0.24, 0.13, 0.11)$.

### Module 5 — Human transitions
**In:** $p_{ee}, g, p_{obj}$ per frame, plus the target position from the recording log (dot coordinates) and the goal height. **Out:** `data/human/transitions.npz`.
**Logic:**
1. **Time stretch** τ = 2 (people move ~2× faster than our robot), then sample at 10 Hz: $t_k = k \cdot 0.1 / \tau$ in video seconds.
2. Build $s_k$ (Section 2), $a_k = [p_{ee}(k{+}1) - p_{ee}(k),\ g(k{+}1)]$.
3. Labels: $\Delta p_{obj,k} = p_{obj}(k{+}1) - p_{obj}(k)$, and **attached** $= g = 1 \wedge z_{obj} > b/2 + 0.01$. The *same rule* is used for the robot.
4. Keep the `is_failure_clip` and `clip_id` fields (needed for the ablation and the split).
**Size:** 45 clips × ~10 s × τ 2 × 10 Hz ≈ **9k transitions**.

---

## 4. Simulation modules (MuJoCo on Mac)

### Module 6 — Two-anchor retargeting (search prior)
**Purpose:** turn one of my clips into a robot path for a new layout, used as the **starting point** of the planner.
**In:** human $p_{ee}(t)$, grasp/release indices, sim block position b and target position q, residual θ. **Out:** target EE path $p^*(k)$ at 10 Hz plus a gripper schedule.
**Theory:** a 2D similarity transform (rotation + uniform scale + shift) has 4 degrees of freedom, and two point correspondences give exactly 4 equations. So forcing *grasp → block* and *release → target* fixes it uniquely, and the path keeps the shape of my motion.
**Equations** (2D points as complex numbers):
$$T(x) = \alpha x + \beta,\quad \alpha = \frac{P - G}{x_r - x_g},\quad \beta = G - \alpha x_g$$
$$z^*(t) = z_{grasp} + \kappa\,(z_h(t) - z_h(t_g)) + h_{tgt}\,s(t),\quad \kappa = \operatorname{clip}(|\alpha|, 0.5, 1.5)$$
Here $G = b^{xy}$ and $P = q^{xy}$; s(t) ramps 0 → 1 between grasp and release. If $|x_r - x_g| < 5$ cm, α = 1.
**Worked example:** $x_g = (0.10, 0.05)$, $x_r = (0.30, 0.15)$, $G = (0.50, 0.10)$, $P = (0.60, -0.12)$ → $\alpha = -0.04 - 1.08i$ (scale 1.08, rotation −92°), $\beta = 0.45 + 0.21i$. Check: $T(x_r) = 0.60 - 0.12i = P$ ✓.
**Naive baseline:** one fixed transform for all clips that ignores the object positions.

### Module 7 — Scene + IK tracker
**Scene** (`assets/scene.xml`): MuJoCo Menagerie `franka_emika_panda` (via the `robot_descriptions` package or a git clone), floor at z = 0, a 4 cm block (free joint, 50 g, friction 1.5), plate (static thin cylinder, r = measured real plate radius), pad (static 10×10×0.5 cm), box (static 8 cm cube). Physics options: `timestep=0.002`, `integrator="implicitfast"`, `cone="elliptic"`, `impratio="10"` (reduces slipping in grasps). Add a `tcp` site between the fingertips (≈ 0.1034 m below the hand body; check it visually). Layout sampling: block and target in x ∈ [0.40, 0.65], y ∈ [−0.20, 0.20], ≥ 15 cm apart.
**Tracker theory (damped least-squares IK):** the Jacobian J (6×7) maps joint velocities to TCP velocity. To move the TCP by the error e (3 position + 3 orientation, keeping the gripper pointing down):
$$\Delta q = J^\top (JJ^\top + \lambda^2 I)^{-1} e,\qquad \lambda = 0.05$$
$$q_{cmd} = q + \Delta q + (I - J^{+}J)\,k_n (q_{home} - q)$$
The damping λ keeps the motion bounded near singular poses. The second term uses the 7th "spare" joint to stay near a comfortable posture without disturbing the TCP. Joint position actuators track $q_{cmd}$. Each 10 Hz control step = 50 physics steps; IK is recomputed every 10 physics steps.
**1-D intuition example:** J = 0.5, e = 0.02 → plain inverse Δq = 0.040; DLS gives 0.5·0.02/(0.25 + 0.0025) = **0.0396**. Near a singularity J = 0.01: plain inverse = **2.0 rad** (violent), DLS = 0.01·0.02/(0.0001 + 0.0025) = **0.077 rad** (safe).
**Gripper:** Menagerie actuator 8, ctrl 255 = open, 0 = closed. Check the range in the XML.
**Success:** block centre within the target radius in xy, $|z_{obj} - (h_{tgt} + b/2)| < 1$ cm, gripper open, block speed < 1 cm/s after 1 s of settling.
**Mac notes:** offscreen `mujoco.Renderer` works with plain `python`. The interactive viewer needs `mjpython`.

### Module 8 — Robot data generation
**Logic:** for each episode: random layout (seed ≥ 1000; seeds 0–49 are reserved for evaluation), random goal, random human clip → retarget → add smooth noise (σ = 1.5 cm on 8 knots) → with probability 0.2 inject a failure (early release, or close 2–3 cm off-centre) → track → log the state/action/attach at 10 Hz, exactly as in Module 5. No rendering, so it's fast.
**Size:** 300 episodes × ~80 steps ≈ 24k transitions in about **1–2 min** on an M-series CPU.
**Subsets for the data-efficiency experiment:** N ∈ {0, 5, 10, 25, 50, 100, 300} episodes (nested, fixed seeds).

---

## 5. World model

### Module 9 — Ensemble dynamics model
**Purpose:** predict how the block moves, given the state, the action and *who* acts (hand or robot).
**In:** $x = [\bar s,\ \bar a,\ e]$, where $\bar\cdot$ means standardized (subtract mean, divide by std over the training set) and e ∈ {0 human, 1 robot}. Dimension 8 + 4 + 1 = 13.
**Out:** $\Delta\hat p_{obj} \in \mathbb R^3$ (standardized, then de-normalized), attach logit ∈ ℝ.
**Architecture:** **5 independent MLPs** (13 → 128 → 128 → 128 → 4, SiLU), each with a different random init and its own bootstrap resample of the data. About 35k parameters each.

**Theory, piece by piece:**
- **Predict the change, not the next state.** Most of the time the block doesn't move, so Δ ≈ 0 is the right default, and the network learns deviations from "nothing happens". This is easier and more stable than predicting the absolute next state.
- **Ensembles measure ignorance (epistemic uncertainty).** Where there was lots of training data, the 5 members agree. Where there was none (e.g. a robot grasp 2 cm off-centre that no human ever did), they disagree. The disagreement $\sigma_{ens}$ tells the planner where *not* to trust the model.
- **Embodiment flag.** Shared physics (an object follows a closed hand, falls when released, stays put when untouched) is learned once from both sources. Robot-specific differences (gripper tolerance is much tighter than a hand's) are carried by e.
- **Compounding error and multi-step loss.** A model that is 1 mm wrong per step can be 2 cm wrong after 20 steps, because each prediction is fed back in. Training on its own 5-step rollouts teaches the model to correct its drift:

$$\mathcal L = \sum_{k=1}^{5}\Big(\big\lVert \Delta\hat p^{(k)}_{obj} - \Delta p^{(k)}_{obj}\big\rVert^2 + w_a\,\operatorname{BCE}\big(\hat c^{(k)}, c^{(k)}\big)\Big),\qquad w_a = 0.5$$

where step k's input state is built from the model's own predictions at steps < k.

**Training recipe (3 variants, compared in E2):**
1. *Sim-only:* train on N robot episodes.
2. *Human-pretrain → fine-tune (ours):* train on all human data (e = 0), then continue on N robot episodes (e = 1) with lr ×0.3, mixing 20% human batches to avoid forgetting.
3. *Co-train:* human + robot from the start, with robot batches upweighted.

Adam, lr 1e-3, batch 256, 3k steps pretrain + 1.5k fine-tune per member. **About 1 min total on CPU.**

**Example (ensemble uncertainty):**
- Centred grasp, then lift: members predict $\Delta z_{obj}$ = {0.020, 0.021, 0.019, 0.020, 0.022} → mean 2.0 cm, std 0.1 cm → **confident**.
- Grasp 2.5 cm off-centre (never in the human data): {0.020, 0.000, 0.017, 0.001, 0.000} → mean 0.8 cm, std 0.9 cm → **uncertain**. The planner avoids it.

### Why deliberate failure clips matter (theory)
A planner *optimizes* against the model. Wherever the model is wrong in an optimistic way, the optimizer will find that spot and use it. This is called **model exploitation**. A model trained only on successful demos has never seen a missed grasp, so it may predict that the block lifts even when the fingers close next to it. Failure clips put real data in exactly those regions, and ensemble disagreement covers what's still missing. Experiment E6 tests this directly.

---

## 6. Planning

### Module 10 — Shrinking-horizon MPC with CEM, seeded by my path
**Purpose:** choose the robot's next moves by imagining many slightly different versions of my retargeted path inside the world model, keeping the one that ends best, executing a little, then re-planning.
**In:** current sim state s, the retargeted prior path $p^*$ (Module 6), the WM ensemble. **Out:** the next 5 actions (0.5 s) to execute.

**Decision variables** $\theta \in \mathbb R^{26}$: 8 knots × 3 (xyz residual added to the prior path, linearly interpolated, first knot fixed at 0), plus 2 integer shifts of the close and open times (±5 steps).

**CEM theory (cross-entropy method):** keep a Gaussian over θ. Sample many candidates, score them, keep the best few ("elites"), refit the Gaussian to the elites, and repeat. It needs no gradients and handles discrete timing shifts.
$$\theta_i \sim \mathcal N(\mu, \operatorname{diag}\sigma^2),\quad \mu \leftarrow \operatorname{mean}(\theta_{elite}),\quad \sigma \leftarrow \max(\operatorname{std}(\theta_{elite}),\ \sigma_{min})$$
N = 256 samples, 25 elites, 4 iterations. Initial σ: 1.5 cm for knots, 2 steps for timings.

**Cost** (predicted at the end of the plan; m indexes ensemble members):
$$J(\theta) = \underbrace{\frac{1}{M}\sum_m D_m(\theta)}_{\text{expected miss}} + \beta\,\underbrace{\operatorname{std}_m D_m(\theta)}_{\text{model doubt}} + w_s\sum_k\lVert r_k - r_{k-1}\rVert^2$$
$$D_m = \frac{\lVert \hat d^{\,xy}_{to}\rVert^2}{(2\,\text{cm})^2} + \frac{(\hat z_{obj} - h_{tgt} - b/2)^2}{(1\,\text{cm})^2},\qquad \beta = 1,\ w_s = 10$$
The β term is **pessimism**: of two plans with the same expected result, prefer the one the model is sure about.

**MPC theory (receding / shrinking horizon):** plan to the end of the task, execute only the first 5 steps, observe the real state, re-plan from there. The horizon shrinks as the task progresses. Warm start: the next plan starts from the previous μ, shifted by 5 steps. Re-planning closes the loop, so model errors and disturbances (a block that slipped or got pushed) are corrected.

**Example (1-D CEM intuition):** the best grasp x-offset is +1.0 cm. Iteration 1: μ = 0, σ = 1.5, elites average +0.8. Iteration 2: μ = 0.8, σ = 0.6, elites average +0.97. Iteration 3: μ = 0.97, σ = 0.3. Converged.

**Cost per re-plan:** 4 × 256 × ~60 steps × 5 members of a tiny MLP, batched in torch on CPU ≈ **50–100 ms**.

---

## 7. Experiments

All control evaluation uses **layout seeds 0–49** × 3 repeats; report mean ± std. WM prediction uses **held-out** robot episodes (seeds 900–999) and **held-out** human clips (5 clips).

| ID | Question | Setup | Output |
|---|---|---|---|
| **E1** | Does object-centric retargeting beat naive? | open-loop replay, naive vs two-anchor | success % |
| **E2** ★ | **How much robot data do my clips save?** | 10-step block-position error on held-out sim vs N ∈ {0,5,10,25,50,100,300}, for sim-only / pretrain→FT / co-train | curve: error vs N (log x) |
| **E3** ★ | Does it control better? | success on seeds 0–49: open-loop two-anchor; MPC+WM sim-only N=25; **MPC+WM pretrain→FT N=25**; MPC+WM human-only N=0 (zero-shot); MPC+WM sim-only N=300 (data-rich reference); MPC with the true simulator as model, N=32 samples (upper bound) | table |
| **E4** | Does my path help the search? | MPC (ours) with human prior vs isotropic prior (straight-line scripted path), samples ∈ {16, 64, 256} | success vs samples |
| **E5** | Robustness | block pushed 5 cm at step 15; object position noise 2 cm; open loop vs MPC | success % |
| **E6** | Do failure clips matter? | WM pretrained with vs without the 15 failure clips: attach AUROC on robot failure states, plus E3 success | table |
| **E7** | One video, three goals | MPC (ours) on plate / pad / box-top | success % per goal |
| **Fig** | Embodiment gap made visible | predicted P(lift succeeds) vs grasp offset (2D heatmap): human-only WM, adapted WM, true sim | 3 heatmaps |
| **Fig** | WM on my real video | roll the WM on a held-out clip from the hand motion only; draw the predicted block on the video frames | mp4 |

**Main table** (fill with measured numbers only):

| Method | plate | pad | box-top |
|---|---|---|---|
| Naive retarget, open loop | | | |
| Two-anchor retarget, open loop | | | |
| MPC + WM sim-only (25 ep) | | | |
| **MPC + WM human-pretrained (25 ep)** | | | |
| MPC + WM human-only (0 ep) | | | |
| MPC + WM sim-only (300 ep) | | | |
| MPC + true simulator (upper bound) | | | |

**What would count as a positive result:** pretrain→FT reaches the sim-only error at N ≈ 25 with fewer robot episodes (the curve shifts left), and E3 "ours (25)" approaches "sim-only (300)". **A negative result is still a result:** if human pretraining doesn't help, the heatmap figure will show *why* (e.g. the embodiment gap in grasp tolerance), and you say that in the README.

---

## 8. Stretch ideas (only if the core is done)
- **Distill MPC into a policy:** collect (s, MPC action) pairs and train an MLP policy. Compare speed (~0.05 ms vs ~80 ms per step) and success.
- **Residual RL:** PPO on residual actions over the retargeted path, with WM rollouts as extra imagined data.

---

## 9. Glossary
| Term | Meaning |
|---|---|
| PnP | Find an object's pose from known 3D points and their pixels |
| DLS IK | Damped least-squares inverse kinematics: joint motion for a desired gripper motion, stable near singularities |
| Ensemble | Several models trained differently; their disagreement estimates what the model doesn't know |
| Epistemic uncertainty | Uncertainty from lack of data (reducible), vs aleatoric (noise, irreducible) |
| Model exploitation | An optimizer finding and using the model's errors |
| CEM | Sample → keep the best → refit → repeat |
| MPC | Plan ahead, execute a little, re-plan |
| Pretrain → fine-tune | Learn general structure on big/cheap data, adapt on small/expensive data |
| AUROC | Chance a classifier ranks a random positive above a random negative (0.5 = guessing) |
