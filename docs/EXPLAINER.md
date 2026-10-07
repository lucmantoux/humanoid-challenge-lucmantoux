# palm-prior — equations, frames, formats

Numbers live in [RESULTS.md](../RESULTS.md). How to run lives in [README.md](../README.md). This file is the rest: frames, the state, the equations, defaults.

Frames: **C** camera, **T** table (ArUco ORIGIN, x right, y toward the sitter, z up), **W** sim world (z up). Units m, rad, s. \(X_C = R_{CT} X_T + t_{CT}\). Control rate 10 Hz.

v1 is `results/v1/` (8-D state, rest-scale \(k\), two-anchor height from the video). v2 is `results/v2/` (11-D state, palm-size height, hybrid). Code still implements both. Where they differ, both are written.

## §2 State and action

State \(s \in \mathbb{R}^8\):

| Index | Symbol | Meaning |
|---|---|---|
| 0–2 | \(d_{eo}=p_{ee}-p_{obj}\) | fingers relative to the lid |
| 3–4 | \(d_{to}=p_{tgt}^{xy}-p_{obj}^{xy}\) | target relative to the lid, horizontal |
| 5 | \(z_{obj}\) | lid centre height |
| 6 | \(h_{tgt}\) | target top: plate 0.015, pad 0.005, box 0.08 |
| 7 | \(g\) | closed 1, open 0 |

Action \(a=[p_{ee}(t{+}1)-p_{ee}(t),\; g(t{+}1)]\in\mathbb{R}^4\).

v2 appends \(p_{tgt}\in\mathbb{R}^3\), constant inside an episode: \(s\in\mathbb{R}^{11}\). `next_state` copies the tail.

Horizontal quantities are differences, so a table shift does not change \(s\). Height is absolute.

Given a predicted lid motion \(\Delta\hat p_{obj}\):

$$
d_{eo}'=d_{eo}+a_{xyz}-\Delta\hat p_{obj},\quad
d_{to}'=d_{to}-\Delta\hat p_{obj}^{xy},\quad
z_{obj}'=z_{obj}+\Delta\hat p_{obj}^{z},\quad
h_{tgt}'=h_{tgt},\quad g'=a_g.
$$

Worked example. Hand 1 cm above a 4 cm block, closed, target 15 cm in \(x\) and 5 cm in \(y\), plate. Lift 2 cm.

\(s=[0,0,0.01,\;0.15,0.05,\;0.02,\;0.015,\;1]\), \(a=[0,0,0.02,1]\).

If \(\Delta p_{obj}=[0,0,0.019]\), then \(s'=[0,0,0.011,\;0.15,0.05,\;0.039,\;0.015,\;1]\). A miss is \(\Delta p_{obj}\approx 0\): the gap grows, \(z_{obj}\) stays.

Attach label, human and robot: \(g=1\) and \(z_{obj}>b/2+0.01\).

Built only in `src/palm_prior/state.py`.

## §7 Phone to centimetres

Pinhole: \([u,v,1]^\top\sim K[X,Y,Z]^\top\). Calibrate \(K\) and `dist` from a 9×6 checkerboard, 25 mm squares. RMS must be \(<0.5\) px. Stabilisation off.

A pixel is a ray. The ArUco board (`DICT_4X4_50`, 2×2, 60 mm markers, 15 mm gaps) gives \(R_{CT},t_{CT}\). One static pose per clip: median translation, mean rotation. Object points are mirrored in \(y\) so \(z_T\) points out of the table. Ray \(r=K^{-1}[u,v,1]^\top\), \(n=R_{CT}e_z\), \(c=n\cdot t_{CT}\):

$$
Z_{\mathrm{plane}}=\frac{c}{n\cdot r},\qquad X_C=Z_{\mathrm{plane}}\,r,\qquad X_T=R_{CT}^\top(X_C-t_{CT}).
$$

A point at known height \(h\) meets \(z_T=h\).

**v1 rest-scale.** MediaPipe world landmarks have the right shape and the wrong size. PnP pose times one clip constant \(k=\mathrm{median}_{rest} Z_{\mathrm{plane}}/Z_{\mathrm{pnp}}\), from the first and last 2 s on the table. Pinch \(p_C=k(R_h\tfrac12(X_4+X_8)+t_h)\).

**v2 palm size.** Eight measured back-of-hand lengths in `configs/default.yaml`. Per frame, tilt-correct pixel lengths, least-squares scale \(s=\sum L\ell'/\sum(\ell')^2\), \(Z=f s\), \(f=(f_x+f_y)/2\). Drop \(c<0.7\) or residual \(>0.15\); need \(\ge 4\) segments. Kalman on \(\log Z\), RTS smooth. \(\kappa=Z_{\mathrm{lid}}/Z_{\mathrm{ruler}}\) at grasp and release, interpolated. Pinch is landmarks 4 and 8, One Euro, lid-locked while closed. Undistort pixels with `cv2.undistortPoints(..., P=K)` before the fit. Gripper: rest fingers together is open; holding the lid spreads them and counts as closed.

Aperture \(a=\lVert\ell_4-\ell_8\rVert/\lVert\ell_0-\ell_9\rVert\). Lid: HSV blob, ray at \(z_T=b/2\) on the table, pinch depth when held. Filmed lid \(b=0.017\); sim cube \(0.04\) (the gripper cannot pinch 1.7 cm). Plate radius 0.0585.

## §8 Two-anchor, naive, hybrid

Similarity \(T(x)=\alpha x+\beta\), two points, no fit:

$$
\alpha=\frac{P-G}{x_r-x_g},\qquad \beta=G-\alpha x_g,\qquad
\kappa=\mathrm{clip}(|\alpha|,0.5,1.5).
$$

\(G\) lid xy, \(P\) target xy. If \(|x_r-x_g|<0.05\), \(\alpha=1\). Height in two-anchor: \(z^*=z_{\mathrm{grasp}}+\kappa(z_h-z_h(t_g))+h_{tgt}\,s(t)\), \(s\) ramps 0 to 1 from grasp to release.

Worked example, metres: \(x_g=(0.10,0.05)\), \(x_r=(0.30,0.15)\), \(G=(0.50,0.10)\), \(P=(0.60,-0.12)\) give \(\alpha=-0.04-1.08i\), \(\beta=0.45+0.21i\). \(T(x_r)=P\).

**Naive:** one fixed \(\alpha,\beta\) from clip 1, reused; objects are not an input.

**Hybrid (v2):** same \(\alpha,\beta\), then endpoint correction so grasp xy is the lid and release xy is the target. Height is minimum jerk, \(z(\tau)=z_0+(z_1-z_0)(10\tau^3-15\tau^4+6\tau^5)\). \(z_{\mathrm{grasp}}=\) lid centre. \(z_{\mathrm{release}}=h_{tgt}+b/2+0.005\). Carry \(\max(0.12,z_{\mathrm{release}}+0.10)\). Timing \(k_g,k_r\) from the video gripper.

## §9 Damped IK

$$
\Delta q=J^\top(JJ^\top+\lambda^2 I)^{-1}e,\qquad\lambda=0.05,
$$

plus null-space \(k_n(q_{\mathrm{home}}-q)\), \(k_n=0.1\). Gripper down. Worked check: \(J=0.5\), \(e=0.02\) → damped \(0.0396\); \(J=0.01\) → undamped \(2.0\) rad, damped \(0.077\) rad.

Menagerie `franka_emika_panda`, no sites. TCP injected at hand-local `[0,0,0.1034]`. Actuators `actuator1`–`8`; `actuator8` gripper 255 open, 0 closed. Physics: timestep 0.002, `implicitfast`, elliptic cone, `impratio` 10. Fifty substeps per 10 Hz step.

Success: lid xy inside the target, \(|z_{obj}-(h_{tgt}+b/2)|<0.01\), gripper open, speed \(<0.01\) m/s after 1 s.

## §10 CEM-MPC (implemented, not scored)

Decision \(\theta\in\mathbb{R}^{26}\): 8 residual knots × 3, first knot 0, plus close/open shifts \(\pm 5\) steps. CEM: 256 samples, 25 elites, 4 iters. Execute 5 steps, re-plan.

$$
J=\mathrm{mean}_m D_m+\beta\,\mathrm{std}_m D_m+w_s\sum\lVert r_k-r_{k-1}\rVert^2,
$$

\(D\) is xy miss over 2 cm plus height miss over 1 cm, \(\beta=1\), \(w_s=10\). Embodiment bit \(e\in\{0,1\}\) on the world model. E3 was not run.

## World model (Module 9)

Five MLPs, hidden 128×3 SiLU. v1 input 13 = 8+4+1. v2 input 16 = 11+4+1. Output \(\Delta\hat p_{obj}\) and one attach logit. Loss: 5-step self-rollout, MSE on \(\Delta p\) plus \(0.5\cdot\mathrm{BCE}\). Adam \(10^{-3}\), batch 256. 3000 pretrain steps, 1500 finetune at \(0.3\times\) lr with 20% human batches.

Recipes: **robot only** (`sim_only`) 1500 steps of robot data; **phone then robot** (`pretrain_ft`) 3000 on clips then 1500 mixed; **both from the start** (`cotrain`) 3000 mixed. Human \(e=0\), robot \(e=1\).

## Pipeline I/O

| Module | In | Out |
|---|---|---|
| 1 calib | `calib.mp4` | `data/calib/camera.yaml` |
| 2 ArUco | frame, \(K\) | \(R_{CT},t_{CT}\) |
| 3 hand | frame, \(K\) | \(p_{ee}(t)\), \(g(t)\) |
| 4 lid | frame, HSV, \(p_{ee}\) | \(p_{obj}(t)\) |
| 5 transitions | those paths, \(\tau=2\) | 10 Hz \(s,a,\Delta p,\mathrm{attach}\) |
| 6 retarget | clip, lid, target | \(p^\star\), grip |
| 7 scene / IK | \(p^\star\) | Panda state |
| 8 robot data | hybrid, seeds 1000+ train, 900–999 hold, 0–49 eval | `episodes.npz` |
| 9 WM | \(s,a,e\) | \(\Delta\hat p_{obj}\), attach |
| 10 MPC | WM, prior path | actions (not scored) |

v1 QC: success / F1–F3 exactly one grasp, F4 zero. v2 QC: success one grasp; F1–F4 may have 1–8; F4 may have 0. Start error \(\le 2\) cm vs photographed opening lid.

Human time stretch \(\tau=2\). Nested \(N\in\{0,5,10,25,50,100,300\}\).

## §25 Experiments

| ID | Question | What ran |
|---|---|---|
| E1 | naive vs two-anchor vs hybrid | yes, open loop, seeds 0–49 × 3 goals |
| E2 | clips save robot tries? | yes, 10-step lid error and attach AUROC |
| E3–E7 | closed-loop, prior, robustness, failure ablation, per-goal MPC | not run |

Every parameter: `configs/default.yaml`.
