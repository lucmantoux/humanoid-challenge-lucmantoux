"""E2: how many robot episodes the phone clips save (EXPLAINER §25).

Three variants, N in {0, 5, 10, 25, 50, 100, 300}, three seeds. The number reported
is the 10-step block-position error on the held-out robot episodes, plus attach AUROC.
The curve includes the baseline that says the block never moves.

Usage:
    uv run python scripts/exp_e2.py
    uv run python scripts/exp_e2.py overwrite=true
"""

from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _bootstrap  # noqa: E402,F401

from palm_prior.utils import Timer, ensure_dir, load_config, resolve, set_seed, set_torch_threads  # noqa: E402
from palm_prior.wm.model import Ensemble, NormStats, fit_stats  # noqa: E402
from palm_prior.wm.train import (  # noqa: E402
    evaluate,
    probe_lift,
    train_ensemble,
    windows_from_ids,
)


def _heldout_clips(clip_id: np.ndarray, is_failure: np.ndarray, n: int, seed: int) -> set[int]:
    """n clip ids, at least one of them a failure. Deterministic in `seed`."""
    order = []
    fail = {}
    for c, f in zip(clip_id.tolist(), is_failure.tolist()):
        if c not in fail:
            order.append(int(c))
            fail[int(c)] = bool(f)
    failures = [c for c in order if fail[c]]
    successes = [c for c in order if not fail[c]]
    rng = np.random.default_rng(seed)
    if not failures:
        raise RuntimeError("no failure clip in the human transitions")
    chosen = [int(rng.choice(failures))]
    chosen += [int(c) for c in rng.choice(successes, size=n - 1, replace=False)]
    return set(chosen)


def _take(s, a, dp, att, ids, mask):
    return s[mask], a[mask], dp[mask], att[mask], ids[mask]


def _never_moves(dp, ids, horizon: int) -> float:
    from palm_prior.wm.train import episode_starts

    starts = episode_starts(ids, horizon, dp)
    offs = np.arange(horizon)
    win = starts[:, None] + offs[None, :]
    total = np.asarray(dp[win], float).sum(axis=1)
    return float(np.linalg.norm(total, axis=1).mean() * 100.0)


def _rows_done(path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["variant"], int(float(row["N"])), int(float(row["seed"]))))
    return done


def _append(path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "N", "seed", "pos_err_cm", "attach_auroc"])
        if new:
            writer.writeheader()
        writer.writerow(row)


def _fit_and_windows(parts: list[tuple], horizon: int):
    """parts: (s, a, dp, att, ids, embodiment) used both to fit stats and to train."""
    s = np.concatenate([p[0] for p in parts])
    a = np.concatenate([p[1] for p in parts])
    dp = np.concatenate([p[2] for p in parts])
    stats = fit_stats(s, a, dp)
    pools = []
    for s_, a_, dp_, att_, ids_, emb, frac in parts:
        pools.append((windows_from_ids(s_, a_, dp_, att_, ids_, horizon), emb, frac))
    return stats, pools


def _train_variant(cfg, variant: str, n: int, seed: int, human, robot, held_robot) -> tuple[Ensemble, NormStats, float, float]:
    w = cfg.wm
    horizon = int(w.rollout_len)
    hidden = tuple(int(h) for h in w.hidden)
    model = Ensemble(int(w.n_members), hidden)
    if n == 0:
        robot_part = None
    else:
        train_seeds = np.arange(int(cfg.robot_data.train_seed_start), int(cfg.robot_data.train_seed_start) + n)
        mask = np.isin(robot[4], train_seeds)
        robot_part = _take(*robot, mask)

    if variant == "sim_only":
        stats, pools = _fit_and_windows([(*robot_part, 1.0, 1.0)], horizon)
        steps, lr = int(w.pretrain_steps), float(w.lr)
    elif variant == "pretrain_ft":
        # One human pretrain per seed, reused for every N. Stats stay the human ones
        # so the fine-tune sees the same normalisation the pretrain did.
        cache = ensure_dir(resolve(cfg.paths.checkpoints)) / f"_human_pretrain_seed{seed}.pt"
        stats, human_windows = _fit_and_windows([(*human, 0.0, 1.0)], horizon)
        if cache.exists():
            blob = torch.load(cache, weights_only=False)
            model.load_state_dict(blob["model"])
            stats = NormStats.load(blob["stats"])
        else:
            train_ensemble(
                model, [(human_windows[0][0], 0.0, 1.0)], int(w.pretrain_steps), float(w.lr),
                int(w.batch_size), horizon, float(w.attach_weight), stats, seed,
            )
            torch.save({"model": model.state_dict(), "stats": stats.save()}, cache)
        if robot_part is None:
            pools = [(human_windows[0][0], 0.0, 1.0)]
            steps, lr = 0, float(w.lr)
        else:
            robot_windows = windows_from_ids(*robot_part, horizon)
            steps, lr = int(w.finetune_steps), float(w.lr) * float(w.finetune_lr_factor)
            pools = [
                (robot_windows, 1.0, 1.0 - float(w.human_batch_frac)),
                (human_windows[0][0], 0.0, float(w.human_batch_frac)),
            ]
    elif variant == "cotrain":
        if robot_part is None:
            stats, pools = _fit_and_windows([(*human, 0.0, 1.0)], horizon)
        else:
            stats, pools = _fit_and_windows(
                [
                    (*robot_part, 1.0, 1.0 - float(w.human_batch_frac)),
                    (*human, 0.0, float(w.human_batch_frac)),
                ],
                horizon,
            )
        steps, lr = int(w.pretrain_steps), float(w.lr)
    else:
        raise ValueError(variant)

    if not (variant == "sim_only" and n == 0):
        train_ensemble(
            model, pools, steps, lr, int(w.batch_size), horizon,
            float(w.attach_weight), stats, seed,
        )
    if variant == "sim_only" and n == 0:
        pos, auroc, _ = float("nan"), float("nan"), 0.0
        # An untrained model is not a baseline. The caller writes the never-moves row.
        return model, stats, pos, auroc
    pos, auroc, _spread = evaluate(
        model, *held_robot, int(w.eval_horizon), stats, embodiment=1.0,
    )
    return model, stats, pos, auroc


def _plot(rows: list[dict], baseline: float, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"sim_only": "C1", "pretrain_ft": "C0", "cotrain": "C2"}
    labels = {"sim_only": "sim only", "pretrain_ft": "human pretrain, then fine-tune", "cotrain": "co-train"}
    ns = sorted({int(r["N"]) for r in rows})
    xs = [n if n > 0 else 0.4 for n in ns]
    for variant, colour in colours.items():
        mean, std = [], []
        for n in ns:
            vals = [float(r["pos_err_cm"]) for r in rows if r["variant"] == variant and int(r["N"]) == n]
            vals = [v for v in vals if np.isfinite(v)]
            mean.append(float(np.mean(vals)) if vals else np.nan)
            std.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        mean_a = np.array(mean)
        std_a = np.array(std)
        ax.plot(xs, mean_a, color=colour, marker="o", label=labels[variant])
        ax.fill_between(xs, mean_a - std_a, mean_a + std_a, color=colour, alpha=0.15)
    ax.axhline(baseline, color="0.4", ls="--", label="block never moves")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("robot episodes N")
    ax.set_ylabel("10-step block error (cm)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    set_seed(int(cfg.seed))
    set_torch_threads()
    results = ensure_dir(resolve(cfg.paths.results))
    ckpt_dir = ensure_dir(resolve(cfg.paths.checkpoints))
    out_csv = results / "e2.csv"
    overwrite = bool(cfg.get("overwrite", False))
    if overwrite and out_csv.exists():
        out_csv.unlink()

    human_z = np.load(resolve(cfg.paths.human_dir) / "transitions.npz")
    hold = _heldout_clips(
        human_z["clip_id"], human_z["is_failure"], int(cfg.human.n_heldout_clips), int(cfg.seed),
    )
    print("held-out human clips", sorted(hold))
    train_mask = ~np.isin(human_z["clip_id"], list(hold))
    human = _take(human_z["s"], human_z["a"], human_z["dp_obj"], human_z["attach"], human_z["clip_id"], train_mask)

    robot_z = np.load(resolve(cfg.paths.robot_dir) / "episodes.npz")
    robot = (robot_z["s"], robot_z["a"], robot_z["dp_obj"], robot_z["attach"], robot_z["episode_id"])
    held_mask = np.isin(
        robot_z["episode_id"],
        np.arange(int(cfg.robot_data.heldout_seed_start), int(cfg.robot_data.heldout_seed_start) + int(cfg.robot_data.n_heldout_episodes)),
    )
    held_robot = _take(*robot, held_mask)
    baseline = _never_moves(held_robot[2], held_robot[4], int(cfg.wm.eval_horizon))
    print(f"block-never-moves error {baseline:.2f} cm")

    done = set() if overwrite else _rows_done(out_csv)
    ns = [int(n) for n in cfg.robot_data.subsets]
    seeds = [int(cfg.seed) + i for i in range(3)]
    variants = ("sim_only", "pretrain_ft", "cotrain")

    with Timer("exp_e2"):
        for variant in variants:
            for n in ns:
                for seed in seeds:
                    key = (variant, n, seed)
                    if key in done:
                        print("skip", key)
                        continue
                    if variant == "sim_only" and n == 0:
                        row = {"variant": variant, "N": n, "seed": seed, "pos_err_cm": round(baseline, 4), "attach_auroc": 0.5}
                        _append(out_csv, row)
                        print(f"{variant} N={n} seed={seed}  {baseline:.2f} cm  (no robot data)")
                        continue
                    model, stats, pos, auroc = _train_variant(cfg, variant, n, seed, human, robot, held_robot)
                    row = {
                        "variant": variant, "N": n, "seed": seed,
                        "pos_err_cm": round(pos, 4), "attach_auroc": round(auroc, 4),
                    }
                    _append(out_csv, row)
                    print(f"{variant} N={n} seed={seed}  {pos:.2f} cm  auroc {auroc:.3f}")
                    if seed == seeds[0] and variant == "pretrain_ft" and n in (0, 25):
                        path = ckpt_dir / f"wm_{variant}_N{n}_seed{seed}.pt"
                        torch.save({"model": model.state_dict(), "stats": stats.save(), "cfg_hidden": list(cfg.wm.hidden)}, path)
                        centred, spread_c = probe_lift(model, stats, 1.0, 0.0)
                        missed, spread_m = probe_lift(model, stats, 1.0, 0.03)
                        print(
                            f"  lift Δz centred {centred*100:.2f} cm (spread {spread_c*100:.2f}), "
                            f"3 cm off {missed*100:.2f} cm (spread {spread_m*100:.2f})"
                        )

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    fig = results / "e2_curve.png"
    _plot(rows, baseline, fig)
    print(f"wrote {out_csv}")
    print(f"wrote {fig}")


if __name__ == "__main__":
    main(sys.argv[1:])
