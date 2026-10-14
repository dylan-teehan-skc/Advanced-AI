"""Validation-only pilots for the frozen condition, on the cached features (a few minutes on a laptop).

Stage 1, images per class:     python scripts/pilot.py n
    N = 10, 20, 50, 100, 200 at the config's learning rate, 5 seeds each.
    Rule agreed before running (14 Oct): take the smallest N whose mean validation balanced accuracy is
    between 70% and 85%, so the frozen reference is neither failing nor at the ceiling, and LoRA has room.
Stage 2, frozen learning rate: python scripts/pilot.py lr --n <chosen N>
    lr = 0.001, 0.003, 0.01, 0.03 at the chosen N, 5 seeds each. Take the lowest mean validation CE.

Results go to results/pilot_n.csv, results/pilot_lr.csv and runs/pilot/, never results/runs.csv.
The test set is never loaded. Rerunning skips runs that already finished.
Run from the repo root.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `src` imports work from scripts/
from src.frozen import run_frozen
from src.run import load_config

SEEDS = range(5)
N_GRID = [10, 20, 50, 100, 200]
LR_GRID = [0.001, 0.003, 0.01, 0.03]
BAND = (0.70, 0.85)  # agreed 14 Oct, before any pilot results were seen


def sweep(settings, out_dir, results_csv):
    """Run every (N, lr) setting for all seeds, then return one summary row per setting."""
    base = load_config("configs/frozen.yaml")
    for n, lr in settings:
        folder = Path(out_dir) / f"n{n}_lr{lr:g}"
        folder.mkdir(parents=True, exist_ok=True)
        cfg_path = folder / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({**base, "n_per_class": n, "lr": lr}))
        for seed in SEEDS:
            r = run_frozen(cfg_path, seed, folder, results_csv)
            print(f"N={n:<4} lr={lr:<6g} seed {seed}: val CE {r['val_ce']:.4f}, "
                  f"balanced acc {r['val_bal_acc']:.3f}, best epoch {r['best_epoch']}")

    df = pd.read_csv(results_csv)
    wanted = {(n, round(lr, 10)) for n, lr in settings}
    df = df[[(n, round(lr, 10)) in wanted for n, lr in zip(df["n_per_class"], df["lr"])]]
    table = df.groupby(["n_per_class", "lr"]).agg(
        runs=("seed", "count"), ce_mean=("val_ce", "mean"), ce_sd=("val_ce", "std"),
        acc_mean=("val_bal_acc", "mean"), acc_sd=("val_bal_acc", "std"),
        max_best_epoch=("best_epoch", "max")).reset_index()

    print(f"\n{'N':>5} {'lr':>7}  runs   val CE (mean ± SD)   balanced acc (mean ± SD)")
    for t in table.itertuples():
        print(f"{t.n_per_class:>5} {t.lr:>7g}  {t.runs:>4}   {t.ce_mean:.4f} ± {t.ce_sd:.4f}      "
              f"{t.acc_mean:.3f} ± {t.acc_sd:.3f}")
    capped = table[table["max_best_epoch"] >= base["max_epochs"]]
    for t in capped.itertuples():
        print(f"WARNING: N={t.n_per_class} lr={t.lr:g} was still improving at the {base['max_epochs']}-epoch cap")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Frozen validation-only pilots")
    ap.add_argument("stage", choices=["n", "lr"])
    ap.add_argument("--n", type=int, help="chosen N, required for the lr stage")
    a = ap.parse_args()

    if a.stage == "n":
        lr = load_config("configs/frozen.yaml")["lr"]
        table = sweep([(n, lr) for n in N_GRID], "runs/pilot/n", "results/pilot_n.csv")
        inside = table[table["acc_mean"].between(*BAND)]
        if len(inside):
            print(f"\nRule: smallest N with mean balanced accuracy in {BAND[0]:.0%}-{BAND[1]:.0%} -> "
                  f"N = {inside['n_per_class'].min()}")
        else:
            print(f"\nNo N has mean balanced accuracy in {BAND[0]:.0%}-{BAND[1]:.0%}; look at the table and discuss")
    else:
        if a.n is None:
            ap.error("the lr stage needs --n")
        table = sweep([(a.n, lr) for lr in LR_GRID], f"runs/pilot/lr_n{a.n}", "results/pilot_lr.csv")
        best = table.loc[table["ce_mean"].idxmin()]
        print(f"\nRule: lowest mean validation CE -> lr = {best.lr:g}")
        if best.lr in (min(LR_GRID), max(LR_GRID)):
            print("WARNING: the best lr is at the edge of the grid; the true best may lie outside it")
