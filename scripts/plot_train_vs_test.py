"""Training vs test loss only, for the write-up and slides.

Run from the repo root:  python3 scripts/plot_train_vs_test.py   ->  results/figures/train_vs_test_*.png
Training CE comes from runs/<run>/history.json (per epoch); test CE from runs/<run>/row.json, measured once
at the kept checkpoint (lowest validation CE). Nothing here chooses anything, it only plots finished results.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"frozen": "#1baf7a", "r4": "#2a78d6", "r32": "#eb6834"}
NAME = {"frozen": "Frozen + linear head", "r4": "LoRA r=4", "r32": "LoRA r=32"}
INK, MUTED, GRID = "#222222", "#6b6b66", "#e6e5e0"
SEEDS = range(5)
GROUP = {"frozen": "Frozen", "r4": "R4", "r32": "R32"}  # runs/<GROUP>/<cond>_s<seed>/
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def load(c, s):
    d = RUNS / GROUP[c] / f"{c}_s{s}"
    hist = json.loads((d / "history.json").read_text())
    row = json.loads((d / "row.json").read_text())
    best = row["best_epoch"]
    train_at_best = next(h["train_ce"] for h in hist if h["epoch"] == best)
    return hist, best, train_at_best, float(row["test_ce"])


R = {c: [load(c, s) for s in SEEDS] for c in COL}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# A. Training loss per epoch, with the test loss of the kept checkpoint as a star
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
for ax, c in zip(axes, COL):
    for hist, best, tr, te in R[c]:
        ax.plot([h["epoch"] for h in hist], [h["train_ce"] for h in hist], color=COL[c], lw=1.8, alpha=0.8)
        ax.plot([best, best], [tr, te], color=MUTED, lw=1, ls=":")  # the train-test gap at the kept epoch
        ax.plot(best, te, "*", ms=12, color=INK, mec="white", mew=0.8, zorder=5)
    ax.set_title(NAME[c], loc="left", color=INK, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 2.4)
axes[0].set_ylabel("Cross-entropy (nats)")
axes[0].plot([], [], color=INK, lw=1.8, label="Training loss (per epoch)")
axes[0].plot([], [], "*", ms=12, color=INK, mec="white", label="Test loss (kept checkpoint)")
axes[0].plot([], [], ":", color=MUTED, label="Gap at kept epoch")
axes[0].legend(loc="upper right")
fig.suptitle("Training vs test loss, 5 seeds per condition", x=0.01, y=1.03, ha="left", color=INK, fontsize=12)
save(fig, "train_vs_test_curves.png")


# B. Training vs test CE at the kept checkpoint (mean ± std, dots = seeds)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
w = 0.36
for i, c in enumerate(COL):
    for j, (k, lab, alpha) in enumerate([(2, "Training", 0.35), (3, "Test", 1.0)]):
        v = [r[k] for r in R[c]]
        x = i + (j - 0.5) * w
        ax.bar(x, st.mean(v), w * 0.92, color=COL[c], alpha=alpha, yerr=st.stdev(v), capsize=3,
               error_kw={"lw": 1, "ecolor": INK})
        ax.scatter([x] * len(v), v, s=10, color=INK, zorder=3)
        ax.annotate(f"{st.mean(v):.3f}", (x, max(v) + (st.stdev(v) if j == 0 else 0)), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(3), [NAME[c] for c in COL])
ax.bar(0, 0, color=MUTED, alpha=0.35, label="Training")
ax.bar(0, 0, color=MUTED, label="Test")
ax.legend(loc="upper right")
ax.set_ylabel("Cross-entropy (nats)")
ax.set_title("Training vs test loss at the kept checkpoint", loc="left", color=INK, fontsize=11)
save(fig, "train_vs_test_bars.png")


# C. Per seed: training (open) and test (filled) side by side, so every run is visible
fig, ax = plt.subplots(figsize=(7.2, 3.9))
for i, c in enumerate(COL):
    for s, (hist, best, tr, te) in enumerate(R[c]):
        x = i + (s - 2) * 0.12
        ax.plot([x, x], [tr, te], color=COL[c], lw=1.2, alpha=0.6)
        ax.plot(x, tr, "o", ms=7, mfc="white", mec=COL[c], mew=1.6)
        ax.plot(x, te, "o", ms=7, color=COL[c], mec="white", mew=1)
ax.set_xticks(range(3), [NAME[c] for c in COL])
ax.plot([], [], "o", mfc="white", mec=INK, label="Training")
ax.plot([], [], "o", color=INK, label="Test")
ax.legend(loc="upper right")
ax.set_ylabel("Cross-entropy (nats)")
ax.set_title("Every run: training vs test loss (one pair per seed, s0 to s4 left to right)", loc="left", color=INK, fontsize=10.5)
save(fig, "train_vs_test_per_seed.png")

for c in COL:
    tr = [r[2] for r in R[c]]; te = [r[3] for r in R[c]]
    print(f"{c}: train {st.mean(tr):.3f} ± {st.stdev(tr):.3f} | test {st.mean(te):.3f} ± {st.stdev(te):.3f} | gap {st.mean(te) - st.mean(tr):.3f}")
