"""Compare the three conditions using every test prediction, not just one number per run.

Run from the repo root:  python3 scripts/compare_outputs.py   ->  results/figures/compare_*.png + compare_summary.txt
Reads runs/<Group>/<run>/test_logits.npy (one row per test image, in split_v1.csv test order) and row.json.
Everything here is analysis of finished runs: nothing is chosen with the test set.

What it shows and why:
- Paired differences in test CE between conditions, with two 95% intervals:
  (a) across the 5 seeds (t-interval, n=5): how much the result depends on training randomness;
  (b) cluster bootstrap over test grid cells: how much it depends on which test tiles we happened to get.
      Cells, not images, are resampled because neighbouring tiles are correlated (the split is by cell).
- Test CE per class, to see where the conditions differ.
- Reliability diagram + expected calibration error (ECE): CE punishes both wrong and over-confident
  predictions, so this separates "more often right" from "better calibrated".
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = ROOT / "data" / "splits" / "split_v1.csv"
CLASSES = sorted(["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                  "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"])  # same order as src/data.py
COL = {"frozen": "#1baf7a", "r4": "#2a78d6", "r32": "#eb6834"}
NAME = {"frozen": "Frozen + linear head", "r4": "LoRA r=4", "r32": "LoRA r=32"}
GROUP = {"frozen": "Frozen", "r4": "R4", "r32": "R32"}
INK, MUTED, GRID = "#222222", "#6b6b66", "#e6e5e0"
SEEDS = range(5)
T_975_DF4 = 2.776  # t critical value, 95% two-sided, 4 degrees of freedom (5 paired seeds)
N_BOOT = 10_000
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

test = pd.read_csv(SPLIT)
test = test[test["split"] == "test"].reset_index(drop=True)
y = test["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
cells = test["cell_id"].to_numpy()


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


# per-image NLL and probabilities for every run; check each reproduces the recorded test CE
NLL, PROB = {}, {}
for c in COL:
    NLL[c], PROB[c] = [], []
    for s in SEEDS:
        d = RUNS / GROUP[c] / f"{c}_s{s}"
        lp = log_softmax(np.load(d / "test_logits.npy").astype(np.float64))
        assert lp.shape[0] == len(y), f"{d}: {lp.shape[0]} logits vs {len(y)} test images"
        nll = -lp[np.arange(len(y)), y]
        recorded = float(json.loads((d / "row.json").read_text())["test_ce"])
        assert abs(nll.mean() - recorded) < 1e-5, f"{d}: recomputed {nll.mean():.6f} vs recorded {recorded}"
        NLL[c].append(nll)
        PROB[c].append(np.exp(lp))
    NLL[c] = np.stack(NLL[c])            # (5 seeds, n images)
    PROB[c] = np.stack(PROB[c])          # (5, n, 10)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# cluster bootstrap over grid cells (same resampled cells for both conditions, so it stays paired)
rng = np.random.default_rng(0)
uniq, inv = np.unique(cells, return_inverse=True)
cell_n = np.bincount(inv)


def cell_boot_ci(per_image_diff):
    cell_sum = np.bincount(inv, weights=per_image_diff)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boots = cell_sum[idx].sum(axis=1) / cell_n[idx].sum(axis=1)
    return np.percentile(boots, [2.5, 97.5])


PAIRS = [("r4", "frozen"), ("r32", "frozen"), ("r32", "r4")]
summary = [f"test images: {len(y)}, test grid cells: {len(uniq)} (median {int(np.median(cell_n))} images per cell)\n"]
res = {}
for a, b in PAIRS:
    d_seed = NLL[a].mean(axis=1) - NLL[b].mean(axis=1)          # one paired difference per seed
    m, sd = d_seed.mean(), d_seed.std(ddof=1)
    t_ci = (m - T_975_DF4 * sd / np.sqrt(5), m + T_975_DF4 * sd / np.sqrt(5))
    c_ci = cell_boot_ci((NLL[a] - NLL[b]).mean(axis=0))          # seed-averaged per-image difference
    wins = int((d_seed < 0).sum())
    res[(a, b)] = (d_seed, m, t_ci, c_ci, wins)
    summary.append(f"{a} - {b}: mean diff {m:+.4f} nats | 95% CI over seeds [{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] | "
                   f"95% CI over test cells [{c_ci[0]:+.4f}, {c_ci[1]:+.4f}] | {a} lower on {wins}/5 seeds")

# 1. Estimation plot: paired seeds (left) and differences with both intervals (right)
fig, ax = plt.subplots(figsize=(6, 4.4))
order = list(COL)
seed_ce = {c: NLL[c].mean(axis=1) for c in COL}
for s in SEEDS:
    ax.plot(range(3), [seed_ce[c][s] for c in order], color=MUTED, lw=1, alpha=0.5)
for i, c in enumerate(order):
    ax.scatter([i] * 5, seed_ce[c], s=46, color=COL[c], edgecolor="white", lw=1.5, zorder=3)
    ax.annotate(f"{seed_ce[c].mean():.3f}", (i, seed_ce[c].max()), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
ax.set_xticks(range(3), [NAME[c] for c in order])
ax.set_xlim(-0.4, 2.4)
ax.set_ylabel("Test cross-entropy (nats)")
ax.set_title("Test CE per seed (lines join the same seed)", loc="left", color=INK, fontsize=11)
save(fig, "compare_test_ce_per_seed.png")

fig, bx = plt.subplots(figsize=(7.5, 4.6))

labels = []
for k, (a, b) in enumerate(PAIRS):
    d_seed, m, t_ci, c_ci, wins = res[(a, b)]
    yk = len(PAIRS) - 1 - k
    bx.plot(t_ci, [yk + 0.12] * 2, color=COL[a], lw=6, alpha=0.35, solid_capstyle="butt")
    bx.plot(c_ci, [yk - 0.12] * 2, color=INK, lw=2)
    bx.scatter(d_seed, [yk + 0.12] * 5, s=22, color=COL[a], edgecolor="white", lw=1, zorder=3)
    bx.plot(m, yk, "D", ms=8, color=COL[a], mec="white", zorder=4)
    bx.annotate(f"mean {m:+.3f}  ·  {a} lower on {wins}/5 seeds", (-0.345, yk + 0.38),
                va="center", fontsize=9, color=INK)
    labels.append(f"{NAME[a]}\nvs {NAME[b]}")
bx.axvline(0, color=INK, lw=1)
bx.set_yticks(range(len(PAIRS))[::-1], labels)
bx.grid(axis="y", visible=False)
bx.set_xlabel("Difference in test CE (nats)   ·   below 0 = first-named model has lower loss")
bx.set_xlim(-0.35, 0.08)
bx.set_ylim(-0.45, len(PAIRS) - 0.4)
bx.plot([], [], color=MUTED, lw=6, alpha=0.35, label="95% CI across 5 seeds (training randomness)")
bx.plot([], [], color=INK, lw=2, label="95% CI across test grid cells (test sampling)")
bx.scatter([], [], s=22, color=MUTED, label="one seed")
bx.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=8.5)
bx.set_title("Paired differences with 95% intervals", loc="left", color=INK, fontsize=11)
save(fig, "compare_paired_differences.png")

# 2. Test CE per class: mean over seeds, bar = min..max across seeds
per_class = {c: np.array([[NLL[c][s][y == k].mean() for k in range(10)] for s in SEEDS]) for c in COL}
cls_order = np.argsort(per_class["frozen"].mean(axis=0))[::-1]
fig, ax = plt.subplots(figsize=(8, 5.2))
for j, c in enumerate(COL):
    pc = per_class[c]
    for r, k in enumerate(cls_order):
        yy = r + (j - 1) * 0.24
        ax.plot([pc[:, k].min(), pc[:, k].max()], [yy, yy], color=COL[c], lw=2, alpha=0.5)
        ax.plot(pc[:, k].mean(), yy, "o", ms=7, color=COL[c], mec="white", mew=1)
    ax.plot([], [], "o", color=COL[c], label=NAME[c])
counts = np.bincount(y, minlength=10)
ax.set_yticks(range(10), [f"{CLASSES[k]}  (n={counts[k]})" for k in cls_order])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Test cross-entropy (nats)   ·   dot = mean of 5 seeds, line = seed range")
ax.legend(loc="lower right")
ax.set_title("Test loss per class (sorted by frozen loss)", loc="left", color=INK, fontsize=11)
save(fig, "compare_per_class.png")
summary.append("\nper-class test CE (mean of 5 seeds):")
summary.append(f"{'class':22s}" + "".join(f"{c:>9s}" for c in COL) + "   r32-r4")
for k in cls_order:
    vals = [per_class[c][:, k].mean() for c in COL]
    summary.append(f"{CLASSES[k]:22s}" + "".join(f"{v:9.3f}" for v in vals) + f"   {vals[2] - vals[1]:+.3f}")

# 3. Reliability diagram (pooled over seeds) + ECE per seed
bins = np.linspace(0, 1, 11)


def reliability(prob):
    conf = prob.max(axis=1)
    correct = prob.argmax(axis=1) == y
    which = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    acc = np.array([correct[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    cm = np.array([conf[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    n = np.bincount(which, minlength=10)
    ece = np.nansum(n / len(y) * np.abs(acc - cm))
    return cm, acc, n, ece


fig, ax = plt.subplots(figsize=(6.2, 5))
fig2, hx = plt.subplots(figsize=(6.2, 3))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
ax.annotate("perfectly calibrated", (0.1, 0.13), rotation=38, color=MUTED, fontsize=8.5)
summary.append("\nexpected calibration error (10 bins), mean ± sd over seeds; mean top-class confidence vs accuracy:")
for c in COL:
    eces = [reliability(PROB[c][s])[3] for s in SEEDS]
    pooled = PROB[c].reshape(-1, 10)
    conf = pooled.max(axis=1); correct = pooled.argmax(axis=1) == np.tile(y, 5)
    which = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    acc = np.array([correct[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    cm = np.array([conf[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    n = np.bincount(which, minlength=10)
    ok = n >= 20  # skip near-empty bins so a handful of images don't draw a point
    ax.plot(cm[ok], acc[ok], "-o", color=COL[c], lw=2, ms=6, mec="white",
            label=f"{NAME[c]}  (ECE {st.mean(eces):.3f})")
    nz = n > 0  # log scale: leave out empty bins
    hx.plot(((bins[:-1] + bins[1:]) / 2)[nz], (n / n.sum())[nz], "-o", color=COL[c], lw=1.5, ms=4)
    summary.append(f"{c:8s} ECE {st.mean(eces):.4f} ± {st.stdev(eces):.4f} | confidence {conf.mean():.3f} vs accuracy {correct.mean():.3f}")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("Model confidence in its top class")
ax.set_ylabel("Accuracy in bin")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Reliability diagram (test, 5 seeds pooled)", loc="left", color=INK, fontsize=11)
hx.set_yscale("log")
hx.set_ylabel("Share of test images (log)")
hx.set_xlabel("Model confidence in its top class")
hx.set_xlim(0, 1.02)
for c in COL:
    hx.plot([], [], "-o", color=COL[c], label=NAME[c])
hx.legend(loc="upper left", fontsize=9)
hx.set_title("How confident each model is (test, 5 seeds pooled)", loc="left", color=INK, fontsize=11)
save(fig, "compare_reliability.png")
save(fig2, "compare_confidence_histogram.png")

# 4. Confusion matrices (row-normalised, 5 seeds pooled), for the appendix
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
short = ["AnnCrop", "Forest", "HerbVeg", "Highway", "Indust", "Pasture", "PermCrop", "Resid", "River", "SeaLake"]
for ax, c in zip(axes, COL):
    pred = PROB[c].argmax(axis=2).ravel(); true = np.tile(y, 5)
    cm = np.zeros((10, 10))
    np.add.at(cm, (true, pred), 1)
    cm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), short, rotation=60, fontsize=7.5)
    ax.set_yticks(range(10), short, fontsize=7.5)
    ax.grid(False)
    ax.set_xlabel("Predicted")
    ax.set_title(NAME[c], loc="left", color=INK, fontsize=11)
axes[0].set_ylabel("True class")
fig.suptitle("Confusion matrices (share of each true class, 5 seeds pooled; values under .02 hidden)",
             x=0.01, y=1.02, ha="left", color=INK, fontsize=12)
save(fig, "compare_confusion.png")

(OUT / "compare_summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
