"""Make the loss/comparison figures for the write-up and slides.

Run from the repo root:  python3 scripts/plot_results.py
Reads results/*.csv and runs/*/history.json, writes PNGs to results/figures/.
Only plots numbers that already exist; nothing here chooses anything, so the test set is not used for design.
"""
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, RUNS = ROOT / "results", ROOT / "runs"
OUT = RES / "figures"
OUT.mkdir(exist_ok=True)

# one colour per condition, same in every figure
COL = {"frozen": "#1baf7a", "r4": "#2a78d6", "r32": "#eb6834"}
NAME = {"frozen": "Frozen + linear head", "r4": "LoRA r=4", "r32": "LoRA r=32"}
INK, MUTED, GRID = "#222222", "#6b6b66", "#e6e5e0"
SEEDS = range(5)
GROUP = {"frozen": "Frozen", "r4": "R4", "r32": "R32"}  # runs/<GROUP>/<cond>_s<seed>/

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else None


def final_rows():
    """Final run per condition and seed, from runs/<cond>_s<seed>/row.json."""
    out = {}
    for c in COL:
        out[c] = []
        for s in SEEDS:
            p = RUNS / GROUP[c] / f"{c}_s{s}" / "row.json"
            if p.exists():
                out[c].append(json.loads(p.read_text()))
    return out


def history(cond, seed):
    p = RUNS / GROUP[cond] / f"{cond}_s{seed}" / "history.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


F = final_rows()
lora = ["r4", "r32"]


# 1. Training vs validation loss curves, one panel per condition (5 seeds each)
conds_with_hist = [c for c in COL if any(history(c, s) for s in SEEDS)]
fig, axes = plt.subplots(1, len(conds_with_hist), figsize=(4.2 * len(conds_with_hist), 3.4), sharey=True, squeeze=False)
for ax, c in zip(axes[0], conds_with_hist):
    for s in SEEDS:
        h = history(c, s)
        if not h:
            continue
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_ce"] for r in h], color=COL[c], lw=1.4, ls="--", alpha=0.55)
        ax.plot(ep, [r["val_ce"] for r in h], color=COL[c], lw=2)
        best = min(h, key=lambda r: r["val_ce"])
        ax.plot(best["epoch"], best["val_ce"], "o", ms=6, color=COL[c], mec="white", mew=1.5, zorder=4)
        t = num(F[c][s].get("test_ce")) if s < len(F[c]) else None
        if t is not None:  # test CE is one number, measured once at the kept checkpoint
            ax.plot(best["epoch"], t, "*", ms=11, color=INK, mec="white", mew=0.8, zorder=5)
    ax.set_title(NAME[c], color=INK, loc="left", fontsize=11)
    ax.set_ylim(0, 2.45)
    ax.set_xlabel("Epoch")
axes[0][0].set_ylabel("Cross-entropy (nats)")
axes[0][0].plot([], [], color=INK, lw=2, label="Validation")
axes[0][0].plot([], [], color=INK, lw=1.4, ls="--", alpha=0.55, label="Training")
axes[0][0].plot([], [], "o", color=INK, mec="white", label="Best val (checkpoint kept)")
axes[0][0].plot([], [], "*", ms=11, color=INK, mec="white", label="Test CE at kept checkpoint")
axes[0][0].legend(loc="upper right")
fig.suptitle("Training vs validation loss per epoch, with test loss at the kept checkpoint (5 seeds each)", x=0.01, y=1.04, ha="left", color=INK, fontsize=12)
save(fig, "fig1_train_val_curves.png")


# 2. Test CE per seed for all conditions, joined by seed (paired design: same seed = same training subset)
order = [c for c in ["frozen", "r4", "r32"] if all(num(r.get("test_ce")) is not None for r in F[c]) and F[c]]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s_ in SEEDS:
    ys = [F[c][s_]["test_ce"] for c in order]
    ax.plot(range(len(order)), ys, color=MUTED, lw=1, alpha=0.5, zorder=1)
for i_, c in enumerate(order):
    v = [r["test_ce"] for r in F[c]]
    ax.scatter([i_] * len(v), v, s=46, color=COL[c], edgecolor="white", lw=1.5, zorder=3)
    m, sd = st.mean(v), st.stdev(v)
    ax.errorbar(i_ + 0.22, m, yerr=sd, fmt="D", ms=7, color=COL[c], capsize=4, lw=2, zorder=4)
    ax.annotate(f"mean {m:.3f}\n± {sd:.3f}", (i_ + 0.3, m), ha="left", va="center", color=INK, fontsize=9)
ax.set_xticks(range(len(order)), [NAME[c] for c in order])
ax.set_xlim(-0.4, len(order) - 0.2)
ax.set_ylabel("Test cross-entropy (nats, lower is better)")
ax.set_title("Test loss by seed (grey lines join the same seed)", loc="left", color=INK, fontsize=11)
save(fig, "fig2_test_ce_paired.png")


# 3. Paired difference r32 - r4 against the pre-registered "≈" rule
d = [F["r32"][s]["test_ce"] - F["r4"][s]["test_ce"] for s in SEEDS]
md, sdd = st.mean(d), st.stdev(d)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.bar([f"seed {s}" for s in SEEDS], d, color=[COL["r32"] if v < 0 else COL["r4"] for v in d], width=0.55, zorder=2)
ax.axhline(0, color=INK, lw=1)
ax.axhline(md, color=INK, lw=1.5, ls="--")
ax.axhspan(-sdd, sdd, color=MUTED, alpha=0.12, lw=0, zorder=0)
ax.annotate(f"mean\n{md:+.5f}", (4.4, md - 0.003), va="top", color=INK, fontsize=9)
ax.annotate(f"±1 std\n({sdd:.5f})", (4.4, sdd), va="top", color=MUTED, fontsize=9)
ax.annotate("below 0 = r32 has lower loss", (1.55, 0.004), va="bottom", ha="center", color=MUTED, fontsize=8)
ax.set_xlim(-0.5, 5.4)
ax.set_ylabel("r32 − r4 test CE (nats)")
verdict = "|mean| < std, so ≈ by the rule" if abs(md) < sdd else "|mean| ≥ std, so not ≈"
ax.set_title(f"r32 − r4 test loss, paired by seed\n|mean| {abs(md):.5f} vs std {sdd:.5f}: {verdict}, by a hair",
             loc="left", color=INK, fontsize=10.5)
save(fig, "fig3_paired_difference.png")


# 4. Train vs validation vs test CE at the kept checkpoint (mean ± std, dots = seeds)
def at_best(c, s, key):
    """Train CE comes from history at the best epoch; val/test CE from row.json."""
    if key == "train_ce":
        h = history(c, s)
        if not h:
            return None
        return min(h, key=lambda r: r["val_ce"])["train_ce"]
    return num(F[c][s].get(key)) if s < len(F[c]) else None

fig, ax = plt.subplots(figsize=(7, 3.9))
w = 0.26
splits = [("train_ce", "Training", 0.3), ("val_ce", "Validation", 0.6), ("test_ce", "Test", 1.0)]
for i, c in enumerate(COL):
    for j, (key, lab, alpha) in enumerate(splits):
        v = [x for x in (at_best(c, s, key) for s in SEEDS) if x is not None]
        xpos = i + (j - 1) * w
        if not v:
            ax.annotate("not yet\nevaluated", (xpos, 0.02), ha="center", fontsize=8, color=MUTED)
            continue
        ax.bar(xpos, st.mean(v), w * 0.92, color=COL[c], alpha=alpha,
               yerr=st.stdev(v), capsize=3, error_kw={"lw": 1, "ecolor": INK})
        ax.scatter([xpos] * len(v), v, s=10, color=INK, zorder=3)
        ax.annotate(f"{st.mean(v):.3f}", (xpos, max(v)), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(3), [NAME[c] for c in COL])
for key, lab, alpha in splits:
    ax.bar(0, 0, color=MUTED, alpha=alpha, label=lab)
ax.legend(loc="upper right", ncol=3)
ax.set_ylim(0, 0.7)
ax.set_ylabel("Cross-entropy (nats)")
ax.set_title("Training vs validation vs test loss at the kept checkpoint", loc="left", color=INK, fontsize=11)
save(fig, "fig4_val_vs_test_ce.png")


# 5. Learning-rate pilots: best validation CE vs LR per condition (validation only)
pil = {"frozen": [r for r in rows(RES / "pilot_lr.csv")],
       "r4": rows(RES / "pilot_lora_lr_r4.csv"), "r32": rows(RES / "pilot_lora_lr_r32.csv")}
fig, ax = plt.subplots(figsize=(6, 3.8))
for c, rs in pil.items():
    by = {}
    for r in rs:
        by.setdefault(float(r["lr"]), []).append(float(r["val_ce"]))
    lrs = sorted(by)
    for lr in lrs:
        ax.scatter([lr] * len(by[lr]), by[lr], s=12, color=COL[c], alpha=0.5)
    ax.plot(lrs, [st.mean(by[lr]) for lr in lrs], "-o", color=COL[c], lw=2, ms=6, mec="white", label=NAME[c])
    chosen = {"frozen": 0.01, "r4": 0.001, "r32": 0.001}[c]
    ax.plot(chosen, st.mean(by[chosen]), "o", ms=13, mfc="none", mec=COL[c], mew=1.5)
ax.set_xscale("log")
ax.set_xlabel("Learning rate (log)")
ax.set_ylabel("Best validation CE (nats)")
ax.legend(loc="upper right")
ax.set_title("LR pilots (validation only) · ring = LR used for final runs", loc="left", color=INK, fontsize=11)
save(fig, "fig5_lr_pilots.png")


# 6. Cost vs benefit: trainable params, train time, best epoch, test balanced accuracy
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
panels = [("trainable_params", "Trainable parameters (log)", True),
          ("train_time_s", "Train time per run (min, T4)", False),
          ("best_epoch", "Best epoch (early stopping)", False),
          ("test_bal_acc", "Test balanced accuracy", False)]
for ax, (key, title, logy) in zip(axes, panels):
    for i, c in enumerate(COL):
        v = [num(r[key]) for r in F[c] if num(r[key]) is not None]
        if key == "train_time_s":
            v = [t / 60 for t in v] if c != "frozen" else []  # frozen ran on cached CPU features, not comparable
        if not v:
            ax.annotate("CPU run,\nn/a" if key == "train_time_s" else "not yet\nevaluated", (i, 0.45),
                        xycoords=("data", "axes fraction"), ha="center", va="center", color=MUTED, fontsize=8)
            continue
        ax.bar(i, st.mean(v), 0.6, color=COL[c], yerr=st.stdev(v) if len(set(v)) > 1 else None,
               capsize=3, error_kw={"lw": 1, "ecolor": INK})
        ax.scatter([i] * len(v), v, s=10, color=INK, zorder=3)
        lab = f"{st.mean(v):,.0f}" if key in ("trainable_params", "train_time_s", "best_epoch") else f"{st.mean(v):.3f}"
        top = max(v + [st.mean(v) + (st.stdev(v) if len(set(v)) > 1 else 0)])
        ax.annotate(lab, (i, top), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK)
    if logy:
        ax.set_yscale("log")
    if key == "test_bal_acc":
        ax.set_ylim(0.78, 0.975)
    ax.set_xticks(range(3), ["Frozen", "r=4", "r=32"])
    ax.set_title(title, loc="left", color=INK, fontsize=10)
fig.suptitle("What each condition costs and what it buys", x=0.01, y=1.06, ha="left", color=INK, fontsize=12)
save(fig, "fig6_cost_vs_benefit.png")


# 7. Frozen data-size pilot: val CE vs N per class (justifies N = 20)
by = {}
for r in rows(RES / "pilot_n.csv"):
    by.setdefault(int(r["n_per_class"]), []).append(float(r["val_ce"]))
ns = sorted(by)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for n in ns:
    ax.scatter([n] * len(by[n]), by[n], s=12, color=COL["frozen"], alpha=0.5)
ax.plot(ns, [st.mean(by[n]) for n in ns], "-o", color=COL["frozen"], lw=2, mec="white")
ax.plot(20, st.mean(by[20]), "o", ms=13, mfc="none", mec=COL["frozen"], mew=1.5)
ax.set_xscale("log")
ax.set_xticks(ns, [str(n) for n in ns])
ax.set_xlabel("Training images per class (log)")
ax.set_ylabel("Best validation CE (nats)")
ax.set_title("Frozen N pilot (validation only) · ring = N used", loc="left", color=INK, fontsize=11)
save(fig, "fig7_frozen_n_pilot.png")

# 8. All conditions on one axis: mean validation and training CE per epoch (band = min..max over seeds)
fig, ax = plt.subplots(figsize=(7, 4))
for c in COL:
    hs = [h for h in (history(c, s) for s in SEEDS) if h]
    if not hs:
        continue
    for key, ls, lw, alpha in [("val_ce", "-", 2.2, 0.15), ("train_ce", "--", 1.4, 0.0)]:
        L = max(len(h) for h in hs)
        ep, mean, lo, hi = [], [], [], []
        for e in range(L):
            vals = [h[e][key] for h in hs if len(h) > e]
            if len(vals) < 3:  # only plot epochs most seeds reached, so the mean isn't one seed
                break
            ep.append(e + 1); mean.append(st.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
        ax.plot(ep, mean, ls, color=COL[c], lw=lw, alpha=1 if key == "val_ce" else 0.7)
        if alpha:
            ax.fill_between(ep, lo, hi, color=COL[c], alpha=alpha, lw=0)
        if key == "val_ce":
            ax.annotate(NAME[c], (ep[-1], mean[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
ax.plot([], [], "-", color=INK, lw=2.2, label="Validation (mean, band = seed range)")
ax.plot([], [], "--", color=INK, lw=1.4, alpha=0.7, label="Training (mean)")
ax.legend(loc="upper right")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy (nats)")
ax.set_ylim(0, 2.45)
ax.set_xlim(0.5, None)
ax.set_title("Loss curves of all three conditions (epochs reached by ≥3 of 5 seeds)", loc="left", color=INK, fontsize=11)
save(fig, "fig8_all_curves_overlay.png")


# 9. Test CE overall vs averaged per class (class-avg weights every class equally)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
w = 0.34
for i, c in enumerate(COL):
    for j, (key, lab, alpha) in enumerate([("test_ce", "Overall", 1.0), ("test_ce_class_avg", "Class-averaged", 0.5)]):
        v = [num(r.get(key)) for r in F[c] if num(r.get(key)) is not None]
        if not v:
            continue
        xpos = i + (j - 0.5) * w
        ax.bar(xpos, st.mean(v), w * 0.92, color=COL[c], alpha=alpha, yerr=st.stdev(v), capsize=3,
               error_kw={"lw": 1, "ecolor": INK})
        ax.scatter([xpos] * len(v), v, s=10, color=INK, zorder=3)
        ax.annotate(f"{st.mean(v):.3f}", (xpos, max(v)), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3), [NAME[c] for c in COL])
ax.bar(0, 0, color=MUTED, label="Overall")
ax.bar(0, 0, color=MUTED, alpha=0.5, label="Class-averaged")
ax.legend(loc="upper right")
ax.set_ylabel("Test cross-entropy (nats)")
ax.set_title("Test loss: overall vs class-averaged", loc="left", color=INK, fontsize=11)
save(fig, "fig9_test_ce_class_avg.png")


# Table of every run (spec: report every run), for the write-up appendix
with open(OUT / "all_runs_table.csv", "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["run_id", "condition", "seed", "lr", "best_epoch", "epochs_run", "train_ce_at_best", "val_ce",
                  "test_ce", "test_ce_class_avg", "val_bal_acc", "test_bal_acc", "trainable_params", "train_time_s"])
    for c in COL:
        for s, r in enumerate(F[c]):
            h = history(c, s)
            wtr.writerow([r["run_id"], c, s, r["lr"], r["best_epoch"], len(h) if h else "",
                          f"{at_best(c, s, 'train_ce'):.4f}" if h else "", r["val_ce"], r.get("test_ce", ""),
                          r.get("test_ce_class_avg", ""), r["val_bal_acc"], r.get("test_bal_acc", ""),
                          r["trainable_params"], r["train_time_s"]])
print("wrote", OUT / "all_runs_table.csv")

print(f"\nr32 - r4 test CE: mean {md:+.4f}, std {sdd:.4f}, per seed {[round(x, 4) for x in d]}")
