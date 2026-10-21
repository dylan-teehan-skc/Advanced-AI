"""Score every final run on the test set, once. Run only after all design choices are fixed.

For each final run it loads best.pt (the lowest-validation-CE epoch), re-scores validation and checks it matches the
val_ce already recorded (proof the right checkpoint was loaded), then scores test and writes test_ce and
test_bal_acc into that run's row in the results CSV (results/final_results/runs_<condition>.csv) and its row.json. It also saves the test logits
(test_logits.npy) for later error analysis. A run whose test_ce is already filled in is skipped, so test is never
scored twice.

Frozen (laptop, seconds, uses the cached features):
    python scripts/evaluate_test.py frozen
LoRA (Colab T4, a few minutes per run; images in data/raw/EuroSAT_RGB as in the LoRA notebook, else --image-root):
    python scripts/evaluate_test.py r4  --runs-root <DRIVE>/runs --results <DRIVE>/results/runs_r4.csv
    python scripts/evaluate_test.py r32 --runs-root <DRIVE>/runs --results <DRIVE>/results/runs_r32.csv
Run from the repo root.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `src` imports work from scripts/
from src.data import CLASS_TO_IDX, CLASSES, ROOT, read_split
from src.metrics import balanced_accuracy, class_averaged_cross_entropy, cross_entropy
from src.run import load_config

SEEDS = range(5)
VAL_TOL = 1e-3   # same tolerance run.py and frozen.py used when they first recorded val_ce


def frozen_logits(run_dir, cfg, split, features_dir):
    """Head on cached view-0 features (no augmentation), exactly as frozen.py scored validation."""
    from src.frozen import load_features, rows_for
    from src.model import init_head
    feats, index = load_features(features_dir)
    head = init_head(0)
    head.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    head.eval()
    df = read_split(cfg["split"], split)
    x = torch.from_numpy(np.asarray(feats[rows_for(df, index), 0], dtype=np.float32))
    with torch.no_grad():
        return head(x), df


def lora_logits(run_dir, cfg, seed, split, image_root, model_cache):
    """Rebuild the LoRA model, load the best trainable weights, predict with eval_transform (no augmentation)."""
    from torch.utils.data import DataLoader
    from src.data import EuroSATDataset
    from src.model import build_model
    from src.run import predict
    from src.transforms import eval_transform
    if seed not in model_cache:
        model = build_model(cfg["rank"], seed, cfg["lora_dropout"])
        missing, unexpected = model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"), strict=False)
        assert not unexpected, f"best.pt doesn't match this model: {unexpected[:3]}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cache.clear()
        model_cache[seed] = (model.to(device), device)
    model, device = model_cache[seed]
    df = read_split(cfg["split"], split)
    loader = DataLoader(EuroSATDataset(df, eval_transform, image_root), batch_size=64, num_workers=2)
    logits, _ = predict(model, loader, device)
    return logits, df


def update_csv(path, run_id, values):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
    hits = [r for r in rows if r["run_id"] == run_id]
    assert len(hits) == 1, f"{path}: expected one row for {run_id}, found {len(hits)}"
    hits[0].update(values)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="One-time test evaluation of the final runs")
    ap.add_argument("condition", choices=["frozen", "r4", "r32"])
    ap.add_argument("--runs-root", default=None, help="default runs/Frozen, runs/R4 or runs/R32")
    ap.add_argument("--results", default=None, help="default results/final_results/runs_<condition>.csv")
    ap.add_argument("--features", default="data/fingerprints")
    ap.add_argument("--image-root", default=str(ROOT), help="unzipped EuroSAT_RGB folder (LoRA only)")
    a = ap.parse_args()
    a.runs_root = a.runs_root or f"runs/{ {'frozen': 'Frozen', 'r4': 'R4', 'r32': 'R32'}[a.condition] }"
    a.results = a.results or f"results/final_results/runs_{a.condition}.csv"

    cfg = load_config(f"configs/{a.condition}.yaml")
    test = read_split(cfg["split"], "test")
    counts = test["label"].value_counts().reindex(CLASSES)
    print(f"test set: {len(test)} images, per class {counts.min()} to {counts.max()}")
    with open(a.results, newline="") as f:
        recorded = {r["run_id"]: r for r in csv.DictReader(f)}

    cache, summary = {}, []
    for seed in SEEDS:
        run_id = f"{a.condition}_s{seed}"
        run_dir = Path(a.runs_root) / run_id
        row = recorded[run_id]
        if row["test_ce"] not in ("", None):
            print(f"{run_id}: test already scored ({row['test_ce']}), skipping")
            summary.append((run_id, float(row["test_ce"]), float(row["test_bal_acc"])))
            continue

        def logits_for(split):
            if a.condition == "frozen":
                return frozen_logits(run_dir, cfg, split, a.features)
            assert Path(a.image_root).is_dir(), f"no images at {a.image_root}: pass --image-root"
            return lora_logits(run_dir, cfg, seed, split, a.image_root, cache)

        # 1. same checkpoint? re-score validation and compare with the val_ce recorded at training time
        v_logits, v_df = logits_for("val")
        v_ce = cross_entropy(v_logits, [CLASS_TO_IDX[c] for c in v_df["label"]])
        assert abs(v_ce - float(row["val_ce"])) < VAL_TOL, \
            f"{run_id}: best.pt gives val CE {v_ce:.6f}, recorded {row['val_ce']}; wrong checkpoint?"

        # 2. the one test evaluation
        t_logits, t_df = logits_for("test")
        y = [CLASS_TO_IDX[c] for c in t_df["label"]]
        values = {"test_ce": round(cross_entropy(t_logits, y), 6),
                  "test_bal_acc": round(balanced_accuracy(t_logits, y), 6)}
        extra = {"test_ce_class_avg": round(class_averaged_cross_entropy(t_logits, y), 6)}

        np.save(run_dir / "test_logits.npy", np.asarray(t_logits, dtype=np.float32))
        update_csv(a.results, run_id, values)
        rj = run_dir / "row.json"
        if rj.exists():
            j = json.loads(rj.read_text())
            j.update(values); j.update(extra)
            rj.write_text(json.dumps(j, indent=1))
        print(f"{run_id}: val CE {v_ce:.4f} (matches), test CE {values['test_ce']:.4f}, "
              f"test balanced acc {values['test_bal_acc']:.3f}, class-averaged test CE {extra['test_ce_class_avg']:.4f}")
        summary.append((run_id, values["test_ce"], values["test_bal_acc"]))

    ce = np.array([s[1] for s in summary])
    acc = np.array([s[2] for s in summary])
    print(f"\n{a.condition}: test CE {ce.mean():.4f} ± {ce.std(ddof=1):.4f}, "
          f"test balanced acc {acc.mean():.3f} ± {acc.std(ddof=1):.3f} over {len(ce)} runs")


if __name__ == "__main__":
    main()
