"""Frozen reference condition (rank 0): train the 768 -> 10 head on cached ViT features instead of running the ViT.

The frozen ViT never changes, so its CLS feature for every image in all 8 orientations was computed once
(notebooks/preprocessing/8ViewExtraction.ipynb). This trainer only fits the head on those saved features.
Everything else matches the LoRA path (src/run.py + src/train.py): the same subset (make_subset), image order
(epoch_order), starting head (init_head), optimiser, batch size, early stopping and runs.csv row.
Orientation: one of the 8 cached views per image per epoch, uniform like train_transform (same distribution,
not the same draw per image). Validation uses view 0. The test set is never loaded.

Runs:       python -m src.frozen --seeds 0 1 2 3 4 --split data/splits/split_v1.csv
Self-check: python -m src.frozen   (synthetic features, a few seconds on CPU)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data import CLASS_TO_IDX, read_split
from src.metrics import balanced_accuracy, cross_entropy
from src.model import HEAD_PARAMS, init_head
from src.run import COLUMNS, append_row, git_commit, load_config, make_subset
from src.train import epoch_order

FEATURES = Path("data/fingerprints")


def load_features(features_dir=FEATURES):
    """8-view CLS features (N x 8 x 768, float16, memory-mapped) and their index (row, filename, stem, label)."""
    feats = np.load(Path(features_dir) / "features_8view.npy", mmap_mode="r")
    index = pd.read_csv(Path(features_dir) / "features_8view_index.csv")
    assert feats.shape == (len(index), 8, 768), feats.shape
    return feats, index


def rows_for(df, index):
    """Feature-file row for each split row, checking that both files give the same label."""
    row_of = dict(zip(index["filename"], index["row"]))
    label_of = dict(zip(index["filename"], index["label"]))
    assert all(label_of[f] == lab for f, lab in zip(df["filename"], df["label"])), "split and features disagree on a label"
    return np.array([row_of[f] for f in df["filename"]])


def train_head(x_train, y_train, x_val, y_val, seed, lr, batch_size, weight_decay, max_epochs, patience):
    """x_train: n x 8 x 768 (every view), x_val: m x 768 (view 0). Returns the best-epoch head and a record."""
    head = init_head(seed)
    assert sum(p.numel() for p in head.parameters()) == HEAD_PARAMS
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)  # same optimiser as src/train.py
    n = len(y_train)
    best_val_ce, best_epoch, best_state, bad = float("inf"), 0, None, 0
    history, t0 = [], time.time()

    for epoch in range(1, max_epochs + 1):
        order = epoch_order(n, seed, epoch)  # identical to the LoRA runs' order for this seed and epoch
        # which of the 8 views each image gets this epoch; own generator, offset so it isn't the order's stream
        views = torch.randint(8, (n,), generator=torch.Generator().manual_seed(1_000_000 + seed * 1000 + epoch))
        head.train()
        total = 0.0
        for i in range(0, n, batch_size):
            idx = torch.tensor(order[i:i + batch_size])
            loss = F.cross_entropy(head(x_train[idx, views[idx]]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            val_ce = F.cross_entropy(head(x_val), y_val).item()
        history.append({"epoch": epoch, "train_ce": total / n, "val_ce": val_ce})
        if val_ce < best_val_ce:
            best_val_ce, best_epoch, bad = val_ce, epoch, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:  # same stopping rule as src/train.py
            break

    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "best_val_ce": best_val_ce, "history": history,
                  "train_time_s": time.time() - t0}


def run_frozen(config_path, seed, out_root, results_csv="results/final_results/runs_frozen.csv", split=None, features_dir=FEATURES):
    """One frozen run = one seed: subset, train the head, score validation, save, append one row to runs.csv."""
    cfg = load_config(config_path)
    if split:
        cfg["split"] = str(split)
    assert cfg["rank"] == 0, "src/frozen.py is only for the frozen condition"
    run_id = f"{cfg['condition']}_s{seed}"
    out = Path(out_root) / run_id
    if (out / "row.json").exists():
        print(f"{run_id} already finished, skipping")
        return json.loads((out / "row.json").read_text())

    feats, index = load_features(features_dir)
    train_df = make_subset(read_split(cfg["split"], "train"), cfg["n_per_class"], seed)  # same subset as LoRA
    val_df = read_split(cfg["split"], "val")
    x_train = torch.from_numpy(np.asarray(feats[rows_for(train_df, index)], dtype=np.float32))
    x_val = torch.from_numpy(np.asarray(feats[rows_for(val_df, index), 0], dtype=np.float32))
    y_train = torch.tensor([CLASS_TO_IDX[c] for c in train_df["label"]])
    y_val = torch.tensor([CLASS_TO_IDX[c] for c in val_df["label"]])

    head, rec = train_head(x_train, y_train, x_val, y_val, seed, cfg["lr"], cfg["batch_size"],
                           cfg["weight_decay"], cfg["max_epochs"], cfg["patience"])
    with torch.no_grad():
        logits = head(x_val)
    val_ce = cross_entropy(logits, y_val)
    assert abs(val_ce - rec["best_val_ce"]) < 1e-3, "saved head doesn't reproduce the best validation CE"

    row = {"run_id": run_id, "condition": cfg["condition"], "rank": 0, "seed": seed,
           "n_per_class": cfg["n_per_class"], "lr": cfg["lr"], "alpha": 0, "best_epoch": rec["best_epoch"],
           "val_ce": round(val_ce, 6), "val_bal_acc": round(balanced_accuracy(logits, y_val), 6),
           "test_ce": "", "test_bal_acc": "", "trainable_params": HEAD_PARAMS,
           "train_time_s": round(rec["train_time_s"]), "gpu": "cpu", "git_commit": git_commit()}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "best.pt")
    (out / "history.json").write_text(json.dumps(rec["history"], indent=1))
    (out / "config.json").write_text(json.dumps({**cfg, "features": str(Path(features_dir) / "features_8view.npy")}, indent=1))
    (out / "row.json").write_text(json.dumps(row, indent=1))
    append_row(row, results_csv)
    return row


def _self_check():
    import tempfile

    rng = np.random.default_rng(0)
    classes = sorted(CLASS_TO_IDX)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 30 fake images per class (20 train, 10 val); features = class centre + noise, so a head can learn them
        split = pd.DataFrame([(f"{c}_{i}.jpg", c, "train" if i < 20 else "val") for c in classes for i in range(30)],
                             columns=["filename", "label", "split"])
        split[["lon", "lat", "cell_id"]] = [0.0, 0.0, "x"]
        split.to_csv(tmp / "split.csv", index=False)
        y = np.array([CLASS_TO_IDX[c] for c in split["label"]])
        feats = rng.normal(size=(10, 768))[y][:, None, :] + rng.normal(scale=3, size=(len(split), 8, 768))
        np.save(tmp / "features_8view.npy", feats.astype(np.float16))
        pd.DataFrame({"row": range(len(split)), "filename": split["filename"], "stem": split["filename"].str[:-4],
                      "label": split["label"]}).to_csv(tmp / "features_8view_index.csv", index=False)
        cfg = {**load_config("configs/frozen.yaml"), "split": str(tmp / "split.csv"), "n_per_class": 10, "max_epochs": 20}
        (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg))

        kw = dict(results_csv=tmp / "runs.csv", features_dir=tmp)
        row = run_frozen(tmp / "cfg.yaml", 0, tmp / "runs", **kw)
        again = run_frozen(tmp / "cfg.yaml", 0, tmp / "runs", **kw)  # finished run: skipped, no duplicate row
        with open(tmp / "runs.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and list(rows[0]) == COLUMNS, "expected exactly one row with the Plan's columns"
        assert row == again and row["test_ce"] == "" and row["trainable_params"] == HEAD_PARAMS
        assert row["val_ce"] < np.log(10) - 0.5, f"head didn't learn separable features: val CE {row['val_ce']}"

        repeat = run_frozen(tmp / "cfg.yaml", 0, tmp / "runs_b", tmp / "runs_b.csv", features_dir=tmp)
        assert repeat["val_ce"] == row["val_ce"], "same seed gave a different result"
        print(row)
    print("frozen OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Frozen reference runs on cached features (no args = self-check)")
    ap.add_argument("--seeds", type=int, nargs="+", help="e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--split", help="overrides the config's split, e.g. data/splits/split_v1.csv")
    ap.add_argument("--out-root", default="runs/Frozen")
    ap.add_argument("--results", default="results/final_results/runs_frozen.csv")
    ap.add_argument("--features", default=str(FEATURES))
    a = ap.parse_args()
    if a.seeds is None:
        _self_check()
    else:
        for s in a.seeds:
            r = run_frozen(a.config, s, a.out_root, a.results, a.split, a.features)
            print(f"{r['run_id']}: val CE {r['val_ce']:.4f}, val balanced acc {r['val_bal_acc']:.3f}, "
                  f"best epoch {r['best_epoch']}")
