"""One run = one config + one seed: build the subset and model, train, score validation, append one row to runs.csv.

The row is also saved as row.json in the run folder, so a rerun of a finished run is skipped instead of
appending a duplicate. Test columns stay blank: test is evaluated once per run only after the design freeze.

Self-check: python -m src.run  (tiny CPU run on a 2-per-class split)
"""
import csv
import json
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import ROOT, EuroSATDataset, read_split
from src.metrics import balanced_accuracy, cross_entropy
from src.model import build_model, count_trainable
from src.subset import make_subset  # spread across grid cells, see src/subset.py
from src.train import train
from src.transforms import eval_transform, train_transform

COLUMNS = ["run_id", "condition", "rank", "seed", "n_per_class", "lr", "alpha", "best_epoch", "val_ce",
           "val_bal_acc", "test_ce", "test_bal_acc", "trainable_params", "train_time_s", "gpu", "git_commit"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def append_row(row, path):
    """Append one run to runs.csv, writing the header if the file is new. Paddy's frozen trainer uses this too."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(pixel_values=x.to(device)).logits.cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def run(config_path, seed, out_root, results_csv="results/runs.csv", image_root=ROOT, num_workers=2, device=None):
    cfg = load_config(config_path)
    run_id = f"{cfg['condition']}_s{seed}"
    out = Path(out_root) / run_id
    if (out / "row.json").exists():
        print(f"{run_id} already finished, skipping")
        return json.loads((out / "row.json").read_text())

    train_ds = EuroSATDataset(make_subset(read_split(cfg["split"], "train"), cfg["n_per_class"], seed),
                              train_transform, image_root)
    val_ds = EuroSATDataset(read_split(cfg["split"], "val"), eval_transform, image_root)
    model = build_model(cfg["rank"], seed, cfg["lora_dropout"])
    state = train(model, train_ds, val_ds, out, seed, cfg["lr"], cfg["batch_size"], cfg["weight_decay"],
                  cfg["max_epochs"], cfg["patience"], num_workers, device)

    # score the best epoch, not the last one
    model.load_state_dict(torch.load(out / "best.pt"), strict=False)
    device = next(model.parameters()).device
    logits, labels = predict(model, DataLoader(val_ds, batch_size=64, num_workers=num_workers), device)
    val_ce = cross_entropy(logits, labels)
    assert abs(val_ce - state["best_val_ce"]) < 1e-3, "best.pt doesn't reproduce the best validation CE"

    row = {"run_id": run_id, "condition": cfg["condition"], "rank": cfg["rank"], "seed": seed,
           "n_per_class": cfg["n_per_class"], "lr": cfg["lr"], "alpha": cfg["rank"],
           "best_epoch": state["best_epoch"], "val_ce": round(val_ce, 6),
           "val_bal_acc": round(balanced_accuracy(logits, labels), 6), "test_ce": "", "test_bal_acc": "",
           "trainable_params": count_trainable(model), "train_time_s": round(state["train_time_s"]),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "git_commit": git_commit()}
    (out / "row.json").write_text(json.dumps(row, indent=1))
    append_row(row, results_csv)
    return row


if __name__ == "__main__":
    import tempfile

    # decision 4: the three conditions may differ only in condition, rank and lr
    cfgs = [load_config(f"configs/{c}.yaml") for c in ("frozen", "r4", "r32")]
    shared = lambda c: {k: v for k, v in c.items() if k not in ("condition", "rank", "lr")}
    assert shared(cfgs[0]) == shared(cfgs[1]) == shared(cfgs[2]), "configs differ beyond condition, rank and lr"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        import pandas as pd

        temp = "data/splits/split_temp.csv"
        tiny = pd.concat([read_split(temp, s).groupby("label").head(2) for s in ("train", "val")])
        tiny.to_csv(tmp / "tiny_split.csv", index=False)
        cfg = {**cfgs[1], "split": str(tmp / "tiny_split.csv"), "n_per_class": 2, "max_epochs": 2}
        (tmp / "tiny.yaml").write_text(yaml.safe_dump(cfg))

        csv_path = tmp / "runs.csv"
        row = run(tmp / "tiny.yaml", seed=0, out_root=tmp / "runs", results_csv=csv_path, num_workers=0, device="cpu")
        again = run(tmp / "tiny.yaml", seed=0, out_root=tmp / "runs", results_csv=csv_path, num_workers=0, device="cpu")

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and list(rows[0]) == COLUMNS, "expected exactly one row with the Plan's columns"
        assert row == again and rows[0]["run_id"] == "r4_s0" and rows[0]["test_ce"] == ""
        print(rows[0])

    print("run OK")
