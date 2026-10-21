"""Copy the per-epoch loss history out of each LoRA run's last.pt into runs/<run>/history.json.

Run in Colab after mounting Drive (needs torch). Then pull the history.json files into the repo's runs/ folder
and rerun scripts/plot_results.py: figure 1 gains r4 and r32 panels automatically.
src/train.py stores the history in last.pt under ckpt["state"]["history"] (one dict per epoch: train_ce, val_ce).
"""
import json
import sys
from pathlib import Path

import torch

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/aai-project/runs")
for ckpt_path in sorted(runs.glob("r*_s*/last.pt")):
    hist = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state"]["history"]
    (ckpt_path.parent / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"{ckpt_path.parent.name}: {len(hist)} epochs")
