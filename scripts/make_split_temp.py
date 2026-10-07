"""Fake random split with the same columns as split_v1.csv, so loaders can be built before the real spatial split exists.

Columns: filename, label, lon, lat, cell_id, split. filename and label come from the real EuroSAT RGB files
(data/eurosat/EuroSAT_RGB/<label>/<filename>); lon/lat/cell_id/split are random placeholders.
Do NOT use this split for any reported result.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/eurosat/EuroSAT_RGB")

rng = np.random.default_rng(0)
rows = sorted((p.name, p.parent.name) for p in ROOT.glob("*/*.jpg"))
df = pd.DataFrame(rows, columns=["filename", "label"])
df["lon"] = rng.uniform(-10, 30, len(df)).round(6)  # rough Europe bounding box
df["lat"] = rng.uniform(36, 60, len(df)).round(6)
df["cell_id"] = rng.integers(0, 500, len(df))
df["split"] = rng.choice(["train", "val", "test"], len(df), p=[0.7, 0.15, 0.15])

assert len(df) == 27000 and df.label.nunique() == 10 and df.filename.is_unique, (len(df), df.label.nunique())
assert (df.groupby("split").label.nunique() == 10).all()

df.to_csv("data/splits/split_temp.csv", index=False)
print(pd.crosstab(df.label, df.split, margins=True))
