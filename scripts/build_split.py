"""Build the spatial train/val/test split and save it to data/splits/split_v1.csv.

1. Put every tile in a 41 km x 41 km grid cell (EPSG:3035).
2. Deal whole cells into 7 folds with StratifiedGroupKFold: fold 0 = test, fold 1 = val, the rest = train.
3. Buffer: any train photo within 41 km of a val/test photo becomes "buffer" (src/data.py read_split skips it).
4. Check for leakage and class balance, then save. Same columns as split_temp.csv.

Cell size and buffer come from notebooks/preprocessing/CellSize.ipynb (see docs/decision-log/dylan.md).
Run from the repo root: .venv/bin/python scripts/build_split.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KDTree

CELL_KM = 41      # grid cell size
BUFFER_KM = 41    # min distance from any train photo to any val/test photo
SEED = 0          # fixed: never redraw the split after training starts
N_FOLDS = 7       # 1 test + 1 val + 5 train, about 14 / 14 / 72 %
MAX_N = 200       # largest per-class training subset the pilot may use

COORDS = Path("data/coords.csv")
IMAGES = Path("data/raw/EuroSAT_RGB")
OUT = Path("data/splits/split_v1.csv")

df = pd.read_csv(COORDS)
assert len(df) == 27_000 and df["filename"].is_unique, (len(df), df["filename"].is_unique)

# Grid cell id like "3412_2687" (column_row), so rows and columns can't collide
cell_m = CELL_KM * 1000
df["cell_id"] = (df["x3035"] // cell_m).astype(int).astype(str) + "_" + (df["y3035"] // cell_m).astype(int).astype(str)

# Deal whole cells into folds, keeping the class mix similar in every fold
folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_of = np.empty(len(df), dtype=int)
for k, (_, rows) in enumerate(folds.split(df, df["label"], groups=df["cell_id"])):
    fold_of[rows] = k
df["split"] = np.select([fold_of == 0, fold_of == 1], ["test", "val"], default="train")

# Buffer: train photos too close to any val/test photo can't be used for training
xy = df[["x3035", "y3035"]].to_numpy()
held_out = df["split"] != "train"
dist_to_held_out, _ = KDTree(xy[held_out]).query(xy[~held_out], k=1)
too_close = np.zeros(len(df), dtype=bool)
too_close[np.flatnonzero(~held_out)] = dist_to_held_out[:, 0] < BUFFER_KM * 1000
df.loc[too_close, "split"] = "buffer"

# ---- Checks: the script stops here if the split is unsafe or unbalanced ----
in_use = df[df["split"] != "buffer"]
assert (in_use.groupby("cell_id")["split"].nunique() == 1).all(), "a cell appears in two splits"

train, held = df[df["split"] == "train"], df[df["split"].isin(["val", "test"])]
min_km = KDTree(held[["x3035", "y3035"]].to_numpy()).query(train[["x3035", "y3035"]].to_numpy(), k=1)[0].min() / 1000
assert min_km >= BUFFER_KM, f"train photo only {min_km:.1f} km from val/test"

counts = pd.crosstab(df["label"], df["split"])
expected = df["label"].value_counts().reindex(counts.index) / N_FOLDS   # each class's own share of one fold, same class order as counts
for split in ["val", "test"]:
    low = counts[split] < expected / 2                            # Decision 7: under half its share = uneven
    assert not low.any(), f"uneven {split}: {counts.loc[low, split].to_dict()}"
assert (counts["train"] >= MAX_N).all(), f"train pool under {MAX_N}: {counts['train'][counts['train'] < MAX_N].to_dict()}"

missing = [f"{lab}/{f}" for f, lab in zip(df["filename"], df["label"]) if not (IMAGES / lab / f"{f}.jpg").exists()]
assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"

# ---- Save in split_temp.csv's format (filename with .jpg) ----
df["filename"] = df["filename"] + ".jpg"
OUT.parent.mkdir(parents=True, exist_ok=True)
df[["filename", "label", "lon", "lat", "cell_id", "split"]].to_csv(OUT, index=False)

print(counts[["train", "val", "test", "buffer"]].to_string())
print(f"\ntotal: {df['split'].value_counts().to_dict()}")
print(f"cells per split: {df.groupby('split')['cell_id'].nunique().to_dict()}")
print(f"closest train photo to val/test: {min_km:.1f} km")
print(f"wrote {OUT}")
