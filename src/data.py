"""Reads any split CSV (filename, label, lon, lat, cell_id, split) and serves (image tensor, label index).

Works unchanged on split_temp.csv and split_v1.csv because it only uses the filename, label and split columns.
Takes a DataFrame, not a path, so it also accepts the rows returned by make_subset(pool, N, seed).

Self-check: python -m src.data [path/to/split.csv]
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "EuroSAT_RGB"
# ponytail: fixed alphabetical order, so label indices are identical for every run, split and person
CLASSES = sorted(["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                  "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def read_split(csv_path, split):
    """Rows of one split ('train', 'val' or 'test'). Any other rows (e.g. a buffer) are ignored."""
    df = pd.read_csv(csv_path)
    return df[df["split"] == split].reset_index(drop=True)


class EuroSATDataset(Dataset):
    def __init__(self, df, transform, root=ROOT):
        self.files = [Path(root) / lab / f for f, lab in zip(df["filename"], df["label"])]
        self.labels = [CLASS_TO_IDX[lab] for lab in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), self.labels[i]


if __name__ == "__main__":
    import sys

    from src.transforms import eval_transform, train_transform

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/splits/split_temp.csv"
    for split, tf in [("train", train_transform), ("val", eval_transform), ("test", eval_transform)]:
        ds = EuroSATDataset(read_split(csv, split), tf)
        assert len(ds) > 0, f"no {split} rows"
        assert all(p.exists() for p in ds.files), f"missing image files in {split}"
        assert set(ds.labels) == set(range(10)), f"{split} is missing a class"
        x, y = ds[0]
        assert x.shape == (3, 224, 224) and isinstance(y, int), (x.shape, y)
        print(f"{split}: {len(ds)} images OK")
    print("data OK")
