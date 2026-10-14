"""make_subset(pool, n, seed): the n training images per class that a seed trains on, spread across grid cells.

Images in the same 41 km cell look alike, and some classes are dominated by a few cells (in split_v1's training
pool one cell holds 300 of SeaLake's 714 images). A plain random pick would spend much of n on near-copies, by a
different amount for each seed. Instead, for each class, the seed shuffles the cells and the images inside each
cell, and the pick goes round the cells taking one image from each, then a second from each, and so on until it
has n. So a cell only gives a second image once every cell of that class has given one.

Properties (checked by the self-check):
  - the same seed gives the same images, whatever order the pool's rows are in
  - exactly n images per class, and an error if a class has fewer than n
  - nested: for one seed, the n=10 subset is the first 10 of the n=20 subset, and so on
  - spread: within a class, no cell has 2+ more images picked than another, unless that one ran out

Self-check: python -m src.subset [path/to/split.csv]   (default data/splits/split_v1.csv)
"""
import sys

import numpy as np
import pandas as pd

from src.data import CLASS_TO_IDX


def make_subset(pool, n, seed):
    parts = []
    for label in sorted(pool["label"].unique()):
        # own random stream per class and seed, so one class's draw never shifts another's
        rng = np.random.default_rng([seed, CLASS_TO_IDX[label]])
        g = pool[pool["label"] == label].sort_values("filename")  # fixed starting order, independent of the pool's
        if n > len(g):
            raise ValueError(f"{label} has only {len(g)} images in the pool, can't take {n}")
        cells = np.sort(g["cell_id"].astype(str).unique())
        cell_pos = dict(zip(cells, rng.permutation(len(cells))))  # shuffled order of the cells
        g = g.assign(_cell=g["cell_id"].astype(str).map(cell_pos), _u=rng.random(len(g)))
        g["_round"] = g.groupby("_cell")["_u"].rank(method="first")  # shuffled order inside each cell
        parts.append(g.sort_values(["_round", "_cell"]).head(n))  # one per cell per round
    return pd.concat(parts).drop(columns=["_cell", "_u", "_round"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data import read_split

    split = sys.argv[1] if len(sys.argv) > 1 else "data/splits/split_v1.csv"
    pool = read_split(split, "train")
    names = lambda df: list(df["filename"])

    a = make_subset(pool, 50, 0)
    assert names(a) == names(make_subset(pool, 50, 0)), "same seed gave different images"
    shuffled = pool.sample(frac=1, random_state=123)
    assert names(a) == names(make_subset(shuffled, 50, 0)), "result depends on the pool's row order"
    assert set(names(a)) != set(names(make_subset(pool, 50, 1))), "different seeds gave the same images"
    assert list(a.columns) == list(pool.columns), "columns changed"

    smallest = pool["label"].value_counts().min()
    for seed in range(5):
        prev = None
        for n in (10, 20, 50, 100, 200, smallest):
            s = make_subset(pool, n, seed)
            assert (s["label"].value_counts() == n).all() and s["label"].nunique() == pool["label"].nunique()
            assert not s["filename"].duplicated().any()
            if prev is not None:  # nested: the smaller subset is the start of the bigger one, class by class
                for lab in s["label"].unique():
                    small, big = names(prev[prev["label"] == lab]), names(s[s["label"] == lab])
                    assert big[:len(small)] == small, f"not nested at n={n}"
            for lab, grp in s.groupby("label"):  # spread: a cell 2+ behind the busiest must have run out
                size = pool.loc[pool["label"] == lab, "cell_id"].value_counts()
                taken = grp["cell_id"].value_counts().reindex(size.index, fill_value=0)
                behind = taken < taken.max() - 1
                assert (taken[behind] == size[behind]).all(), f"{lab} not spread at n={n}"
            prev = s
    try:
        make_subset(pool, smallest + 1, 0)
        raise AssertionError("asking for more images than a class has should fail")
    except ValueError:
        pass

    # how much better spread than a plain random pick (the old stand-in)
    print(" n   distinct cells per image picked: plain random -> spread")
    for n in (10, 20, 50, 100, 200):
        old = np.mean([(pool.groupby("label").sample(n, random_state=s).groupby("label")["cell_id"].nunique() / n).mean()
                       for s in range(5)])
        new = np.mean([(make_subset(pool, n, s).groupby("label")["cell_id"].nunique() / n).mean() for s in range(5)])
        print(f"{n:>3}   {old:.2f} -> {new:.2f}")
    print(f"smallest class has {smallest} images, so n can be at most {smallest}")
    print("subset OK")
