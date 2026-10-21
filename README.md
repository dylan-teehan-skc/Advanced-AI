<p align="center">
  <h1 align="center">Does a Bigger LoRA Rank Help?</h1>
  <p align="center">
    A controlled experiment on parameter-efficient adaptation of a Vision Transformer to satellite imagery.<br/>
    ViT-B/16 on EuroSAT, 20 labelled images per class, a leakage-free spatial split, LoRA rank 4 vs rank 32.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Hugging_Face-PEFT_+_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab T4"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/runs-15_final_(3_×_5_seeds)-brightgreen?style=flat-square" alt="Runs"/>
  <img src="https://img.shields.io/badge/test_set-scored_once-blue?style=flat-square" alt="Test set scored once"/>
  <img src="https://img.shields.io/badge/data-EuroSAT_(MIT)-lightgrey?style=flat-square" alt="EuroSAT MIT"/>
  <img src="https://img.shields.io/badge/model-ViT--B%2F16_(Apache_2.0)-lightgrey?style=flat-square" alt="ViT Apache 2.0"/>
</p>

---

> **With limited labelled data and a spatial split, does increasing LoRA rank beyond a small value improve held-out predictive performance on EuroSAT, measured by cross-entropy, or does performance level off?**

## What this project is

A 4th-year **Advanced AI group project** (Track 3.2: parameter-efficient vision modelling). The aim is not the highest score. It is one clean, controlled comparison with repeated runs, a fair reference condition, and a test set that never influenced a design choice.

We take `google/vit-base-patch16-224-in21k`, pretrained on everyday photos (ImageNet-21k, no satellite data), and adapt it to EuroSAT land-cover tiles under three conditions:

| Condition | What trains | Trainable parameters | Learning rate |
|---|---|---|---|
| **Frozen** (reference, "rank 0") | linear head only | 7,690 | 0.01 |
| **LoRA r = 4** | head + LoRA on query/value, all 12 blocks | 155,146 | 0.001 |
| **LoRA r = 32** | head + LoRA on query/value, all 12 blocks | 1,187,338 | 0.001 |

Everything else is held identical per seed: training images, image order, head initialisation, augmentation, optimiser, batch size and stopping rule. Each learning rate was chosen by the same validation-only sweep.

**Decision rule, fixed before any LoRA run:** for each seed take the paired difference in test cross-entropy (rank 32 minus rank 4). If the mean of the 5 differences is smaller in size than their standard deviation, performance **levels off**.

## Results

Test cross-entropy (nats, lower is better), test balanced accuracy in brackets. Every run reported; each scored on the test set once.

| Seed | Frozen | LoRA r = 4 | LoRA r = 32 | r32 − r4 |
|---|---|---|---|---|
| 0 | 0.443 (84.4%) | 0.258 (92.0%) | 0.185 (94.3%) | −0.073 |
| 1 | 0.470 (82.4%) | 0.247 (91.2%) | 0.207 (92.7%) | −0.041 |
| 2 | 0.494 (81.0%) | 0.249 (90.3%) | 0.177 (94.2%) | −0.072 |
| 3 | 0.394 (85.8%) | 0.162 (95.0%) | 0.193 (94.6%) | +0.031 |
| 4 | 0.463 (83.1%) | 0.220 (93.1%) | 0.157 (95.2%) | −0.062 |
| **Mean ± SD** | **0.453 ± 0.037** | **0.227 ± 0.039** | **0.184 ± 0.018** | **−0.043 ± 0.043** |

- **LoRA vs frozen:** rank 4 is lower on all 5 seeds; mean paired gain 0.226 ± 0.024 nats.
- **Rank 32 vs rank 4:** lower on 4 of 5 seeds, but |mean| (0.0433) < SD (0.0433), so **levels off** under the pre-set rule.
- **Diminishing returns:** rank 4 recovers 84% of rank 32's gain over frozen with 1/8 of the adapter parameters.
- **Noise:** repeating a LoRA run with identical settings moved validation CE by up to 0.024 (non-deterministic GPU ops).

Figures: [`results/figures/`](results/figures/). Raw rows: [`results/final_results/`](results/final_results/).

## How It Works

```
 EuroSAT MS GeoTIFFs ──► tile centres ──► EPSG:3035 (km)          scripts/build_coords.py
                                │
 frozen ViT CLS features ──► same-class similarity vs distance    notebooks/preprocessing/CellSize.ipynb
                                │      plateau at 33 km × 1.25
                                ▼
                    41 km grid cells ──► StratifiedGroupKFold(7)  scripts/build_split.py
                    fold 0 test · fold 1 val · folds 2–6 train
                    + 41 km buffer between train and val/test
                                │
                                ▼
                  data/splits/split_v1.csv
                                │
              make_subset(pool, N = 20, seed)                     src/subset.py
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   Frozen head on        LoRA rank 4            LoRA rank 32       src/frozen.py · src/run.py
   cached 8-view         (live ViT)             (live ViT)         src/model.py · src/train.py
   CLS features
          │                     │                     │
          └──── best validation-CE checkpoint per run ┘
                                │
              scripts/evaluate_test.py (once, after the design was fixed)
```

1. **Locate every tile.** RGB JPEGs carry no location, so each tile's centre is read from its multispectral GeoTIFF and reprojected from its UTM zone to EPSG:3035 (equal-area, metres).
2. **Size the cells from the data.** Same-class frozen-feature similarity decays with distance and flattens at 33 km; × 1.25 margin gives **41 km** cells.
3. **Split whole cells.** `StratifiedGroupKFold` keeps cells intact and the class mix even; training tiles within 41 km of any val/test tile become `buffer` and are never used. Result: 7,318 train · 3,847 val · 3,922 test · 11,913 buffer.
4. **Draw the training subset.** 20 images per class per seed, round-robin across cells so picks are spread out; the same images for every condition.
5. **Train.** AdamW (wd 0.01), batch 32, early stopping on validation CE (patience 5, max 50 epochs), α = r, LoRA dropout 0.1, 8 right-angle orientations as augmentation. The frozen head trains on cached features of the same 8 views.
6. **Score the test set once.** The evaluation script re-scores validation first and refuses to continue if it doesn't match the recorded value (wrong checkpoint), then writes the test score and never scores a run twice.

## Repository Structure

```
src/                        The pipeline, shared by every condition (each file has a self-check: python -m src.<name>)
  data.py                   Reads any split CSV, serves (image, label); drops buffer rows
  transforms.py             Bicubic 64→224, normalise, the 8 orientations
  subset.py                 make_subset: N per class per seed, spread across grid cells
  model.py                  ViT + head + LoRA; init_head(seed); trainable-count assert
  train.py                  Training loop, early stopping, per-epoch checkpoints, exact resume
  run.py                    One LoRA run: config + seed → trained model → one results row
  frozen.py                 One frozen run on cached features
  metrics.py                Cross-entropy, balanced accuracy, class-averaged CE
scripts/                    One-off steps, run by hand from the repo root
  build_coords.py           GeoTIFF centres → data/coords.csv
  build_split.py            41 km cells → folds → buffer → checks → split_v1.csv
  pilot.py                  Frozen N and learning-rate pilots (validation only)
  evaluate_test.py          The one-time test evaluation
  export_histories.py       Per-epoch histories out of LoRA checkpoints (Colab)
  compare_outputs.py        All results figures + compare_summary.txt, from saved test logits
  make_split_temp.py        Placeholder random split used before the real one existed
configs/                    frozen.yaml, r4.yaml, r32.yaml (identical except condition, rank, lr)
notebooks/
  preprocessing/            CellSize, FingerprintsExtraction, 8ViewExtraction
  training/                 LoRASweepAndRuns_r4 / _r32 (Colab: sweep + final runs, with outputs)
data/
  coords.csv                Tile locations (lon/lat + EPSG:3035)
  splits/split_v1.csv       The spatial split (filename, label, lon, lat, cell_id, split)
results/
  final_results/            runs_frozen.csv, runs_r4.csv, runs_r32.csv (the 15 final runs)
  pilot/                    N and learning-rate pilot rows
  baseline_gate/            Week-5 gate runs (N = 50, kept separate)
  figures/                  All figures + all_runs_table.csv
runs/                       Per-run config, per-epoch history, results row, test logits (Frozen/, R4/, R32/, pilot/, gate/)
docs/                       Spec, decision logs (one per person), Colab + Drive setup
```

Raw images, cached features and checkpoints live on Google Drive and are git-ignored.

## Reproducing

### 1. Environment

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # pinned versions
```

### 2. Data

Download `EuroSAT_RGB.zip` and `EuroSAT_MS.zip` from [Zenodo record 7711810](https://zenodo.org/records/7711810) and unzip into `data/raw/`. The split is already committed; to rebuild it:

```bash
.venv/bin/python scripts/build_coords.py     # → data/coords.csv
.venv/bin/python scripts/build_split.py      # → data/splits/split_v1.csv (fails loudly if any check fails)
```

### Seeds

Seeds `0 1 2 3 4`, the same for every condition. Seed `s` fixes the training subset (`make_subset`), the head init (`init_head`) and the per-epoch shuffle (`seed * 1000 + epoch`, `src/train.py`), so seed `s` uses the same images in the same order in every condition. All other settings are in `configs/*.yaml`. Only `condition`, `rank` and `lr` differ between the files.

### 3. Check the pipeline

```bash
for m in transforms data subset metrics model train run frozen; do .venv/bin/python -m src.$m; done
```

### 4. Train

- **Frozen** (laptop, seconds per run; needs the cached 8-view features from `notebooks/preprocessing/8ViewExtraction.ipynb` in `data/fingerprints/`):
  ```bash
  .venv/bin/python -m src.frozen --seeds 0 1 2 3 4 --split data/splits/split_v1.csv \
      --out-root runs/Frozen --results results/final_results/runs_frozen.csv
  ```
- **LoRA** (Colab T4, ~25 min per final run): open `notebooks/training/LoRASweepAndRuns_r4.ipynb` or `_r32.ipynb` and run top to bottom. Checkpoints go to Drive after every epoch; a disconnected run resumes exactly. Setup: [`docs/colab-drive-setup.md`](docs/colab-drive-setup.md).

A finished run is skipped if rerun, so results rows are never duplicated.

### 5. Evaluate and plot

```bash
.venv/bin/python scripts/evaluate_test.py frozen    # also r4, r32 (needs data/raw/EuroSAT_RGB for LoRA)
.venv/bin/python scripts/compare_outputs.py         # → results/figures/compare_*.png + compare_summary.txt
```

Each run is scored on the test set once. Rerunning the evaluation skips runs that already have a test score.

### What to expect

Reproduction-ready, not bit-identical. GPU LoRA training uses non-deterministic ops, so rerunning with the same seed moves validation CE by up to about 0.024. The frozen runs and the split rebuild are deterministic. Steps 1–5 are all of it. There are no manual steps apart from downloading the data and opening the Colab notebooks.

## Compute

| | |
|---|---|
| Hardware | Google Colab, NVIDIA T4 16 GB |
| LoRA runs | 40 (30 learning-rate sweep, 10 final) |
| GPU time | ~23 h (18.7 h sweep, 4.4 h final) |
| Final run | ~28 min (r = 4), ~24 min (r = 32) |
| Frozen | one 12-min feature extraction, then seconds per run on a CPU |

## Limitations

- Results are conditional on N = 20, query/value placement, one split and 5 seeds.
- Both LoRA ranks chose the top learning rate of their grid (0.001).
- Validation is not buffered from test, so validation may be optimistic as a guide to test.
- 41 km is the median correlation range; some classes stay correlated out to 80–200 km.
- Single-label 640 m tiles, 64 → 224 upsampling, Europe only.

## Team

| | Area |
|---|---|
| **Dylan Teehan** | Data, tile coordinates, cell size, spatial split |
| **Edmund Phelan** | Transforms, ViT + LoRA, training loop, compute |
| **Patrick Walsh** | Frozen reference, metrics, pilots, test evaluation |

Design decisions and their reasons are logged in [`docs/decision-log/`](docs/decision-log/).

## Data, Model and Licences

- **EuroSAT** (Helber et al., 2019), Sentinel-2 imagery, [Zenodo 7711810](https://zenodo.org/records/7711810), MIT licence.
- **ViT-B/16** ([`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)), Dosovitskiy et al., 2021, Apache 2.0.
- **LoRA**: Hu et al., 2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)), via Hugging Face PEFT.

Developed as a group project for the Advanced AI module.
