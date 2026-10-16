# Decision Log: Paddy

Area: frozen reference, metrics and analysis. Newest at the top.

## 14 Oct 2026 - Frozen learning rate = 0.01
- **Decision:** The frozen condition keeps lr = 0.01 (the gate placeholder), now chosen by the pilot rule: lowest mean validation CE at N = 20.
- **Why:** 5 seeds each: lr 0.001 gave CE 0.740 +/- 0.017; 0.003 gave 0.582 +/- 0.023; 0.01 gave 0.559 +/- 0.026; 0.03 gave 0.606 +/- 0.026. The best is inside the grid, not at an edge. 0.001 and 0.003 were still improving at the 50-epoch cap, but the cap is part of the shared training budget for every condition, so the pick is the best rate within that budget. The lr 0.01 runs reproduced the N pilot's N = 20 runs to every printed digit (same seeds), which confirms the frozen path is deterministic. Results in `results/pilot_lr.csv`.
- **Affects:** `lr` in `configs/frozen.yaml`, the final frozen runs. The LoRA conditions get their own rates from Eddie's sweep.

## 14 Oct 2026 - N = 20 images per class (amends the N pilot rule)
- **Decision:** N = 20 per class (200 training images) for every condition. This amends the rule below, which picked N = 10. The amendment was made after the frozen N pilot and before any LoRA run.
- **Why:** Frozen pilot, lr 0.01, 5 seeds, validation: N = 10 gave CE 0.744 +/- 0.034 and balanced accuracy 73.8%; N = 20 gave 0.559 +/- 0.026 and 80.2%; N = 50 gave 0.424 +/- 0.019 and 85.1%; N = 100 gave 0.346 and 88.1%; N = 200 gave 0.302 and 90.0%. Both 10 and 20 are in the 70-85% band. 10 is too small in practice: an epoch is 4 batches, so patience 5 allows only 20 updates without improvement and LoRA at low learning rates could stop before it gets going. 20 doubles that, stays clearly limited data (about 1% of each class) and sits mid-band, leaving room above the frozen reference. No LoRA results existed, so the change cannot favour either answer to the rank question. Results in `results/pilot_n.csv`.
- **Affects:** `n_per_class: 20` in all three configs, the frozen learning-rate sweep (`python scripts/pilot.py lr --n 20`), Eddie's LoRA learning-rate sweep.

## 14 Oct 2026 - Training subset spread across grid cells (make_subset)
- **Decision:** `make_subset(pool, n, seed)` lives in `src/subset.py` and replaces the random stand-in in `src/run.py`. For each class the seed shuffles the 41 km cells and the images inside each cell, and the pick goes round the cells taking one image from each until it has n. Paddy took this over from Dylan (agreed 14 Oct).
- **Why:** The training pool is clustered: one cell holds 300 of SeaLake's 714 images, 251 of PermanentCrop's 952 and 231 of HerbaceousVegetation's 693. A plain random pick of 50 per class gets only 0.54 distinct cells per image (spread pick: 0.95), so much of n goes on near-copies, by a different amount for each seed. Spreading follows the same reasoning as the spatial split. It also makes subsets nested (the n = 10 pick is the start of the n = 20 pick), so the N pilot compares more data, not a different draw. Self-check: `python -m src.subset`.
- **Affects:** every run (frozen and LoRA use the same images per seed), the N pilot, the maximum N (532, River). The gate runs used the old random pick and are not reused.

## 14 Oct 2026 - N pilot rule fixed before running: accuracy band 70-85%
- **Decision:** N = the smallest of 10, 20, 50, 100, 200 images per class whose mean frozen validation balanced accuracy (5 seeds) is between 70% and 85%. The frozen learning rate is then the lowest mean validation CE of 0.001, 0.003, 0.01, 0.03 at that N. `scripts/pilot.py`, results in `results/pilot_n.csv` and `results/pilot_lr.csv`.
- **Why:** Limited labelled data is part of the question, and a frozen reference near the ceiling would leave LoRA no room to differ. Fixing the rule before seeing results stops N being picked to favour an answer. Validation only; test is never loaded.
- **Affects:** `n_per_class` in all three configs, the frozen `lr`, Eddie's LoRA learning-rate sweep (needs N first).

## 9 Oct 2026 - Gate reference runs kept separate from the final runs
- **Decision:** The baseline-gate frozen runs (N = 50, placeholder LR 0.01, validation only) go to `runs/gate/` and `results/gate_runs.csv`, not `runs/` and `results/runs.csv`.
- **Why:** Their settings are provisional. A run is skipped if its folder already has `row.json`, so in `runs/` they would silently stand in for the real frozen runs after the pilot.
- **Affects:** `results/gate_runs.csv`, `runs/gate/`, the frozen runs after the design freeze.

## 9 Oct 2026 - Frozen trainer: one cached view per image per epoch
- **Decision:** `src/frozen.py` trains the head on cached features with the same subset (`make_subset`), order (`epoch_order`), head (`init_head`), optimiser and early stopping as LoRA. Each epoch, each image gets one of its 8 views at random (own generator, seed 1,000,000 + seed x 1000 + epoch). Validation uses view 0.
- **Why:** Same augmentation distribution as LoRA (uniform over the same 8 orientations). An exact per-image match isn't possible, because LoRA draws orientations inside DataLoader workers. The separate generator keeps view draws off the order's random stream.
- **Affects:** `src/frozen.py`, the pairing claim in the write-up (same distribution, not the same draw per image).

## 7 Oct 2026 - Metrics: cross-entropy primary, balanced accuracy secondary, one shared file
- **Decision:** `src/metrics.py` scores every run. Primary metric: mean cross-entropy over images, in nats (random guessing over 10 classes = ln 10 = 2.303). Secondary: balanced accuracy (accuracy within each class, averaged over classes). Fallback primary if the test set turns out uneven: class-averaged cross-entropy. Computed in float64 from logits, accepting numpy or torch.
- **Why:** Cross-entropy uses the probability given to the correct class, so it also measures confidence and can pick up smaller differences between conditions than accuracy, which only checks the top guess. Balanced accuracy is easier to read and stops a large class dominating. One file means every condition is scored the same way. The self-check (`python -m src.metrics`) checks hand-worked cases, agrees with scikit-learn's `log_loss` and `balanced_accuracy_score`, and gives the same cross-entropy as PyTorch's `F.cross_entropy`, so training, best-epoch choice and reported results all use one definition.
- **Affects:** `src/frozen.py`, `src/run.py` (both use it for validation and later test), the results table and figure, the checkpoint choice (lowest validation CE).

## 7 Oct 2026 - 8-view features for all 27,000 images, on the pinned versions
- **Decision:** Cache all 8 orientations of every image (not only the training pool) with `eight_views()` from `src/transforms.py`, on the `requirements.txt` versions. `features_8view.npy` is the only feature file from now on; its view 0 replaces `fingerprints_view0.npy`.
- **Why:** No wait on the split and no re-extraction if it changes. Same code as LoRA's augmentation (Decision 2). View 0 matches the 6 Oct file (mean cosine 1.00000, min 0.99988), so the cell-size analysis stands.
- **Affects:** `notebooks/preprocessing/8ViewExtraction.ipynb`, the linear-head trainer, Drive `aai-project/features/`.

## 6 Oct 2026 - Features computed in half precision, stored as float16
- **Decision:** Run the frozen ViT under fp16 autocast and save the features as float16.
- **Why:** Much faster on a T4 and half the file size. Rounding is at about the third significant digit.
- **Affects:** feature extraction notebooks, the tolerance for the cached-vs-live check (to agree with Eddie).

## 6 Oct 2026 - Feature = CLS token after the final layernorm, no pooler
- **Decision:** Save `last_hidden_state[:, 0]` from `ViTModel(add_pooling_layer=False)`, not `pooler_output`.
- **Why:** It is the vector the classifier in `ViTForImageClassification` reads, so the frozen condition is rank 0 of the LoRA model. The pooler adds a dense + tanh layer the LoRA conditions never pass through.
- **Affects:** feature extraction notebooks, the frozen head's input, the cached-vs-live check.
