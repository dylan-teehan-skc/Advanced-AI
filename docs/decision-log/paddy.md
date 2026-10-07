# Decision Log: Paddy

Area: frozen reference, metrics and analysis. Newest at the top.

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
