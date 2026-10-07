# Decision Log: Eddie

Area: model and LoRA training. Newest at the top.

## 7 Oct 2026 - Bicubic resize, normalise with mean/std 0.5
- **Decision:** Resize 64→224 with bicubic. Normalise with mean 0.5, std 0.5.
- **Why:** 0.5/0.5 is what the checkpoint's `preprocessor_config.json` uses. It sets no resize method, so HF would default to bilinear; bicubic is our choice, applied the same everywhere.
- **Affects:** `src/transforms.py`, every condition's input.

## 7 Oct 2026 - Right-angle turns, not arbitrary-angle rotation
- **Decision:** Augment with 90° rotations + flip only. No `RandomRotation` with arbitrary angles.
- **Why:** Arbitrary angles can't be cached for the frozen head, so it would get different augmentation from LoRA. They also add black corners and blur that test images never have.
- **Affects:** Decision 8, `src/transforms.py`, augmentation section of the write-up.

## 7 Oct 2026 - Train augmentation is always one of the 8 cached views
- **Decision:** `train_transform` resizes, then applies one random orientation of 8 (4 rotations × optional horizontal flip). `eight_views` applies the same 8. View 0 = `eval_transform`.
- **Why:** Guarantees the frozen head (cached 8 views) and LoRA (live augmentation) see identical augmentation (Decision 2). Randomness comes from torch, so the run seed controls it.
- **Affects:** `src/transforms.py`, Paddy's feature extraction (save views in this order).
