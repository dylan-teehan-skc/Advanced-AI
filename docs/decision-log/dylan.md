# Decision Log: Dylan

Area: data and spatial split. Newest at the top.

## 14 Oct 2026 - Keep split_v1; validation and test not buffered from each other
- **Decision:** `data/splits/split_v1.csv` is the final split. The 41 km buffer stays between train and val/test only. Validation and test are not buffered from each other; this is reported as a limitation. (Written by Paddy with Dylan's agreement.)
- **Why:** About a third of validation images are within 41 km of a test image (closest 0.6 km), so validation can look like test and validation scores may be slightly optimistic as a guide to test. Neither is used for training, so it does not leak into what the model learns, and test is still evaluated once after the freeze. Buffering val/test would shrink both sets further or need a new split days before the results checkpoint, after the buffer had already removed 62% of the training pool.
- **Affects:** all runs, the limitations section of the write-up, the oral.

## 7 Oct 2026 - Uneven-class check uses each class's own size
- **Decision:** A class is uneven in test if it has under half of 15% of its own class size, i.e., Pasture is flagged under 150 and Forest under 225.
- **Why:** Classes range from 2,000 to 3,000 images. A flat cutoff would flag Pasture every time.
- **Affects:** split checks in the split script, Decision 8 in the plan.

## 4 Oct 2026 - Tile locations come from the MS GeoTIFF centre point
- **Decision:** Read each `.tif`'s centre, convert from its own UTM zone to lat/lon and to EPSG:3035 (metres). Saved to `data/coords.csv` by `src/build_coords.py`.
- **Why:** The RGB `.jpg` files have no location. The tiles span 10 UTM zones, so they need one shared system before gridding.
- **Affects:** everything in the spatial split.

## 3 Oct 2026 - Data downloaded from Zenodo, RGB and MS
- **Decision:** `EuroSAT_RGB.zip` and `EuroSAT_MS.zip` from Zenodo record 7711810, unzipped into `data/raw/`.
- **Why:** Zenodo is the source the EuroSAT GitHub page now points to. The old DFKI links are marked deprecated.
- **Affects:** provenance section of the write-up.

## 3 Oct 2026 - We predict the land-use label, not the location
- **Decision:** Keep the approved task: classify each image into one of 10 land types. Locations are only used to split the data.
- **Why:** It's what the proposal gate approved. Predicting location would need new approval days before the baseline gate.
- **Affects:** scope of the whole project.
