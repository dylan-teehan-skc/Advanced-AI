"""Read each EuroSAT MS GeoTIFF's centre point and save tile locations to data/coords.csv.

Each tile is stored in its own UTM zone, so every centre is reprojected to
lat/lon (EPSG:4326) and to EPSG:3035 (equal-area Europe, metres) for gridding.
"""
from pathlib import Path
import csv

import rasterio
from pyproj import Transformer

RAW = Path("data/raw")
MS_DIR, RGB_DIR = RAW / "EuroSAT_MS", RAW / "EuroSAT_RGB"
OUT = Path("data/coords.csv")

to_ll, to_3035 = {}, {}  # one Transformer per source CRS
rows, missing_rgb = [], []

for tif in sorted(MS_DIR.glob("*/*.tif")):
    label, stem = tif.parent.name, tif.stem
    if not (RGB_DIR / label / f"{stem}.jpg").exists():
        missing_rgb.append(stem)
    with rasterio.open(tif) as src:
        x, y = src.transform * (src.width / 2, src.height / 2)
        crs = src.crs.to_string()
    if crs not in to_ll:
        to_ll[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        to_3035[crs] = Transformer.from_crs(crs, "EPSG:3035", always_xy=True)
    lon, lat = to_ll[crs].transform(x, y)
    ex, ny = to_3035[crs].transform(x, y)
    rows.append([stem, label, f"{lon:.6f}", f"{lat:.6f}", f"{ex:.1f}", f"{ny:.1f}", crs])

with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "lon", "lat", "x3035", "y3035", "src_crs"])
    w.writerows(rows)

print(f"tiles: {len(rows)}  UTM zones: {len(to_ll)}  missing RGB match: {len(missing_rgb)}")
print(f"wrote {OUT}")
