"""Build YOLO eval set for SPARK-2026 val using Model A's 2022 class space.

- Maps 9 shared 2026 classes -> 2022 class indices
- Excludes VenusExpress (no 2022 equivalent)
- Pascal VOC xyxy -> YOLO normalized labels
- Symlinks images to avoid duplication
"""
import csv, ast
import os
from pathlib import Path
from PIL import Image

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
DST = ROOT / "data/yolo/spark2026_eval"

MAP = {
    "Cheops": 0, "LisaPathfinder": 4, "ObservationSat1": 3, "Proba2": 5,
    "Proba3": 6, "Proba3ocs": 7, "Smart1": 8, "Soho": 9, "XMM Newton": 10,
}
NAMES = ["cheops", "debris", "double_star", "earth_observation_sat_1",
         "lisa_pathfinder", "proba_2", "proba_3_csc", "proba_3_ocs",
         "smart_1", "soho", "xmm_newton"]

img_dir = DST / "images/val"
lbl_dir = DST / "labels/val"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(SRC / "val.csv")))
n_img = n_box = n_skip = 0
for r in rows:
    cls = r["Class"].strip()
    if cls not in MAP:
        n_skip += 1
        continue
    idx = MAP[cls]
    img_name = r["Image name"].strip()
    src_img = SRC / "images" / cls / "val" / img_name
    if not src_img.exists():
        print(f"MISSING: {src_img}")
        continue
    out_name = f"{cls.replace(' ', '_')}_{img_name}"
    link = img_dir / out_name
    if not link.exists():
        link.symlink_to(src_img)
    W, H = Image.open(src_img).size
    x1, y1, x2, y2 = ast.literal_eval(r["Bounding box"])
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    w, h = (x2 - x1) / W, (y2 - y1) / H
    with open(lbl_dir / (Path(out_name).stem + ".txt"), "w") as f:
        f.write(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    n_img += 1
    n_box += 1

yaml = DST / "data.yaml"
with open(yaml, "w") as f:
    f.write(f"path: {DST}\ntrain: images/val\nval: images/val\n")
    f.write("names:\n")
    for i, n in enumerate(NAMES):
        f.write(f"  {i}: {n}\n")

print(f"images: {n_img} | boxes: {n_box} | skipped (VenusExpress): {n_skip}")
print(f"yaml: {yaml}")
