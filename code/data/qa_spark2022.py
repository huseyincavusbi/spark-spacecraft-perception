#!/usr/bin/env python3
"""SPARK-2022 YOLO dataset QA: integrity + visual grid."""
import random
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
V = ROOT / "data/yolo/spark2022"
CLASSES = sorted([
    "xmm_newton", "soho", "smart_1", "proba_3_ocs", "proba_3_csc", "proba_2",
    "lisa_pathfinder", "earth_observation_sat_1", "double_star", "debris", "cheops",
])

for split in ("train", "val", "test"):
    imgs = {p.stem for p in (V / f"images/{split}").iterdir()}
    lbls = {p.stem for p in (V / f"labels/{split}").iterdir()}
    missing = imgs - lbls
    empty = [l for l in lbls if (V / f"labels/{split}/{l}.txt").stat().st_size == 0]
    print(f"{split}: imgs={len(imgs)} labels={len(lbls)} missing={len(missing)} empty={len(empty)}")

# QA grid: 8 random samples with boxes
random.seed(3)
samples = random.sample(list((V / "images/train").iterdir()), 8)
tiles = []
for p in samples:
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    lab = V / "labels/train" / (p.stem + ".txt")
    for line in lab.read_text().strip().splitlines():
        c, cx, cy, bw, bh = line.split()
        x1 = int((float(cx) - float(bw) / 2) * w); y1 = int((float(cy) - float(bh) / 2) * h)
        x2 = int((float(cx) + float(bw) / 2) * w); y2 = int((float(cy) + float(bh) / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, CLASSES[int(c)], (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(cv2.resize(img, (320, 320)))
grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite("/tmp/spark2022_qa.png", grid)
print("QA grid -> /tmp/spark2022_qa.png")
