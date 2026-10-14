#!/usr/bin/env python
"""Inference preview: diverse val images through the final comp20 model.

Picks images per satellite class (min/median/max body area) + a cluttered
scene, runs the final single model, saves raw / predicted / side-by-side
images + an overview grid for inspection.
"""
import os
import glob
import math
from collections import defaultdict

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SPARK = os.environ.get("SPARK_ROOT", str(Path(__file__).resolve().parents[2]))
MODEL = f"{SPARK}/runs/weights_comp20/runs/segment/spark/comp20-n-seg/weights/best.pt"
IMGS = f"{SPARK}/data/yolo/spark2026_comp/images/val"
LBLS = f"{SPARK}/data/yolo/spark2026_comp20/labels/val"
OUT = f"{SPARK}/inspection"
os.makedirs(OUT, exist_ok=True)

NAMES = [
    "cheops", "lisapathfinder", "observationsat1", "proba2", "proba3",
    "proba3ocs", "smart1", "soho", "venusexpress", "xmm_newton",
]


def poly_area(coords):
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


print("=== scanning val labels ===")
stats = {}
for lf in glob.glob(f"{LBLS}/*.txt"):
    stem = os.path.basename(lf)[:-4]
    sat_area = defaultdict(float)
    n_objs = 0
    with open(lf) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            cid = int(parts[0])
            n_objs += 1
            if cid % 2 == 0:
                sat_area[cid // 2] += poly_area(parts[1:])
    stats[stem] = {"area": dict(sat_area), "n": n_objs}

picks = {}
for sid, nm in enumerate(NAMES):
    cands = [(s, d["area"][sid]) for s, d in stats.items() if d["area"].get(sid, 0) > 0]
    if not cands:
        continue
    cands.sort(key=lambda t: t[1])
    for tag, (s, _) in [
        ("closeup", cands[-1]),
        ("mid", cands[len(cands) // 2]),
        ("small", cands[0]),
    ]:
        picks.setdefault(s, f"{nm}_{tag}")

by_n = sorted(stats.items(), key=lambda kv: -kv[1]["n"])
picks.setdefault(by_n[0][0], "cluttered")
picks.setdefault(by_n[1][0], "cluttered2")

print(f"{len(picks)} diverse images picked")

model = YOLO(MODEL)
stems = sorted(picks)
paths = [f"{IMGS}/{s}.jpg" for s in stems]
results = model.predict(
    paths, imgsz=640, conf=0.25, iou=0.7, retina_masks=True,
    verbose=False, device="mps",
)

compares = []
for i, (r, stem) in enumerate(zip(results, stems)):
    raw = np.array(Image.open(f"{IMGS}/{stem}.jpg").convert("RGB"))
    plotted = r.plot(labels=True, boxes=True, conf=True, font_size=20)[..., ::-1]
    tag = f"{i:02d}_{picks[stem]}"
    Image.fromarray(raw).save(f"{OUT}/{tag}_raw.jpg", quality=92)
    Image.fromarray(plotted).save(f"{OUT}/{tag}_pred.jpg", quality=92)
    sep = np.full((raw.shape[0], 8, 3), 255, dtype=np.uint8)
    comp = np.hstack([raw, sep, plotted])
    Image.fromarray(comp).save(f"{OUT}/{tag}_compare.jpg", quality=92)
    compares.append((tag, comp))
    n = len(r.boxes) if r.boxes is not None else 0
    print(f"{tag}: {n} detections, {stem}")

cols = 4
tw = 700
rows = math.ceil(len(compares) / cols)
thumbs = []
for tag, comp in compares:
    im = Image.fromarray(comp)
    h = int(im.height * tw / im.width)
    thumbs.append((tag, np.array(im.resize((tw, h)))))
th = thumbs[0][1].shape[0]
header = 34
grid = Image.new("RGB", (cols * tw, rows * (th + header)), (255, 255, 255))
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except OSError:
    font = ImageFont.load_default()
for i, (tag, t) in enumerate(thumbs):
    rr, cc = divmod(i, cols)
    y0 = rr * (th + header)
    grid.paste(Image.fromarray(t), (cc * tw, y0 + header))
    draw.text((cc * tw + 10, y0 + 6), tag, fill=(0, 0, 0), font=font)

grid.save(f"{OUT}/00_overview_grid.jpg", quality=90)

print("DONE →", OUT)
