"""Build 2-class component labels (body/panel) from raw SPARK-2026 masks.

- components made mutually exclusive (body & ~panel, panel & ~body) -> no overlap
- connectedComponentsWithStats for pixel-accurate filtering (keeps thin 1px lines)
"""
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
DST = ROOT / "data/yolo/spark2026_comp"
NAMES = ["body", "panel"]

def _polys_of(m):
    out = []
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append(poly)
    return out

def comp_polys(binary):
    """Split a component mask into hole-free pieces, return polygons."""
    polys = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    stack = [(lab == i).astype(np.uint8) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 20]
    while stack:
        m = stack.pop()
        if m.sum() < 20:
            continue
        ff = m.copy()
        cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
        holes = (ff == 0)
        if not holes.any():
            polys.extend(_polys_of(m))
            continue
        ys, xs = np.where(holes)
        xm = int(np.median(xs))
        left = m.copy(); left[:, xm:] = 0
        right = m.copy(); right[:, :xm] = 0
        if left.sum() == m.sum() or right.sum() == m.sum():
            ym = int(np.median(ys))
            top = m.copy(); top[ym:, :] = 0
            bot = m.copy(); bot[:ym, :] = 0
            if top.sum() == m.sum() or bot.sum() == m.sum():
                polys.extend(_polys_of(m))  # give up: accept filled contour
                continue
            stack.extend([top, bot])
        else:
            stack.extend([left, right])
    return polys

def process(job):
    split, cls, img_name, mask_name = job
    src_img = SRC / "images" / cls / split / img_name
    src_mask = SRC / "mask" / cls / split / mask_name
    if not src_img.exists() or not src_mask.exists():
        return ("missing", 0, 0)
    mk = np.array(Image.open(src_mask))
    R, B = mk[:, :, 0], mk[:, :, 2]
    body_raw = (R > 128) & (B <= 128)
    panel_raw = (B > 128) & (R <= 128)
    masks = {0: (body_raw & ~panel_raw).astype(np.uint8),
             1: (panel_raw & ~body_raw).astype(np.uint8)}
    W, H = Image.open(src_img).size
    out_name = f"{cls.replace(' ', '_')}_{img_name}"
    link = DST / "images" / split / out_name
    if not link.exists():
        link.symlink_to(src_img)
    lines, nb, npan = [], 0, 0
    for cid, binary in masks.items():
        for poly in comp_polys(binary):
            pts = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in poly)
            lines.append(f"{cid} {pts}")
            if cid == 0:
                nb += 1
            else:
                npan += 1
    if not lines:
        return ("empty", 0, 0)
    with open(DST / "labels" / split / (Path(out_name).stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return ("ok", nb, npan)

def main():
    for split in ["train", "val"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, csv_name in [("train", "train.csv"), ("val", "val.csv")]:
        rows = list(csv.DictReader(open(SRC / csv_name)))
        for r in rows:
            jobs.append((split, r["Class"].strip(), r["Image name"].strip(), r["Mask name"].strip()))
        print(f"{split}: {len(rows)} rows", flush=True)
    print(f"total jobs: {len(jobs)}", flush=True)
    results = []
    with Pool(8) as p:
        for i, res in enumerate(p.imap_unordered(process, jobs, chunksize=64)):
            results.append(res)
            if (i + 1) % 20000 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    from collections import Counter
    c = Counter(r[0] for r in results)
    nb = sum(r[1] for r in results)
    npan = sum(r[2] for r in results)
    print(f"results: {dict(c)} | body instances: {nb} | panel instances: {npan}", flush=True)
    with open(DST / "data.yaml", "w") as f:
        f.write(f"path: {DST}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print("data.yaml written", flush=True)

if __name__ == "__main__":
    main()
