"""Clean component labels: ~1 polygon per component (holes filled, no fragmentation)."""
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
DST = ROOT / "data/yolo/spark2026_comp_clean"
NAMES = ["body", "panel"]

def comp_polys(binary):
    polys = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        comp = (lab == i).astype(np.uint8)
        for c in cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(poly) >= 3:
                polys.append(poly)
    return polys

def process(job):
    split, cls, img_name, mask_name = job
    src_img = SRC / "images" / cls / split / img_name
    src_mask = SRC / "mask" / cls / split / mask_name
    if not src_img.exists() or not src_mask.exists():
        return ("missing", 0, 0)
    mk = np.array(Image.open(src_mask))
    R, B = mk[:, :, 0], mk[:, :, 2]
    masks = {0: ((R > 128) & (B <= 128)).astype(np.uint8),
             1: ((B > 128) & (R <= 128)).astype(np.uint8)}
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
            if cid == 0: nb += 1
            else: npan += 1
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
    print(f"instances/image: body={nb/80000:.2f} panel={npan/80000:.2f}", flush=True)
    with open(DST / "data.yaml", "w") as f:
        f.write(f"path: {DST}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print("data.yaml written", flush=True)

if __name__ == "__main__":
    main()
