"""Convert SPARK-2026 train+val to YOLO-seg format."""
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
DST = ROOT / "data/yolo/spark2026_seg"

CLASS_MAP = {'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3,
             'Proba2': 4, 'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9}
NAMES = list(CLASS_MAP.keys())

def poly_from_mask(mask_path):
    mk = np.array(Image.open(mask_path))
    binary = (mk > 128).any(axis=2).astype(np.uint8)
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = [c for c in cnts if cv2.contourArea(c) >= 20]
    if not cnts:
        return None, 0
    merged = 1 if len(cnts) > 1 else 0
    if merged:
        cnts = [cv2.convexHull(np.vstack(cnts))]
    poly = cv2.approxPolyDP(cnts[0], 1.5, True).reshape(-1, 2)
    if len(poly) < 3:
        return None, 0
    return poly, merged

def process(job):
    split, cls, img_name, mask_name, idx = job
    src_img = SRC / "images" / cls / split / img_name
    src_mask = SRC / "mask" / cls / split / mask_name
    if not src_img.exists():
        return ("missing_img", 0)
    if not src_mask.exists():
        return ("missing_mask", 0)
    poly, merged = poly_from_mask(src_mask)
    if poly is None:
        return ("empty_mask", 0)
    W, H = Image.open(src_img).size
    out_name = f"{cls.replace(' ', '_')}_{img_name}"
    img_link = DST / "images" / split / out_name
    if not img_link.exists():
        img_link.symlink_to(src_img)
    pts = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in poly)
    with open(DST / "labels" / split / (Path(out_name).stem + ".txt"), "w") as f:
        f.write(f"{idx} {pts}\n")
    return ("ok", merged)

def main():
    for split in ["train", "val"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs = []
    for split, csv_name in [("train", "train.csv"), ("val", "val.csv")]:
        rows = list(csv.DictReader(open(SRC / csv_name)))
        for r in rows:
            cls = r["Class"].strip()
            jobs.append((split, cls, r["Image name"].strip(), r["Mask name"].strip(), CLASS_MAP[cls]))
        print(f"{split}: {len(rows)} rows", flush=True)

    print(f"total jobs: {len(jobs)}", flush=True)
    results = []
    with Pool(8) as p:
        for i, res in enumerate(p.imap_unordered(process, jobs, chunksize=64)):
            results.append(res)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)

    from collections import Counter
    c = Counter(r[0] for r in results)
    merged = sum(r[1] for r in results)
    print(f"results: {dict(c)} | merged(multi-contour): {merged}", flush=True)

    with open(DST / "data.yaml", "w") as f:
        f.write(f"path: {DST}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print("data.yaml written", flush=True)

if __name__ == "__main__":
    main()
