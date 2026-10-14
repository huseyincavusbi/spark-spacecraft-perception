#!/usr/bin/env python3
"""
SPARK-2022 -> YOLO conversion.

Input : data/raw/spark2022/{train,val,test}/*.jpg + {split}.csv (filename,class,bbox)
        bbox = [x1, y1, x2, y2] on 1024x1024 images
Output: data/yolo/spark2022/{images,labels}/{train,val,test} + data.yaml

Images are moved (not copied) into the YOLO layout to avoid duplication.
"""
import ast
import csv
import shutil
import os
from pathlib import Path

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data/raw/spark2022"
OUT = ROOT / "data/yolo/spark2022"
IMG_SIZE = 1024

CLASSES = sorted([
    "xmm_newton", "soho", "smart_1", "proba_3_ocs", "proba_3_csc", "proba_2",
    "lisa_pathfinder", "earth_observation_sat_1", "double_star", "debris", "cheops",
])
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def move_images(split: str):
    """Move extracted <split>/ into the YOLO images/<split>/ layout."""
    src = RAW / split
    dst = OUT / "images" / split
    if dst.exists():
        print(f"  images/{split} already in place")
        return
    if not src.exists():
        raise FileNotFoundError(f"missing extracted folder {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  moved {src.name}/ -> images/{split}/")


def convert_labels(split: str):
    lbl_dst = OUT / "labels" / split
    lbl_dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(RAW / f"{split}.csv") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            cls = CLASS_IDX[row["class"]]
            x1, y1, x2, y2 = ast.literal_eval(row["bbox"])
            cx = (x1 + x2) / 2 / IMG_SIZE
            cy = (y1 + y2) / 2 / IMG_SIZE
            w = (x2 - x1) / IMG_SIZE
            h = (y2 - y1) / IMG_SIZE
            (lbl_dst / f"{Path(fn).stem}.txt").write_text(
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            )
            n += 1
    return n


def main():
    for split in ("train", "val", "test"):
        print(f"[{split}]")
        move_images(split)
        n = convert_labels(split)
        print(f"  {n} labels")
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"\ndata.yaml written -> {OUT/'data.yaml'}")
    print(f"classes: {CLASSES}")


if __name__ == "__main__":
    main()
