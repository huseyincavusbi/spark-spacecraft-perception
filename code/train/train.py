#!/usr/bin/env python3
"""
SPARK training wrapper.

Usage:
    python code/train.py <name> [--epochs 100] [--model yolo26s.pt]
                         [--imgsz 640] [--batch 16] [--fraction 1.0] [--data ...]
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name (e.g. smoke, modelA-s640)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--data", default=str(ROOT / "data/yolo/spark2022/data.yaml"))
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device="mps",
        project=str(ROOT / "runs"),
        name=args.name,
        seed=42,
        patience=20,
        workers=4,
        exist_ok=True,
        plots=True,
    )
    m = model.val(data=args.data, split="val", device="mps")
    print("VAL mAP50-95:", round(float(m.box.map), 4))
    print("VAL mAP50:", round(float(m.box.map50), 4))


if __name__ == "__main__":
    main()
