"""Evaluate Model A on the SPARK-2022 test split (22k unseen)."""
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
w = ROOT / "runs/kaggle_modelA/runs/detect/spark/modelA-s640/weights/best.pt"
data = ROOT / "data/yolo/spark2022/data.yaml"
model = YOLO(str(w))
m = model.val(data=str(data), split="test", device="mps", verbose=True)
print("TEST mAP50-95:", round(float(m.box.map), 4))
print("TEST mAP50:", round(float(m.box.map50), 4))
