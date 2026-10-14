"""Stage 2: zero-shot 2022 -> 2026 transfer check (Model A on 2026 val)."""
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
w = ROOT / "runs/kaggle_modelA/runs/detect/spark/modelA-s640/weights/best.pt"
data = ROOT / "data/yolo/spark2026_eval/data.yaml"
model = YOLO(str(w))
m = model.val(data=str(data), split="val", imgsz=640, device="mps", verbose=True)
print("TRANSFER 2022->2026 mAP50-95:", round(float(m.box.map), 4))
print("TRANSFER 2022->2026 mAP50:", round(float(m.box.map50), 4))
