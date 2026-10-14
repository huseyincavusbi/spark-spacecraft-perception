"""Smoke test: YOLO26n-seg on component labels (h2/MPS)."""
from ultralytics.utils import SETTINGS
SETTINGS.update({"wandb": False})  # h2 runs: no wandb

import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
D = str(ROOT / "data/yolo/spark2026_comp_smoke/data.yaml")
model = YOLO("yolo26n-seg.pt")
model.train(
    data=D, epochs=5, imgsz=640, batch=16, device="mps", workers=4,
    project="spark_smoke", name="comp-n-seg", exist_ok=True, plots=True,
)
m = model.val(data=D, split="val", imgsz=640, device="mps", verbose=True)
print("SMOKE BOX  mAP50-95:", round(float(m.box.map), 4))
print("SMOKE MASK mAP50-95:", round(float(m.seg.map), 4))
