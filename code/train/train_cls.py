"""Tiny classifier: YOLO26n-cls on SPARK-2026 (h2/MPS)."""
import os
from pathlib import Path
from ultralytics.utils import SETTINGS
SETTINGS.update({"wandb": False})
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
DATA = str(ROOT / "data/cls/spark2026")

model = YOLO("yolo26n-cls.pt")
model.train(
    data=DATA,
    epochs=12, imgsz=224, batch=64, device="mps", workers=4,
    project="spark_cls", name="cls-n-224", exist_ok=True,
)
m = model.val(data=DATA, split="val", imgsz=224, device="mps")
print("TOP1:", round(float(m.top1), 4))
print("TOP5:", round(float(m.top5), 4))
