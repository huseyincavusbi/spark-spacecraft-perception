"""TTA eval of Model B on the 2026 seg val (20k)."""
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
m = YOLO(str(ROOT / "runs/weights_B/best.pt"))
r = m.val(
    data=str(ROOT / "data/yolo/spark2026_seg/data.yaml"),
    split="val", imgsz=640, device="mps", augment=True, verbose=True,
)
print("TTA BOX  mAP50-95:", round(float(r.box.map), 4))
print("TTA BOX  mAP50:   ", round(float(r.box.map50), 4))
print("TTA MASK mAP50-95:", round(float(r.seg.map), 4))
print("TTA MASK mAP50:   ", round(float(r.seg.map50), 4))
