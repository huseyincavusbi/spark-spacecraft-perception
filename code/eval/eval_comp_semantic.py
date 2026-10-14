"""Semantic evaluation of comp-n-seg: union masks per class vs GT component masks."""
import csv, os, random, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/weights_comp/best.pt"

model = YOLO(str(MODEL))
random.seed(42)
rows = list(csv.DictReader(open(SRC / "val.csv")))
sample = random.sample(rows, 2000)

ious = {"body": [], "panel": []}
prec = {"body": [], "panel": []}
rec = {"body": [], "panel": []}
for i, r in enumerate(sample):
    cls = r["Class"].strip()
    img_name = r["Image name"].strip()
    mask_name = r["Mask name"].strip()
    res = model.predict(str(SRC / "images" / cls / "val" / img_name),
                        imgsz=640, conf=0.1, verbose=False, device="mps")[0]
    H, W = res.orig_shape
    pred = {"body": np.zeros((H, W), np.uint8), "panel": np.zeros((H, W), np.uint8)}
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for m, c in zip(masks, clss):
            mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            name = "body" if c == 0 else "panel"
            pred[name] |= mm.astype(np.uint8)
    mk = np.array(Image.open(SRC / "mask" / cls / "val" / mask_name))
    R, B = mk[:, :, 0], mk[:, :, 2]
    gt = {"body": ((R > 128) & (B <= 128)).astype(np.uint8),
          "panel": ((B > 128) & (R <= 128)).astype(np.uint8)}
    for name in ["body", "panel"]:
        inter = (pred[name] & gt[name]).sum()
        union = (pred[name] | gt[name]).sum()
        if union > 0:
            ious[name].append(inter / union)
        if pred[name].sum() > 0:
            prec[name].append(inter / pred[name].sum())
        if gt[name].sum() > 0:
            rec[name].append(inter / gt[name].sum())
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/2000", flush=True)

for name in ["body", "panel"]:
    a = np.array(ious[name]); p = np.array(prec[name]); rr = np.array(rec[name])
    print(f"{name:6s}: IoU mean={a.mean():.4f} p10={np.percentile(a,10):.4f} | pixelP={p.mean():.4f} pixelR={rr.mean():.4f} | n={len(a)}")
all_iou = np.concatenate([np.array(ious["body"]), np.array(ious["panel"])])
print(f"MEAN semantic IoU (both classes): {all_iou.mean():.4f}")
