"""Scoreboard for the 20-class single model: seg (mIoU) + cls (from classes) + det (union+pad)."""
import csv, os, random, sys, ast
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
MODEL = ROOT / "runs/weights_comp20/runs/segment/spark/comp20-n-seg/weights/best.pt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DELTA = 0.02

model = YOLO(str(MODEL))
random.seed(42)
rows = list(csv.DictReader(open(SRC / "val.csv")))
sample = random.sample(rows, N)

iou_ts = np.arange(0.5, 1.0, 0.05)
cls_correct = 0
det_hits = {t: 0 for t in iou_ts}
seg_ious = {"body": [], "panel": []}
det_ious = []
for i, r in enumerate(sample):
    clsname = r["Class"].strip(); img_name = r["Image name"].strip(); mask_name = r["Mask name"].strip()
    img_path = SRC / "images" / clsname / "val" / img_name
    res = model.predict(str(img_path), imgsz=640, conf=0.1, verbose=False, device="mps")[0]
    H, W = res.orig_shape
    body = np.zeros((H, W), np.uint8); panel = np.zeros((H, W), np.uint8)
    pred_sat = None
    if res.masks is not None and len(res.boxes) > 0:
        masks = res.masks.data.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
        pred_sat = int(clss[best_i]) // 2  # satellite index from the class
        for m, c in zip(masks, clss):
            mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            (body if c % 2 == 0 else panel)[:] |= mm.astype(np.uint8)
    SATS = ["Cheops", "LisaPathfinder", "ObservationSat1", "Proba2", "Proba3",
            "Proba3ocs", "Smart1", "Soho", "VenusExpress", "XMM Newton"]
    ok_cls = pred_sat is not None and SATS[pred_sat] == clsname
    cls_correct += int(ok_cls)
    union = (body | panel)
    pbox = None
    if union.sum() > 0:
        ys, xs = np.where(union)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        dx = (x2 - x1) * DELTA; dy = (y2 - y1) * DELTA
        pbox = (x1 - dx, y1 - dy, x2 + dx, y2 + dy)
    gx1, gy1, gx2, gy2 = ast.literal_eval(r["Bounding box"])
    if pbox:
        inter = max(0, min(pbox[2], gx2) - max(pbox[0], gx1)) * max(0, min(pbox[3], gy2) - max(pbox[1], gy1))
        ap = (pbox[2]-pbox[0])*(pbox[3]-pbox[1]); ag = (gx2-gx1)*(gy2-gy1)
        biou = inter / (ap + ag - inter) if (ap+ag-inter) > 0 else 0
    else:
        biou = 0
    det_ious.append(biou)
    for t in iou_ts:
        if ok_cls and biou >= t:
            det_hits[t] += 1
    mk = np.array(Image.open(SRC / "mask" / clsname / "val" / mask_name))
    R, B = mk[:, :, 0], mk[:, :, 2]
    gts = {"body": ((R > 128) & (B <= 128)).astype(np.uint8), "panel": ((B > 128) & (R <= 128)).astype(np.uint8)}
    preds = {"body": body, "panel": panel}
    for k in ["body", "panel"]:
        u = (preds[k] | gts[k]).sum()
        if u > 0:
            seg_ious[k].append((preds[k] & gts[k]).sum() / u)
    if (i+1) % 500 == 0: print(f"  {i+1}/{N}", flush=True)

S_cls = cls_correct / N
S_det50 = det_hits[iou_ts[0]] / N
S_det = np.mean([det_hits[t] / N for t in iou_ts])
S_seg = np.mean([np.mean(seg_ious["body"]), np.mean(seg_ious["panel"])])
S_acc = 0.2 * S_cls + 0.4 * S_det + 0.4 * S_seg
print(f"\n=== COMP20 SINGLE-MODEL SCOREBOARD (n={N}) ===")
print(f"S_cls (from component classes): {S_cls:.4f}")
print(f"S_det AP50: {S_det50:.4f} | AP50-95: {S_det:.4f}")
print(f"S_seg mIoU: body {np.mean(seg_ious['body']):.4f} / panel {np.mean(seg_ious['panel']):.4f} -> {S_seg:.4f}")
print(f"mean box IoU: {np.mean(det_ious):.4f}")
print(f"S_acc = {S_acc:.4f}")
