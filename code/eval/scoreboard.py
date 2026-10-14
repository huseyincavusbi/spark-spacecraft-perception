"""Multi-task scoreboard: classification + detection (from component union) + component seg.
Usage: python scoreboard.py <comp_model_path> [n_images]
"""
import csv, os, random, sys
from pathlib import Path
import numpy as np, cv2, ast
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
COMP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/weights_comp/best.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

comp = YOLO(str(COMP))
cls = YOLO(str(ROOT / "runs/classify/spark_cls/cls-n-224/weights/best.pt"))
cls_names = [n.replace(" ", "") for n in cls.names.values()]

random.seed(42)
rows = list(csv.DictReader(open(SRC / "val.csv")))
sample = random.sample(rows, N)

def norm(s): return s.strip().replace(" ", "")

iou_ts = np.arange(0.5, 1.0, 0.05)
cls_correct = 0
det_hits = {t: 0 for t in iou_ts}
seg_ious = {"body": [], "panel": []}
det_ious = []
n = 0
for r in sample:
    clsname = r["Class"].strip()
    img_name = r["Image name"].strip()
    mask_name = r["Mask name"].strip()
    img_path = SRC / "images" / clsname / "val" / img_name
    # ---- component prediction -> union masks + box
    res = comp.predict(str(img_path), imgsz=640, conf=0.1, verbose=False, device="mps")[0]
    H, W = res.orig_shape
    body = np.zeros((H, W), np.uint8); panel = np.zeros((H, W), np.uint8)
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for m, c in zip(masks, clss):
            mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            (body if c == 0 else panel)[:] |= mm.astype(np.uint8)
    union = (body | panel)
    if union.sum() > 0:
        ys, xs = np.where(union)
        pbox = (xs.min(), ys.min(), xs.max(), ys.max())
    else:
        pbox = None
    # ---- classifier
    cres = cls.predict(str(img_path), imgsz=224, verbose=False, device="mps")[0]
    pred_cls = cls_names[int(cres.probs.top1)]
    ok_cls = (pred_cls == norm(clsname))
    cls_correct += int(ok_cls)
    # ---- GT
    gx1, gy1, gx2, gy2 = ast.literal_eval(r["Bounding box"])
    if pbox is not None:
        inter = max(0, min(pbox[2], gx2) - max(pbox[0], gx1)) * max(0, min(pbox[3], gy2) - max(pbox[1], gy1))
        area_p = (pbox[2]-pbox[0])*(pbox[3]-pbox[1]); area_g = (gx2-gx1)*(gy2-gy1)
        biou = inter / (area_p + area_g - inter) if (area_p+area_g-inter) > 0 else 0
    else:
        biou = 0
    det_ious.append(biou)
    for t in iou_ts:
        if ok_cls and biou >= t:
            det_hits[t] += 1
    # ---- seg GT
    mk = np.array(Image.open(SRC / "mask" / clsname / "val" / mask_name))
    R, B = mk[:, :, 0], mk[:, :, 2]
    gts = {"body": ((R > 128) & (B <= 128)).astype(np.uint8),
           "panel": ((B > 128) & (R <= 128)).astype(np.uint8)}
    preds = {"body": body, "panel": panel}
    for k in ["body", "panel"]:
        u = (preds[k] | gts[k]).sum()
        if u > 0:
            seg_ious[k].append((preds[k] & gts[k]).sum() / u)
    n += 1
    if n % 500 == 0:
        print(f"  {n}/{N}", flush=True)

S_cls = cls_correct / n
S_det50 = det_hits[iou_ts[0]] / n
S_det = np.mean([det_hits[t] / n for t in iou_ts])
S_seg = np.mean([np.mean(seg_ious["body"]), np.mean(seg_ious["panel"])])
S_acc = 0.2 * S_cls + 0.4 * S_det + 0.4 * S_seg

print(f"\n=== SCOREBOARD (n={n}) ===")
print(f"S_cls (accuracy)      : {S_cls:.4f}")
print(f"S_det  AP50           : {S_det50:.4f}")
print(f"S_det  AP50-95        : {S_det:.4f}")
print(f"S_seg  mIoU (body/pan): {np.mean(seg_ious['body']):.4f} / {np.mean(seg_ious['panel']):.4f}  -> {S_seg:.4f}")
print(f"mean box IoU          : {np.mean(det_ious):.4f}")
print(f"S_acc = 0.2*cls + 0.4*det + 0.4*seg = {S_acc:.4f}")
