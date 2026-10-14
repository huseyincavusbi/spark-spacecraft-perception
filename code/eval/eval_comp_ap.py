"""Component-level AP: merge predicted fragments -> components, match vs GT components."""
import csv, os, random
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/raw/spark2026/extracted/stream-1-train-val"
model = YOLO(str(ROOT / "runs/weights_comp/best.pt"))
random.seed(42)
rows = list(csv.DictReader(open(SRC / "val.csv")))
sample = random.sample(rows, 2000)

# collect detections (per image per class: list of (conf, mask)) and gts (list of masks)
data = {"body": [], "panel": []}
for i, r in enumerate(sample):
    cls = r["Class"].strip()
    img_name = r["Image name"].strip()
    mask_name = r["Mask name"].strip()
    res = model.predict(str(SRC / "images" / cls / "val" / img_name),
                        imgsz=640, conf=0.05, verbose=False, device="mps")[0]
    H, W = res.orig_shape
    # union masks per class + per-instance confidences
    unions = {"body": np.zeros((H, W), np.uint8), "panel": np.zeros((H, W), np.uint8)}
    confs = {"body": [], "panel": []}
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        cfs = res.boxes.conf.cpu().numpy()
        for m, c, cf in zip(masks, clss, cfs):
            mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            name = "body" if c == 0 else "panel"
            unions[name] |= mm.astype(np.uint8)
            confs[name].append((mm.astype(np.uint8), float(cf)))
    # GT
    mk = np.array(Image.open(SRC / "mask" / cls / "val" / mask_name))
    R, B = mk[:, :, 0], mk[:, :, 2]
    gts = {"body": ((R > 128) & (B <= 128)).astype(np.uint8),
           "panel": ((B > 128) & (R <= 128)).astype(np.uint8)}
    for name in ["body", "panel"]:
        # predicted components from union
        n, lab, stats, _ = cv2.connectedComponentsWithStats(unions[name], 8)
        preds = []
        for ci in range(1, n):
            if stats[ci, cv2.CC_STAT_AREA] < 10:
                continue
            comp = (lab == ci)
            cf = max((c for m, c in confs[name] if (m & comp.astype(np.uint8)).any()), default=0.0)
            preds.append((cf, comp))
        # GT components
        ng, labg, statsg, _ = cv2.connectedComponentsWithStats(gts[name], 8)
        gtlist = [ (labg == gi) for gi in range(1, ng) if statsg[gi, cv2.CC_STAT_AREA] >= 10 ]
        data[name].append((preds, gtlist))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/2000", flush=True)

def iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return inter / union if union else 0.0

def compute_ap(items, iouv):
    # items: list of (preds[(conf,mask)], gts[mask])
    aps = []
    for thr in iouv:
        # gather per-image matching
        TP = FP = FN = 0
        all_dets = []  # (conf, tp?)  — need global sorting: use per-image greedy matching first
        for preds, gts in items:
            used = [False] * len(gts)
            order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
            for pi in order:
                cf, pm = preds[pi]
                best_iou, best_j = 0.0, -1
                for j, gm in enumerate(gts):
                    if used[j]:
                        continue
                    v = iou(pm, gm)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0 and best_iou >= thr:
                    used[best_j] = True
                    all_dets.append((cf, 1))
                else:
                    all_dets.append((cf, 0))
            FN += sum(1 for u in used if not u)
        if not all_dets:
            aps.append(0.0)
            continue
        all_dets.sort(key=lambda x: -x[0])
        tp = np.cumsum([d[1] for d in all_dets])
        fp = np.cumsum([1 - d[1] for d in all_dets])
        npos = TP + FN + tp[-1]
        rec = tp / max(npos, 1)
        prec = tp / np.maximum(tp + fp, 1e-9)
        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = prec[rec >= t].max() if (rec >= t).any() else 0.0
            ap += p / 101
        aps.append(ap)
    return aps

iouv = np.arange(0.5, 1.0, 0.05)
for name in ["body", "panel"]:
    aps = compute_ap(data[name], iouv)
    print(f"{name:6s}: AP50={aps[0]:.4f} | AP50-95={np.mean(aps):.4f} | per-thr={[round(a,3) for a in aps]}")
