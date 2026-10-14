"""SPARK-2026 Stream 1 - Run B: YOLO26s-seg fine-tuned from COCO-seg (control arm).

2xT4 DDP, wandb callback, dataset copied to local disk (mount I/O fix).
"""
import glob, os, shutil, subprocess, sys, json
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "wandb"], check=True)

print("=== GPUs ===", flush=True)
os.system("nvidia-smi")

# locate dataset
cands = glob.glob("/kaggle/input/**/data.yaml", recursive=True)
print("data.yaml candidates:", cands, flush=True)
src = Path(cands[0]).parent
print("dataset root:", src, flush=True)

# copy to local disk
dst = Path("/kaggle/working/ds")
if not dst.exists():
    print("copying dataset to local disk...", flush=True)
    shutil.copytree(src, dst)
    print("copy done", flush=True)

# rewrite data.yaml (local paths) + val subset for per-epoch validation
import yaml
d = yaml.safe_load(open(dst / "data.yaml"))
d["path"] = str(dst)
d["train"] = "images/train"
d["val"] = "images/val"
yaml.safe_dump(d, open(dst / "data.yaml", "w"))
print("data.yaml:", d, flush=True)

# val subset: 400/class = 4k images
sub = dst / "images" / "val_sub"
sub.mkdir(exist_ok=True)
from collections import Counter
cnt = Counter()
for f in sorted((dst / "images" / "val").iterdir()):
    cls = f.name.split("_")[0]
    if cnt[cls] < 400:
        link = sub / f.name
        if not link.exists():
            link.symlink_to(f)
        cnt[cls] += 1
print("val_sub:", sum(cnt.values()), flush=True)
d_sub = dict(d); d_sub["val"] = "images/val_sub"
yaml.safe_dump(d_sub, open(dst / "data_sub.yaml", "w"))

# ---- wandb login (key from Kaggle Secret) ----
os.environ["WANDB_ENTITY"] = "huseyincavusbi-home"
os.environ["WANDB_PROJECT"] = "spark"
wandb_ok = False
try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("wandb key from Kaggle Secret", flush=True)
    wandb.login(key=key, relogin=True)
    wandb_ok = True
    print("wandb enabled -> project 'spark'", flush=True)
except Exception as e:
    print("wandb disabled:", str(e)[:120], flush=True)

from ultralytics.utils import SETTINGS
SETTINGS.update({"wandb": True})
print("SETTINGS['wandb']:", SETTINGS["wandb"], flush=True)

from ultralytics import YOLO
from ultralytics.utils.callbacks import wb as wbmod
print("wandb integration active:", wbmod.wb is not None, flush=True)

model = YOLO("yolo26s-seg.pt")  # COCO-seg init (control arm)

results = model.train(
    data=str(dst / "data_sub.yaml"),
    epochs=30, imgsz=640, batch=32, device=[0, 1], workers=8, time=10.5,
    project="spark", name="2026seg-B-s640", patience=8, save_period=5,
    plots=True,
)

# full val on 20k
best = Path("runs/segment/spark/2026seg-B-s640/weights/best.pt")
print("best exists:", best.exists(), flush=True)
model2 = YOLO(str(best))
m = model2.val(data=str(dst / "data.yaml"), split="val", imgsz=640, device=0)
print("FULL VAL mAP50-95:", round(float(m.box.map), 4), flush=True)
print("FULL VAL mAP50:", round(float(m.box.map50), 4), flush=True)
print("FULL VAL seg mAP50-95:", round(float(m.seg.map), 4), flush=True)
print("FULL VAL seg mAP50:", round(float(m.seg.map50), 4), flush=True)

if wandb_ok:
    wandb.finish()
