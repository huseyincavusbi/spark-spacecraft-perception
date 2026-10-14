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

# NO COPY — train straight from mount; val subset as txt list
import yaml
random.seed(42)
val_imgs = sorted((src / "images/val").glob("*.jpg"))
sub = random.sample(val_imgs, min(4000, len(val_imgs)))
val_list = Path("/kaggle/working/val_subset.txt")
val_list.write_text("\n".join(str(p) for p in sub))
print("val subset:", len(sub), flush=True)
d = yaml.safe_load(open(src / "data.yaml"))
d["path"] = str(src)
d_sub = dict(d); d_sub["val"] = str(val_list)
yaml.safe_dump(d_sub, open("/kaggle/working/data_sub.yaml", "w"))
yaml.safe_dump(d, open("/kaggle/working/data_full.yaml", "w"))

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
    data="/kaggle/working/data_sub.yaml",
    epochs=30, imgsz=640, batch=32, device=[0, 1], workers=8, time=10.5,
    project="spark", name="2026seg-B-s640", patience=8, save_period=5,
    plots=True,
)

# full val on 20k
best = Path("runs/segment/spark/2026seg-B-s640/weights/best.pt")
print("best exists:", best.exists(), flush=True)
model2 = YOLO(str(best))
m = model2.val(data="/kaggle/working/data_full.yaml", split="val", imgsz=640, device=0)
print("FULL VAL mAP50-95:", round(float(m.box.map), 4), flush=True)
print("FULL VAL mAP50:", round(float(m.box.map50), 4), flush=True)
print("FULL VAL seg mAP50-95:", round(float(m.seg.map), 4), flush=True)
print("FULL VAL seg mAP50:", round(float(m.seg.map50), 4), flush=True)

if wandb_ok:
    wandb.finish()
