"""SPARK-2026 component seg (body/panel) - YOLO26n-seg - 2xT4 + wandb.

Dataset trick: images come from the mounted spark2026-seg dataset; the 2-class
labels come from the small spark2026-comp-labels dataset. Both copied to local disk.
"""
import glob, os, random, shutil, subprocess, sys
from pathlib import Path

print("cwd:", os.getcwd(), flush=True)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:800], flush=True)

# ---- deps ----
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "wandb"], check=True)

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

# ---- locate datasets ----
roots = [Path(c) for c in glob.glob("/kaggle/input/datasets/*/*/") + glob.glob("/kaggle/input/*/")]
seg_root = next((r for r in roots if (r / "images/train").is_dir()), None)
comp_root = next((r for r in roots if (r / "labels/train").is_dir() and not (r / "images").is_dir()), None)
print("seg_root:", seg_root, flush=True)
print("comp_root:", comp_root, flush=True)
assert seg_root is not None and comp_root is not None, f"datasets not found; roots={roots}"

# ---- local copy: images from seg, labels from comp ----
dst = Path("/kaggle/working/ds")
if not dst.exists():
    print("copying images + comp labels to local disk...", flush=True)
    for sub in ["images/train", "images/val"]:
        shutil.copytree(seg_root / sub, dst / sub)
    for sub in ["labels/train", "labels/val"]:
        shutil.copytree(comp_root / sub, dst / sub)
    print("copy done", flush=True)

import yaml
SATS = ["Cheops", "LisaPathfinder", "ObservationSat1", "Proba2", "Proba3",
        "Proba3ocs", "Smart1", "Soho", "VenusExpress", "XMM Newton"]
NAMES20 = {}
for _i, _s in enumerate(SATS):
    NAMES20[2 * _i] = f"{_s.lower().replace(' ', '_')}_body"
    NAMES20[2 * _i + 1] = f"{_s.lower().replace(' ', '_')}_panel"
d = {"path": str(dst), "train": "images/train", "val": "images/val", "names": NAMES20}
yaml.safe_dump(d, open(dst / "data.yaml", "w"))

random.seed(42)
val_imgs = sorted((dst / "images/val").glob("*.jpg"))
sub = random.sample(val_imgs, min(4000, len(val_imgs)))
val_list = Path("/kaggle/working/val_subset.txt")
val_list.write_text("\n".join(str(p) for p in sub))
print("val subset:", len(sub), flush=True)
d_sub = dict(d); d_sub["val"] = str(val_list)
yaml.safe_dump(d_sub, open("/kaggle/working/data_sub.yaml", "w"))

# ---- enable ultralytics wandb integration (default is False in 8.4.x) ----
from ultralytics.utils import SETTINGS
SETTINGS.update({"wandb": True})
print("SETTINGS['wandb']:", SETTINGS["wandb"], flush=True)

from ultralytics import YOLO
from ultralytics.utils.callbacks import wb as wbmod
print("wandb integration active:", wbmod.wb is not None, flush=True)

model = YOLO("yolo26n-seg.pt")  # compact: ~2.6M params (efficiency play)
model.train(
    data="/kaggle/working/data_sub.yaml",
    epochs=30, imgsz=640, batch=64, device=[0, 1], workers=8, time=10.5,
    project="spark", name="comp20-n-seg", patience=8, save_period=5,
    seed=42, plots=True, exist_ok=True,
)

# ---- final full val (20k) ----
best = Path("runs/segment/spark/comp20-n-seg/weights/best.pt")
print("best exists:", best.exists(), flush=True)
model2 = YOLO(str(best))
m = model2.val(data=str(dst / "data.yaml"), split="val", imgsz=640, device=0, verbose=True)
print("FULL VAL box mAP50-95:", round(float(m.box.map), 4), flush=True)
print("FULL VAL box mAP50:", round(float(m.box.map50), 4), flush=True)
print("FULL VAL mask mAP50-95:", round(float(m.seg.map), 4), flush=True)
print("FULL VAL mask mAP50:", round(float(m.seg.map50), 4), flush=True)

if wandb_ok:
    wandb.finish()
