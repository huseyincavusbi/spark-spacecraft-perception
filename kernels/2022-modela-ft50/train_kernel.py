"""SPARK 2022 — Model A continued fine-tune (ft40) — 2xT4 DDP + wandb.

Continued fine-tune from Model A: epochs 50, lr0 0.003, batch 64 total
(32/GPU), device [0,1], patience 10, graceful stop at 11.5h, wandb callback.
"""
import glob
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

print("cwd:", os.getcwd())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:800])

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

from ultralytics import YOLO

CLASSES = ['cheops', 'debris', 'double_star', 'earth_observation_sat_1',
           'lisa_pathfinder', 'proba_2', 'proba_3_csc', 'proba_3_ocs',
           'smart_1', 'soho', 'xmm_newton']

# ---- locate dataset (new Kaggle mount layout) ----
cands = (
    glob.glob("/kaggle/input/datasets/*/*/spark2022")
    + glob.glob("/kaggle/input/*/spark2022")
    + glob.glob("/kaggle/input/*/*/spark2022")
    + glob.glob("/kaggle/input/*/*/*/spark2022")
)
base = next((Path(c) for c in cands if (Path(c) / "images").is_dir()), None)
assert base is not None, f"SPARK dataset not found; candidates: {cands}"
print("dataset base:", base)
print("train imgs:", len(list((base / 'images/train').glob('*.jpg'))))
print("val imgs:", len(list((base / 'images/val').glob('*.jpg'))))

# ---- fast per-epoch val subset (2k) ----
random.seed(42)
val_imgs = sorted((base / "images/val").glob("*.jpg"))
sub = random.sample(val_imgs, min(2000, len(val_imgs)))
val_list = Path("/kaggle/working/val_subset.txt")
val_list.write_text("\n".join(str(p) for p in sub))
print("val subset:", len(sub))

names = "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
yaml_sub = Path("/kaggle/working/data_sub.yaml")
yaml_sub.write_text(f"path: {base}\ntrain: images/train\nval: {val_list}\nnames:\n{names}")
yaml_full = Path("/kaggle/working/data_full.yaml")
yaml_full.write_text(f"path: {base}\ntrain: images/train\nval: images/val\nnames:\n{names}")

# ---- train (project='spark' keeps the wandb project clean) ----
# ---- enable ultralytics wandb integration (default is False in 8.4.x) ----
from ultralytics.utils import SETTINGS
SETTINGS.update({"wandb": True})
print("ultralytics SETTINGS['wandb']:", SETTINGS["wandb"])

wg = (glob.glob("/kaggle/input/*/modelA_best.pt")
      + glob.glob("/kaggle/input/datasets/*/*/modelA_best.pt")
      + glob.glob("/kaggle/input/*/*/modelA_best.pt"))
print("weights candidates:", wg, flush=True)
assert wg, "modelA_best.pt not found in /kaggle/input"
model = YOLO(wg[0])
from ultralytics.utils.callbacks import wb as wbmod
print("wandb integration active:", wbmod.wb is not None)
model.train(
    data=str(yaml_sub),
    epochs=50,
    imgsz=640,
    batch=64,
    workers=4,
    device=[0, 1],
    lr0=0.003,
    time=10.5,
    project="spark",
    name="modelA-ft50",
    seed=42,
    patience=10,
    save_period=5,
    plots=True,
    exist_ok=True,
)

# ---- ensure runs land in the output dir regardless of cwd ----
cwd_runs = Path("spark")
out_runs = Path("/kaggle/working/spark")
if cwd_runs.exists() and cwd_runs.resolve() != out_runs.resolve():
    shutil.copytree(cwd_runs, out_runs, dirs_exist_ok=True)
    print("copied runs ->", out_runs)
print("runs present:", list(Path('/kaggle/working/spark').rglob('*.pt'))[:6])

# ---- final full validation (22k) ----
m = model.val(data=str(yaml_full), split="val", device=0)
print("FULL VAL mAP50-95:", round(float(m.box.map), 4))
print("FULL VAL mAP50:", round(float(m.box.map50), 4))

if wandb_ok:
    wandb.finish()
