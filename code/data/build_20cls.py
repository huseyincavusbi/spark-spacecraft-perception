"""Build 20-class decomposition labels from the clean 2-class component labels.
Class id = sat_idx*2 + comp (0=body, 1=panel).
"""
import csv
from multiprocessing import Pool
import os
from pathlib import Path
from PIL import Image

ROOT = Path(os.environ.get("SPARK_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "data/yolo/spark2026_comp_clean"
DST = ROOT / "data/yolo/spark2026_comp20"
SATS = ["Cheops", "LisaPathfinder", "ObservationSat1", "Proba2", "Proba3",
        "Proba3ocs", "Smart1", "Soho", "VenusExpress", "XMM Newton"]
SAT2IDX = {s: i for i, s in enumerate(SATS)}
NAMES = []
for s in SATS:
    NAMES += [f"{s.lower().replace(' ', '_')}_body", f"{s.lower().replace(' ', '_')}_panel"]

def sat_of(fname):
    # filenames: e.g. XMM_Newton_image_00001_img.jpg ; prefixes with underscores
    for s in SATS:
        if fname.startswith(s.replace(" ", "_") + "_"):
            return s
    return None

def process(job):
    split, name = job
    src = SRC / "labels" / split / name
    sat = sat_of(name)
    if sat is None:
        return "nomatch"
    base = SAT2IDX[sat] * 2
    out = []
    for line in src.read_text().splitlines():
        parts = line.split()
        cid = int(parts[0])
        out.append(f"{base + cid} " + " ".join(parts[1:]))
    (DST / "labels" / split / name).write_text("\n".join(out) + "\n")
    return "ok"

def main():
    for split in ["train", "val"]:
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ["train", "val"]:
        for f in sorted((SRC / "labels" / split).glob("*.txt")):
            jobs.append((split, f.name))
    print("jobs:", len(jobs), flush=True)
    from collections import Counter
    res = Counter()
    with Pool(8) as p:
        for r in p.imap_unordered(process, jobs, chunksize=256):
            res[r] += 1
    print("results:", dict(res), flush=True)
    with open(DST / "data.yaml", "w") as f:
        f.write(f"path: {DST}\ntrain: labels/train\nval: labels/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print("data.yaml written (20 classes)", flush=True)

if __name__ == "__main__":
    main()
