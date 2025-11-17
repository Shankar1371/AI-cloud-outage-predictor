import torch
import numpy as np
from pathlib import Path

def convert(pt_file):
    out = pt_file.with_suffix(".npz")
    pkg = torch.load(pt_file, map_location="cpu")
    X = pkg["X"]
    y = pkg["y"]

    # ensure numpy
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    np.savez(out, X=X, y=y)
    print(f"[OK] wrote {out}")

root = Path("data/processed/openstack_DEPL")

for split in ["train", "val", "test"]:
    pt_file = root / f"{split}.pt"
    if pt_file.exists():
        convert(pt_file)
    else:
        print(f"[WARN] missing {pt_file}")
