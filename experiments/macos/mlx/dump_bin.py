"""Export oracle npz (ref_torch.py) as raw f32 files for the C++ driver: out/ref-bin/<id>.{mel,feats,logits,nvalid}.f32"""
import numpy as np
from common import *

d = OUT / "ref-bin"
d.mkdir(exist_ok=True)
for p in sorted((OUT / "ref").glob("*.npz")):
    z = np.load(p)
    z["mel"][0].astype("<f4").tofile(d / f"{p.stem}.mel.f32")
    z["feats"].astype("<f4").tofile(d / f"{p.stem}.feats.f32")
    z["logits"].astype("<f4").tofile(d / f"{p.stem}.logits.f32")
    np.array([z["mask"].sum()], dtype="<f4").tofile(d / f"{p.stem}.nvalid.f32")
print(sorted(x.stem for x in (OUT / "ref").glob("*.npz")))
