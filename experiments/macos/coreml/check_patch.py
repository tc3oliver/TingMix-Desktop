import numpy as np, torch
from common import OUT, load_tower
from modules import BlockFixed
from tingmix.models.large.encoder import run_layers
tower = load_tower(); sub = torch.from_numpy(np.load(OUT / "oracle_mix_29s.npz")["sub"])
with torch.no_grad():
    ref = run_layers(tower, sub, torch.tensor([750]), 0, 1, proj=False)
    for kw in (dict(rms_prescale=False), dict(rms_prescale=True)):
        tower2 = load_tower()
        out = BlockFixed(tower2, 0, 1, 750, **kw)(sub)
        print("PATCH", kw, "max_abs vs original HF layer", float((out - ref).abs().max()), "ref absmax", float(ref.abs().max()))
