import numpy as np, torch
from common import OUT, load_tower
from modules import BlockFixed
from stages import drive
from harness import record
from tingmix.models.large.encoder import run_layers
tower = load_tower()
o = np.load(OUT / "oracle_mix_29s.npz")
sub = o["sub"]; T = sub.shape[1]
with torch.no_grad():
    ref = run_layers(tower, torch.from_numpy(sub), torch.tensor([T]), 0, 1, proj=False).numpy()  # real tower code path
res = drive("stage3", BlockFixed(tower, 0, 1, T), f"s3_block0_T{T}", {"h": sub}, ref)
record("stage3_block", dict(fixture="mix_29s", input_shape=list(sub.shape), results=res))
