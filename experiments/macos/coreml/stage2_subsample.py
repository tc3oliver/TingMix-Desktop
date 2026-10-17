import numpy as np
from common import OUT, load_tower
from modules import SubsampleFull
from stages import drive
from harness import record
tower = load_tower()
o = np.load(OUT / "oracle_mix_29s.npz")
mel, sub = o["mel"], o["sub"]
res = drive("stage2", SubsampleFull(tower), f"s2_subsample_F{mel.shape[1]}", {"mel": mel}, sub)
record("stage2_subsample", dict(fixture="mix_29s", input_shape=list(mel.shape), results=res))
