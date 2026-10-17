"""Activation magnitude per layer (torch fp32): fp16 RMSNorm squares overflow above 256."""
import numpy as np, torch
from common import OUT, load_tower
tower = load_tower()
sub = torch.from_numpy(np.load(OUT / "oracle_mix_29s.npz")["sub"]); T = sub.shape[1]
from tingmix.models.large.encoder import _mask_and_pos
am, pe = _mask_and_pos(tower, sub, torch.tensor([T]))
h = sub
print("input absmax", float(h.abs().max()))
with torch.no_grad():
    for i, l in enumerate(tower.layers):
        a = l.feed_forward1(h); m1 = float(a.abs().max())
        h = l(h, attention_mask=am, position_embeddings=pe)
        print(f"layer {i:2d}: after ff1 absmax {m1:9.1f}  out absmax {float(h.abs().max()):9.1f}  out rms-sq max {float((h**2).mean(-1).max()):9.1f}")
    print("final proj absmax", float(tower.output_proj(h).abs().max()))
