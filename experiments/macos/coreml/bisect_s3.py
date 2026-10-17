"""Bisect the fp16 CPU_ONLY block mismatch by converting sub-modules of layer 0 separately."""
import json, numpy as np, torch, torch.nn as nn, coremltools as ct
from common import OUT, load_tower
from harness import convert, UNITS, rel_err, BENCH
from modules import BlockFixed, FixedAttn
tower = load_tower(); L = tower.layers[0]
T = 750
sub = np.load(OUT / "oracle_mix_29s.npz")["sub"]
bf = BlockFixed(tower, 0, 1, T)  # patches attention
class Wrap(nn.Module):
    def __init__(s, f): super().__init__(); s.f = f
    def forward(s, x): return s.f(x)
parts = {
  "ff1": lambda x: L.feed_forward1(x),
  "rmsnorm_pre_attn": lambda x: L.norm_pre_attn(x),
  "attn": lambda x: L.self_attn(hidden_states=x, position_embeddings=bf.pe, attention_mask=bf.am_)[0],
  "lconv": lambda x: L.lconv1d(x),
  "ff2": lambda x: L.feed_forward2(x),
}
rows = {}
for name, f in parts.items():
    m = Wrap(f).eval()
    with torch.no_grad(): ref = m(torch.from_numpy(sub)).numpy()
    c = convert(m, (torch.from_numpy(sub),), f"bis3_{name}", precision="fp16", inputs=[ct.TensorType(name="x", shape=sub.shape)])
    if not c["ok"]: rows[name] = c; print("PART", name, "convert failed", c["error"][-200:]); continue
    row = {}
    for u in ("CPU_ONLY", "CPU_AND_GPU"):
        out = ct.models.MLModel(c["path"], compute_units=UNITS[u]).predict({"x": sub})
        row[u] = rel_err(list(out.values())[0], ref)
    rows[name] = row; print("PART", name, {u: round(v["rel_l2"], 5) for u, v in row.items()}, flush=True)
(BENCH / "bisect_stage3_fp16_cpu.json").write_text(json.dumps(rows, indent=1))
