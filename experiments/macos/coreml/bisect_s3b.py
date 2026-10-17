"""Bisect fp16 CPU mismatch inside layer 0 by cutting the real layer.forward at successive points (input = real stage-2 output)."""
import json, numpy as np, torch, torch.nn as nn, coremltools as ct
from common import OUT, load_tower
from harness import convert, UNITS, rel_err, BENCH
from modules import BlockFixed
tower = load_tower(); L = tower.layers[0]
sub = np.load(OUT / "oracle_mix_29s.npz")["sub"]
bf = BlockFixed(tower, 0, 1, 750)
CUTS = ["ff1", "pre_attn_norm", "attn", "post_attn_norm_resid", "lconv", "ff2", "norm_out"]
def make(cut):
    class M(nn.Module):
        def forward(s, h):
            c = L.gradient_clipping
            h = L.feed_forward1(h)
            if cut == "ff1": return h
            r = h
            n = L.norm_pre_attn(torch.clamp(h, -c, c))
            if cut == "pre_attn_norm": return n
            a, _ = L.self_attn(hidden_states=n, position_embeddings=bf.pe, attention_mask=bf.am_)
            if cut == "attn": return a
            h = L.norm_post_attn(torch.clamp(a, -c, c)) + r
            if cut == "post_attn_norm_resid": return h
            h = L.lconv1d(h)
            if cut == "lconv": return h
            h = L.feed_forward2(h)
            if cut == "ff2": return h
            return L.norm_out(torch.clamp(h, -c, c))
    return M().eval()
rows = {}
for cut in CUTS:
    m = make(cut)
    with torch.no_grad(): ref = m(torch.from_numpy(sub)).numpy()
    c = convert(m, (torch.from_numpy(sub),), f"bis3b_{cut}", precision="fp16", inputs=[ct.TensorType(name="x", shape=sub.shape)])
    if not c["ok"]: print("CUT", cut, "convert failed", c["error"][-300:]); continue
    row = {}
    for u in ("CPU_ONLY", "CPU_AND_GPU"):
        out = ct.models.MLModel(c["path"], compute_units=UNITS[u]).predict({"x": sub})
        row[u] = rel_err(list(out.values())[0], ref)
    row["ref_absmax"] = float(np.abs(ref).max())
    rows[cut] = row; print("CUT", cut, {u: round(v["rel_l2"], 5) for u, v in row.items() if u != "ref_absmax"}, "ref absmax", round(row["ref_absmax"], 1), flush=True)
(BENCH / "bisect_stage3_fp16_cpu_cuts.json").write_text(json.dumps(rows, indent=1))
