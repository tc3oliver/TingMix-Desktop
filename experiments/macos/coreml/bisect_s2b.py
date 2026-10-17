"""Variants of layer0 (conv -> LN(channel) -> relu) vs fp32 CPU_ONLY correctness."""
import numpy as np, torch, torch.nn as nn, coremltools as ct
from common import OUT, load_tower
from harness import convert, UNITS, rel_err
tower = load_tower(); s = tower.subsample_conv_projection; l = s.layer0
mel = np.load(OUT / "oracle_mix_29s.npz")["mel"]
def mk(v):
    class M(nn.Module):
        def forward(self, m):
            h = l.conv(m.unsqueeze(1))
            if v == "orig": return l.act(l.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            if v == "relu_nhwc": return l.act(l.norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            if v == "clamp": return torch.clamp(l.norm(h.permute(0, 2, 3, 1)), min=0).permute(0, 3, 1, 2)
            if v == "manual_ln":  # LN over channel dim=1 without permute
                mu = h.mean(1, keepdim=True); var = ((h - mu) ** 2).mean(1, keepdim=True)
                return torch.relu((h - mu) * torch.rsqrt(var + l.norm.eps) * l.norm.weight.view(1, -1, 1, 1))
            if v == "norm_only_nchw": return l.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return M().eval()
for v in ("orig", "relu_nhwc", "clamp", "manual_ln", "norm_only_nchw"):
    m = mk(v)
    with torch.no_grad(): ref = m(torch.from_numpy(mel)).numpy()
    c = convert(m, (torch.from_numpy(mel),), f"bisb_{v}", precision="fp32", inputs=[ct.TensorType(name="mel", shape=mel.shape)])
    row = {}
    for u in ("CPU_ONLY", "CPU_AND_GPU"):
        out = ct.models.MLModel(c["path"], compute_units=UNITS[u]).predict({"mel": mel})
        row[u] = round(rel_err(list(out.values())[0], ref)["rel_l2"], 6)
    print("VAR", v, row, flush=True)
