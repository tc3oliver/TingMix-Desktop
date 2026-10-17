"""Bisect the fp32 CPU_ONLY subsample mismatch: cut the graph after each op and compare CPU_ONLY vs GPU vs torch."""
import numpy as np, torch, torch.nn as nn, coremltools as ct
from common import OUT, load_tower
from harness import convert, UNITS, rel_err
tower = load_tower(); s = tower.subsample_conv_projection
mel = np.load(OUT / "oracle_mix_29s.npz")["mel"]

def cut(k):
    class M(nn.Module):
        def forward(self, m):
            h = m.unsqueeze(1)
            h = s.layer0.conv(h)
            if k == 1: return h
            h = s.layer0.norm(h.permute(0, 2, 3, 1))
            if k == 2: return h
            h = s.layer0.act(h.permute(0, 3, 1, 2))
            if k == 3: return h
            h = s.layer1.conv(h)
            if k == 4: return h
            h = s.layer1.norm(h.permute(0, 2, 3, 1))
            if k == 5: return h
            h = s.layer1.act(h.permute(0, 3, 1, 2))
            b, _, t, _ = h.shape
            h = h.permute(0, 2, 3, 1).reshape(b, t, -1)
            if k == 6: return h
            return s.input_proj_linear(h)
    return M().eval()
for k in range(1, 8):
    m = cut(k)
    with torch.no_grad(): ref = m(torch.from_numpy(mel)).numpy()
    c = convert(m, (torch.from_numpy(mel),), f"bis{k}", precision="fp32", inputs=[ct.TensorType(name="mel", shape=mel.shape)])
    row = {}
    for u in ("CPU_ONLY", "CPU_AND_GPU"):
        out = ct.models.MLModel(c["path"], compute_units=UNITS[u]).predict({"mel": mel})
        row[u] = round(rel_err(list(out.values())[0], ref)["rel_l2"], 6)
    print(k, ref.shape, row, flush=True)
