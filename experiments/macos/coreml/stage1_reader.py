"""Stage 1: Reader only (LayerNorm+Linear, 4 conv blocks, 2-layer BiGRU, Linear). Fixed full-length input (no packing)."""
import sys, numpy as np, torch, torch.nn as nn, coremltools as ct
from common import OUT, load_reader
from modules import ReaderFull
from harness import convert, run_units, placement, record

reader, _ = load_reader()
mod = ReaderFull(reader)
res = {}
fx = "mix_29s"
o = np.load(OUT / f"oracle_{fx}.npz")
x, ref = o["feats"], {"logits": o["logits"]}
with torch.no_grad():
    assert torch.allclose(mod(torch.from_numpy(x)), torch.from_numpy(o["logits"]), atol=1e-5), "wrapper != Reader.forward"
for method in ("trace", "export"):
    for prec in ("fp32", "fp16"):
        key = f"{method}_{prec}"
        ex = (torch.from_numpy(x),)
        c = convert(mod, ex, f"s1_reader_T{x.shape[1]}", method=method, precision=prec, inputs=[ct.TensorType(name="feats", shape=x.shape)])
        res[key] = dict(convert=c)
        print(key, {k: v for k, v in c.items() if k != "error"}, (c.get("error") or "")[-600:])
        if c["ok"]:
            nm = ct.models.MLModel(c["path"]).get_spec().description.output[0].name
            res[key]["run"] = run_units(c["path"], {"feats": x}, {nm: o["logits"]}) if False else None
            m = ct.models.MLModel(c["path"], compute_units=ct.ComputeUnit.CPU_ONLY)
            out_name = list(m.predict({"feats": x}).keys())[0]
            res[key]["run"] = run_units(c["path"], {"feats": x}, {out_name: o["logits"]})
            res[key]["placement"] = placement(c["path"])
            for u, r in res[key]["run"].items():
                print(" ", u, r.get("median_ms"), r.get("parity"), r.get("error", ""))
            print("  placement", res[key]["placement"])
record("stage1_reader", dict(fixture=fx, input_shape=list(x.shape), results=res))
