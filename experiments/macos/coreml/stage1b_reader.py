"""Stage 1b: Reader with EnumeratedShapes / RangeDim, padded-input behaviour, greedy-id agreement (fp16, ALL)."""
import numpy as np, torch, coremltools as ct
from common import OUT, load_reader
from modules import ReaderFull
from harness import convert, run_units, placement, record

reader, _ = load_reader()
mod = ReaderFull(reader)
o29, o08 = (np.load(OUT / f"oracle_{n}.npz") for n in ("mix_29s", "mix_08s"))
x29, x08 = o29["feats"], o08["feats"]
ids = lambda l: l.argmax(-1)
agree = lambda ref: (lambda out: dict(greedy_id_agree=float((ids(next(iter(out.values()))[:, : ref.shape[1]]) == ids(ref)).mean())))
res = {}
# greedy id agreement for the fixed-shape fp16/fp32 models
for prec in ("fp32", "fp16"):
    p = OUT / f"s1_reader_T750.trace.{prec}.mlpackage"
    res[f"fixed750_{prec}"] = run_units(p, {"feats": x29}, None, n=5, extra=agree(o29["logits"]))
# enumerated shapes
for kind, shape in (("enum", ct.EnumeratedShapes(shapes=[(1, 43, 1536), (1, 188, 1536), (1, 750, 1536)], default=(1, 750, 1536))),
                    ("range", ct.Shape(shape=(1, ct.RangeDim(16, 760, default=750), 1536)))):
    c = convert(mod, (torch.from_numpy(x29),), f"s1_reader_{kind}", method="trace", precision="fp16", inputs=[ct.TensorType(name="feats", shape=shape)])
    r = dict(convert=c)
    print(kind, {k: v for k, v in c.items() if k != "error"}, (c.get("error") or "")[-500:])
    if c["ok"]:
        r["run_188"] = run_units(c["path"], {"feats": x08}, None, n=5, extra=agree(o08["logits"]), units={"ALL": ct.ComputeUnit.ALL, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY})
        r["run_750"] = run_units(c["path"], {"feats": x29}, None, n=5, extra=agree(o29["logits"]), units={"ALL": ct.ComputeUnit.ALL, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY})
        print(" ", {k: (v.get("median_ms"), v.get("greedy_id_agree"), v.get("error")) for k, v in r["run_188"].items()})
        print(" ", {k: (v.get("median_ms"), v.get("greedy_id_agree"), v.get("error")) for k, v in r["run_750"].items()})
    res[kind] = r
# padded input on the 750 model: 188 valid frames + zeros (no packing => BiGRU backward pass sees the padding)
xp = np.zeros_like(x29); xp[:, :188] = x08
with torch.no_grad():
    lp = mod(torch.from_numpy(xp)).numpy()[:, :188]
pad = dict(torch_padded_vs_exact_greedy_agree=float((ids(lp) == ids(o08["logits"])).mean()),
           torch_padded_vs_exact_max_abs=float(np.abs(lp - o08["logits"]).max()))
print("padded", pad)
res["padded_188_in_750"] = pad
record("stage1b_reader_dynamic", dict(results=res))
