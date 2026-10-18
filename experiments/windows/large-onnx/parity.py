"""ONNX Runtime vs the CPU fp32 oracle dumps (experiments/macos/reference/out), every fixture chunk, dynamic F.

    uv run python parity.py [out/tingmix-large.onnx] [--provider CPUExecutionProvider] [--json out/parity-cpu.json]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import OUT, REF, rel_err

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=str(OUT / "tingmix-large.onnx"))
ap.add_argument("--provider", default="CPUExecutionProvider")
ap.add_argument("--json")
a = ap.parse_args()
so = ort.SessionOptions()
sess = ort.InferenceSession(a.model, so, providers=[a.provider] if a.provider.startswith("CPU") else [a.provider, "CPUExecutionProvider"])
print("providers", sess.get_providers())
rows = []
for f in sorted(REF.glob("*.npz")):
    d = np.load(f)
    k = 0
    while f"c{k}_mel" in d.files:
        mel = d[f"c{k}_mel"][:, d[f"c{k}_mel_mask"][0]]
        ref = d[f"c{k}_logits"]
        t0 = time.perf_counter()
        lg = sess.run(None, {"mel": mel})[0][0]
        dt = time.perf_counter() - t0
        e = rel_err(lg, ref)
        row = dict(clip=f.stem, chunk=k, frames=int(mel.shape[1]), T=int(lg.shape[0]), ids_equal=bool((lg.argmax(-1) == ref.argmax(-1)).all()),
                   id_agreement=float((lg.argmax(-1) == ref.argmax(-1)).mean()), seconds=round(dt, 3), **e)
        print(row)
        rows.append(row)
        k += 1
if a.json:
    Path(a.json).write_text(json.dumps({"model": a.model, "onnxruntime": ort.__version__, "provider": a.provider, "active": sess.get_providers(), "rows": rows}, indent=1))
