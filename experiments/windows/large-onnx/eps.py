"""ORT execution-provider study for the Large graph (numpy + onnxruntime only, no torch).

    uv run --no-project --with numpy --with onnxruntime-directml python eps.py <model.onnx> <reference-dir> <out.json> [--threads 8]

For every provider the installed onnxruntime offers (strict: no CPU fallback, else reported): parity of all fixture chunks against
the CPU fp32 oracle dumps (rel-L2, max abs, greedy-id agreement) and latency of the 8 s (mix_08s) and 29.5 s (mix_29s chunk 0)
windows: 2 warm-up runs, 10 timed runs, wall time of session.run (mel in, logits out; no frontend, no decode)."""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    ort.preload_dlls()  # onnxruntime-gpu[cuda,cudnn]: CUDA/cuDNN from the pip wheels, no system toolkit
except Exception as e:  # noqa: BLE001
    print("preload_dlls:", str(e)[:120])

model, ref_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 8
chunks = []
for f in sorted(ref_dir.glob("*.npz")):
    d = np.load(f)
    k = 0
    while f"c{k}_mel" in d.files:
        chunks.append((f.stem, k, d[f"c{k}_mel"][:, d[f"c{k}_mel_mask"][0]], d[f"c{k}_logits"]))
        k += 1
timed = {"8s": ("mix_08s", 0), "29.5s": ("mix_29s", 0)}


def session(provider, strict):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if strict and provider != "CPUExecutionProvider":
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(str(model), so, providers=[provider] if (strict or provider == "CPUExecutionProvider") else [provider, "CPUExecutionProvider"])


rows = []
for p in ort.get_available_providers():
    if p in ("AzureExecutionProvider", "TensorrtExecutionProvider"):
        rows.append({"provider": p, "skipped": "not a candidate for this study (Azure: remote; TensorRT: needs libraries not installed)"})
        continue
    row = {"provider": p}
    try:
        t0 = time.perf_counter()
        try:
            s, row["strict_no_cpu_fallback"] = session(p, True), True
        except Exception as e:  # noqa: BLE001
            row["strict_error"] = str(e)[:400]
            s, row["strict_no_cpu_fallback"] = session(p, False), False
        row["session_create_s"] = round(time.perf_counter() - t0, 2)
        row["active_providers"] = s.get_providers()
        par = []
        for name, k, mel, ref in chunks:
            lg = s.run(None, {"mel": mel})[0][0]
            d = np.abs(lg.astype(np.float64) - ref)
            par.append(dict(clip=name, chunk=k, frames=int(mel.shape[1]), max_abs=float(d.max()),
                            rel_l2=float(np.linalg.norm(lg - ref) / np.linalg.norm(ref)),
                            id_agreement=float((lg.argmax(-1) == ref.argmax(-1)).mean())))
        row["parity"] = par
        row["worst_rel_l2"] = max(x["rel_l2"] for x in par)
        row["min_id_agreement"] = min(x["id_agreement"] for x in par)
        lat = {}
        for label, (name, k) in timed.items():
            mel = next(m for n, kk, m, _ in chunks if n == name and kk == k)
            for _ in range(2):
                s.run(None, {"mel": mel})
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                s.run(None, {"mel": mel})
                ts.append(time.perf_counter() - t0)
            audio_s = mel.shape[1] / 100
            lat[label] = dict(frames=int(mel.shape[1]), audio_s=audio_s, p50_ms=float(np.percentile(ts, 50)) * 1e3, p95_ms=float(np.percentile(ts, 95)) * 1e3,
                              min_ms=min(ts) * 1e3, rtf_p50=float(np.percentile(ts, 50)) / audio_s)
        row["latency"] = lat
        print(p, row["strict_no_cpu_fallback"], "rel_l2", row["worst_rel_l2"], {k: round(v["p50_ms"], 1) for k, v in lat.items()})
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:600]
        print(p, "ERROR", row["error"][:200])
    rows.append(row)
out.write_text(json.dumps({"what": "TingMix-Large mel->logits ONNX, per execution provider", "model": str(model), "onnxruntime": ort.__version__,
                           "threads": threads, "platform": platform.platform(), "python": platform.python_version(), "rows": rows}, indent=1))
