"""End-to-end latency/RTF/memory on MLX: wav -> mel (numpy) -> tower -> reader -> greedy CTC.
Usage: uv run python bench.py <fp32|fp16|bf16> [runs=20] [warmup=3] [gpu|cpu]  -> benchmarks/apple/mlx/bench-<dtype>.json
One process per dtype so peak RSS is not polluted by another precision."""
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
import psutil
from common import *

import tingmix_mlx as T
from embed2asr.ctc import collapse_ctc
from mlx_vlm.models.gemma4.audio_feature_extractor import Gemma4AudioFeatureExtractor

dname = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
warm = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dev = sys.argv[4] if len(sys.argv) > 4 else "gpu"  # "cpu" runs the same graph on the MLX CPU stream (GPU-use control)
mx.set_default_device(mx.cpu if dev == "cpu" else mx.gpu)
DUR = [2, 4, 8, 16, 29]
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # MiB (bytes on macOS)
pct = lambda a, q: float(np.percentile(a, q))

proc = psutil.Process()
t0 = time.perf_counter()
m = T.load({"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[dname])
load_s = time.perf_counter() - t0
rss_load, mem_load = rss(), mx.get_peak_memory() / 2**20
src = next(r for r in fixtures() if r["id"] == "mix_29s")["wav"]
ext = Gemma4AudioFeatureExtractor()


def run(wav):
    t = time.perf_counter()
    o = ext(wav)
    mel, mk = o["input_features"][0], o["input_features_mask"][0]
    t_mel = time.perf_counter()
    n = int(mk.sum())
    n_out = (((n + 1) // 2) + 1) // 2  # two stride-2 convs, pad 1, k 3
    f = m.features(mx.array(mel)[None], mx.array(mk)[None])[:, :n_out]
    mx.eval(f)
    t_tower = time.perf_counter()
    lg = m.reader(f.astype(mx.float32))
    ids = np.array(mx.argmax(lg, -1))[0]
    t_end = time.perf_counter()
    collapse_ctc(ids.tolist(), 0)
    return t_mel - t, t_tower - t_mel, t_end - t_tower, time.perf_counter() - t, n_out


rows = []
for d in DUR:
    wav = src[: d * 16000]
    for _ in range(warm):
        run(wav)
    mx.reset_peak_memory()
    c0, w0 = sum(proc.cpu_times()[:2]), time.perf_counter()
    res = [run(wav) for _ in range(runs)]
    cpu = (sum(proc.cpu_times()[:2]) - c0) / (time.perf_counter() - w0)
    a = np.array(res)
    tot = a[:, 3]
    rows.append(dict(
        audio_s=d, frames=int(a[0, 4]), runs=runs, warmup=warm,
        total_ms=dict(mean=float(tot.mean() * 1e3), p50=pct(tot, 50) * 1e3, p95=pct(tot, 95) * 1e3, p99=pct(tot, 99) * 1e3, min=float(tot.min() * 1e3), max=float(tot.max() * 1e3)),
        phase_ms_mean=dict(mel=float(a[:, 0].mean() * 1e3), tower=float(a[:, 1].mean() * 1e3), reader_greedy=float(a[:, 2].mean() * 1e3)),
        rtf_p50=pct(tot, 50) / d, rtf_p95=pct(tot, 95) / d,
        mlx_peak_mem_mib=mx.get_peak_memory() / 2**20, cpu_cores_busy=cpu,
    ))
    print(d, rows[-1]["total_ms"], f"rtf50={rows[-1]['rtf_p50']:.3f}", flush=True)

art = OUT / "tingmix-large-mlx-fp32.safetensors"
out = dict(
    cmd=f"uv run python bench.py {' '.join(sys.argv[1:])}", dtype=dname,
    device=dict(model=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                os=platform.platform(), mem_gib=psutil.virtual_memory().total / 2**30, mlx_default_device=str(mx.default_device()),
                metal=mx.metal.is_available()),
    versions=dict(mlx=__import__("importlib.metadata", fromlist=["x"]).version("mlx"), python=platform.python_version()),
    revisions=dict(large=LARGE_REV, base=BASE_REV, mlx_vlm="73d366ecf9233b3fc6549531a7f2386726d09ff6"),
    artifact_sha256_fp32=(OUT / "tingmix-large-mlx-fp32.sha256").read_text().split()[0],
    load_s=load_s, peak_rss_after_load_mib=rss_load, mlx_peak_after_load_mib=mem_load, peak_rss_final_mib=rss(),
    loadavg_end=os.getloadavg(),
    note="measured while other agents ran concurrent GPU/CPU work on the same machine (see loadavg); power not measured (powermetrics needs sudo)",
    rows=rows,
)
dst = ROOT / "benchmarks" / "apple" / "mlx" / f"bench-{dname}{'-cpu' if dev == 'cpu' else ''}.json"
dst.write_text(json.dumps(out, indent=1))
print("peak rss", out["peak_rss_final_mib"], "->", dst)
