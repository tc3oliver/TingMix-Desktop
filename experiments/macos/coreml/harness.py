"""Convert / run / parity / time / placement helpers shared by every stage. Results are appended verbatim as JSON."""

from __future__ import annotations

import collections
import json
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import OUT, REPO, rel_err

UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}
BENCH = REPO / "benchmarks/apple/coreml"


def hardware() -> dict:
    sp = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()
    return dict(chip=sp("sysctl", "-n", "machdep.cpu.brand_string"), mem_gb=int(sp("sysctl", "-n", "hw.memsize")) // 2**30,
                macos=platform.mac_ver()[0], coremltools=ct.__version__, torch=torch.__version__)


def convert(module, example, name, *, method="trace", precision="fp32", inputs=None, outputs=None, target=None):
    """Returns dict(ok, path, seconds, error). Errors are kept verbatim (full traceback tail)."""
    path = OUT / f"{name}.{method}.{precision}.mlpackage"
    t0 = time.time()
    try:
        module = module.eval()
        with torch.no_grad():
            if method == "trace":
                src = torch.jit.trace(module, example, strict=False)
            elif method == "export":
                src = torch.export.export(module, tuple(example))
                src = src.run_decompositions({})
            else:
                raise ValueError(method)
        kw = dict(convert_to="mlprogram", minimum_deployment_target=target or ct.target.macOS15,
                  compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32)
        if inputs is None:
            inputs = [ct.TensorType(name=f"in{i}", shape=tuple(e.shape)) for i, e in enumerate(example)]
        ml = ct.convert(src, inputs=inputs, outputs=outputs, **kw)
        OUT.mkdir(parents=True, exist_ok=True)
        ml.save(str(path))
        return dict(ok=True, path=str(path), seconds=round(time.time() - t0, 1), size_mb=round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 1))
    except Exception:
        return dict(ok=False, path=None, seconds=round(time.time() - t0, 1), error="".join(traceback.format_exc().splitlines(True)[-25:]))


def run_units(path, feed: dict, ref: dict | None, n=10, warm=3, units=UNITS, extra=None):
    """For each compute unit: load, predict (real execution), parity vs ref, latency (median/min over n runs)."""
    res = {}
    for uname, cu in units.items():
        try:
            t0 = time.time()
            m = ct.models.MLModel(str(path), compute_units=cu)
            load_s = time.time() - t0
            for _ in range(warm):
                out = m.predict(feed)
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                out = m.predict(feed)
                ts.append((time.perf_counter() - t) * 1e3)
            r = dict(ok=True, load_s=round(load_s, 2), median_ms=round(statistics.median(ts), 2), min_ms=round(min(ts), 2), max_ms=round(max(ts), 2), n=n)
            if ref is not None:
                r["parity"] = {k: rel_err(out[k], ref[k]) for k in ref}
            if extra:
                r.update(extra(out))
            res[uname] = r
        except Exception:
            res[uname] = dict(ok=False, error="".join(traceback.format_exc().splitlines(True)[-8:]))
    return res


def placement(path, units=("ALL", "CPU_AND_NE", "CPU_AND_GPU")) -> dict:
    """MLComputePlan: per-op preferred compute device (static plan, not a runtime trace) + supported-device sets."""
    from coremltools.models.compute_plan import MLComputePlan

    m = ct.models.MLModel(str(path))
    compiled = m.get_compiled_model_path()
    out = {}
    for u in units:
        try:
            plan = MLComputePlan.load_from_path(path=compiled, compute_units=UNITS[u])
            prog = plan.model_structure.program
            cnt, ne_supported, ops_total = collections.Counter(), 0, 0
            unsupported = collections.Counter()
            for op in prog.functions["main"].block.operations:
                if op.operator_name == "const":
                    continue
                ops_total += 1
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    cnt["none"] += 1
                    continue
                dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                cnt[dev] += 1
                if any(type(d).__name__ == "MLNeuralEngineComputeDevice" for d in usage.supported_compute_devices):
                    ne_supported += 1
                elif dev != "Neural":
                    unsupported[op.operator_name] += 1
            out[u] = dict(ops=ops_total, preferred=dict(cnt), ne_supported_ops=ne_supported, not_ne_supported_by_type=dict(unsupported.most_common(12)))
        except Exception:
            out[u] = dict(error="".join(traceback.format_exc().splitlines(True)[-6:]))
    return out


def record(stage: str, payload: dict):
    BENCH.mkdir(parents=True, exist_ok=True)
    f = BENCH / f"{stage}.json"
    f.write_text(json.dumps(dict(stage=stage, hardware=hardware(), **payload), indent=1) + "\n")
    print("wrote", f)
