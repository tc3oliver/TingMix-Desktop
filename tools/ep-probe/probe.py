"""ORT execution-provider probe for the Small step graph: for every provider this onnxruntime build offers, does a session
initialise, does it run the whole graph on that provider (or fall back to CPU per node), do the logits match the CPU EP,
and how long does a 320 ms step take. PIPELINE-ONLY: random-init weights, real log-mel from a fixture.

    uv run --no-project --with numpy --with onnxruntime-directml python probe.py <model.onnx> <mel.npy> <out.json> [--note TEXT]

Any acceleration claim needs the numbers this prints; availability alone proves nothing.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:  # onnxruntime-gpu[cuda,cudnn]: load the CUDA/cuDNN DLLs from the pip wheels instead of a system-wide CUDA toolkit
    ort.preload_dlls()
except Exception as e:  # noqa: BLE001 - older/CPU-only builds have no preload_dlls
    print("preload_dlls:", str(e)[:200])

K = 32  # mel frames per step = 320 ms
NAMES = ["mel", "c1", "c2", "h"]


def state(sess):
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    d, (layers, _, hidden) = shapes["c2"][1], (shapes["h"][0], shapes["h"][1], shapes["h"][2])
    return [np.zeros((1, 80, 1), np.float32), np.zeros((1, d, 1), np.float32), np.zeros((layers, 1, hidden), np.float32)]


def run(sess, mel):
    st, outs, times = state(sess), [], []
    for a in range(0, len(mel) // K * K, K):
        t0 = time.perf_counter()
        lp, *st = sess.run(None, dict(zip(NAMES, [mel[a : a + K][None], *st])))
        times.append(time.perf_counter() - t0)
        outs.append(lp[0])
    return np.concatenate(outs), times


def session(path, provider, strict):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if strict and provider != "CPUExecutionProvider":
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    # strict: only the provider under test (the CPU EP may not be listed when fallback is disabled)
    provs = [provider] if strict or provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=provs)


def main():
    model, mel_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
    mel = np.load(mel_path).astype(np.float32)
    ref, ref_t = run(session(model, "CPUExecutionProvider", False), mel)
    rows = []
    for p in ort.get_available_providers():
        row = {"provider": p}
        try:
            for strict in (True, False):
                try:
                    s = session(model, p, strict)
                    row["init_ok"] = True
                    row["runs_entirely_on_provider" if strict else "runs_with_cpu_fallback"] = True
                    break
                except Exception as e:  # noqa: BLE001 - report whatever the provider says
                    row["runs_entirely_on_provider" if strict else "runs_with_cpu_fallback"] = False
                    row["error_strict" if strict else "error"] = str(e)[:300]
            else:
                row["init_ok"] = False
                rows.append(row)
                continue
            for _ in range(3):  # warm-up (kernel compilation, first-run allocations)
                run(s, mel[: K * 4])
            lp, times = run(s, mel)
            d = np.abs(lp - ref)
            row.update(
                active_providers=s.get_providers(),
                max_abs_dlogp_vs_cpu=float(d.max()), mean_abs_dlogp_vs_cpu=float(d.mean()),
                argmax_agreement_vs_cpu=float((lp.argmax(-1) == ref.argmax(-1)).mean()),
                step_ms_p50=float(np.percentile(times, 50)) * 1e3, step_ms_p95=float(np.percentile(times, 95)) * 1e3,
                cpu_step_ms_p50=float(np.percentile(ref_t, 50)) * 1e3,
            )
        except Exception as e:  # noqa: BLE001
            row.setdefault("init_ok", False)
            row["error"] = str(e)[:300]
        rows.append(row)
        print({k: v for k, v in row.items() if k != "error"})
    out.write_text(json.dumps({
        "what": "ORT EP probe on the Small step graph, 320 ms steps over a real log-mel fixture; random-init weights (pipeline-only)",
        "model": str(model), "mel": str(mel_path), "frames": int(len(mel)), "onnxruntime": ort.__version__, "device": ort.get_device(),
        "available_providers": ort.get_available_providers(), "python": platform.python_version(), "platform": platform.platform(),
        "note": note, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
