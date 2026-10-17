"""INT8 study on the step graph (random-init weights: size, latency and drift only, NOT accuracy).

    uv run python quantize.py [tiny|prod]      -> out/<name>.*.onnx, out/quant-<name>.json

Variants
  fp32            exported step graph (ONNX GRU operator)
  dyn_gruop       quantize_dynamic on the above: ORT cannot quantize the GRU operator, so only the output Linear changes
  static_conv     static QDQ on the two Conv layers (calibrated on fixture audio), GRU stays FP32
  fp32_unrolled   GRU as MatMuls, fixed 8 steps (K=32 frames = 320 ms)
  dyn_unrolled    quantize_dynamic (MatMul) on the unrolled graph: GRU input/recurrent projections + output Linear
  static_dyn      static conv QDQ + dynamic MatMul on the unrolled graph
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

from common import PROD, TINY, StepNetUnrolled, load_wav, make_net, zero_state
from export import NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "out"
K = 32  # mel frames per step = 320 ms of audio
FILES = ["en_02s", "mix_04s", "zh_08s"]


def mel_chunks(net, name):
    wav = load_wav(ROOT / f"tests/fixtures/audio/{name}.wav")
    with torch.inference_mode():
        mel = net.mel(wav[None])[0]
    return [mel[a : a + K][None].numpy() for a in range(0, len(mel) - K + 1, K)]


def run_chain(sess, cfg, chunks):
    """Run chunks with state carried; returns logp [T,V] and per-call seconds."""
    st = [s.numpy() for s in zero_state(cfg)]
    outs, times = [], []
    for m in chunks:
        t0 = time.perf_counter()
        lp, *st = sess.run(None, dict(zip(NAMES, [m, *st])))
        times.append(time.perf_counter() - t0)
        outs.append(lp[0])
    return np.concatenate(outs), times


class Calib(CalibrationDataReader):
    """Feeds real mel chunks with FP32 state (state is re-derived by running the fp32 model alongside)."""

    def __init__(self, fp32_path, cfg, net):
        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        self.items = []
        for name in FILES:
            st = [s.numpy() for s in zero_state(cfg)]
            for m in mel_chunks(net, name):
                self.items.append(dict(zip(NAMES, [m, *st])))
                _, *st = sess.run(None, self.items[-1])
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)


def main(which):
    cfg = TINY if which == "tiny" else PROD
    net = make_net(cfg)
    fp32 = OUT / f"{which}.onnx"
    # unrolled fp32 export
    unrolled = OUT / f"{which}.unrolled.onnx"
    mel = torch.randn(1, K, 80)
    torch.onnx.export(StepNetUnrolled(net, K // 4).eval(), (mel, *zero_state(cfg)), str(unrolled), input_names=NAMES,
                      output_names=["logp", "c1_out", "c2_out", "h_out"], opset_version=17, dynamo=False)
    paths = {"fp32": fp32, "fp32_unrolled": unrolled}
    paths["dyn_gruop"] = OUT / f"{which}.dyn_gruop.onnx"
    quantize_dynamic(fp32, paths["dyn_gruop"], weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    paths["static_conv"] = OUT / f"{which}.static_conv.onnx"
    quantize_static(fp32, paths["static_conv"], Calib(fp32, cfg, net), quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"],
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    paths["dyn_unrolled"] = OUT / f"{which}.dyn_unrolled.onnx"
    quantize_dynamic(unrolled, paths["dyn_unrolled"], weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    paths["static_dyn"] = OUT / f"{which}.static_dyn.onnx"
    tmp = OUT / f"{which}.static_unrolled.tmp.onnx"
    quantize_static(unrolled, tmp, Calib(unrolled, cfg, net), quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"],
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    quantize_dynamic(tmp, paths["static_dyn"], weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    tmp.unlink()

    chunks = {n: mel_chunks(net, n) for n in FILES}
    ref = {}
    rows = []
    for variant, path in paths.items():
        row = {"variant": variant, "size_mb": round(path.stat().st_size / 1e6, 3)}
        for threads in (1, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            drift, agree, n, times = [], 0, 0, []
            for name in FILES:
                lp, ts = run_chain(sess, cfg, chunks[name])
                times += ts[1:]  # skip warm-up call
                if variant == "fp32" and threads == 1:
                    ref[name] = lp
                if name in ref:
                    drift.append(np.abs(lp - ref[name]))
                    agree += int((lp.argmax(-1) == ref[name].argmax(-1)).sum())
                    n += len(lp)
            row[f"latency_ms_p50_{threads}t"] = round(float(np.percentile(times, 50)) * 1e3, 3)
            row[f"latency_ms_p95_{threads}t"] = round(float(np.percentile(times, 95)) * 1e3, 3)
            if drift and threads == 1:
                d = np.concatenate([x.ravel() for x in drift])
                row.update(max_abs_dlogp=float(d.max()), mean_abs_dlogp=float(d.mean()), argmax_agreement=round(agree / n, 4))
        ops = sorted({nd.op_type for nd in __import__("onnx").load(str(path)).graph.node})
        row["ops"] = ops
        rows.append(row)
        print({k: v for k, v in row.items() if k != "ops"})
    (OUT / f"quant-{which}.json").write_text(json.dumps({
        "weights": "random-init, pipeline-only (drift is vs the FP32 graph, not accuracy)",
        "chunk_ms": 320, "onnxruntime": ort.__version__, "cfg": cfg, "rows": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny")
