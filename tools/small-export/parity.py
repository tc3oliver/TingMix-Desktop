"""Python-side parity of the streaming step graph (random-init weights, PIPELINE-ONLY):
  offline SmallNet.forward  vs  reference tingmix Stream (160/320/640 ms chunks)  vs  ONNX step graph (same chunks).

    uv run python parity.py [tiny|prod]      -> out/parity-<name>.json"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tingmix.models.small.model import Stream

from common import PROD, TINY, load_wav, make_net
from export import NAMES

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_MS = [160, 320, 640]


def onnx_stream(sess, cfg, net, wav, chunk):
    """Feed audio chunk by chunk like the Rust runtime: mel frames buffered to multiples of 4, state carried."""
    st = [np.zeros((1, 80, 1), np.float32), np.zeros((1, cfg["d"], 1), np.float32), np.zeros((cfg["layers"], 1, cfg["hidden"]), np.float32)]
    samples, buf, outs = torch.zeros(0), torch.zeros(0, 80), []
    for a in range(0, len(wav), chunk):
        samples = torch.cat([samples, wav[a : a + chunk]])
        n = net.mel.frames(len(samples))
        if n:
            buf = torch.cat([buf, net.mel(samples[None, : (n - 1) * 160 + 512])[0]])
            samples = samples[n * 160 :]
        k = len(buf) // 4 * 4
        if k:
            lp, *st = sess.run(None, dict(zip(NAMES, [buf[:k][None].numpy(), *st])))
            outs.append(lp[0])
            buf = buf[k:]
    return np.concatenate(outs)


def main(which):
    cfg = TINY if which == "tiny" else PROD
    net = make_net(cfg)
    sess = ort.InferenceSession(str(Path(__file__).parent / f"out/{which}.onnx"), providers=["CPUExecutionProvider"])
    rows = []
    for name in ["en_02s", "mix_04s", "zh_08s"]:
        wav = load_wav(ROOT / f"tests/fixtures/audio/{name}.wav")
        with torch.inference_mode():
            off, lens = net(wav[None], torch.tensor([len(wav)]))
        off = off[0, : int(lens[0])].numpy()
        for ms in CHUNKS_MS:
            chunk = 16 * ms
            with torch.inference_mode():
                s = Stream(net, lambda ids: "")
                for a in range(0, len(wav), chunk):
                    s.accept(wav[a : a + chunk])
                ref = torch.cat(s.logp).numpy()
            onx = onnx_stream(sess, cfg, net, wav, chunk)
            ids_equal = bool((off.argmax(-1) == onx.argmax(-1)).all())
            row = dict(wav=name, chunk_ms=ms, frames=len(off),
                       torch_stream_vs_offline=float(np.abs(ref - off).max()),
                       onnx_stream_vs_offline=float(np.abs(onx - off).max()), argmax_ids_equal=ids_equal)
            rows.append(row)
            print(row)
    out = Path(__file__).parent / f"out/parity-{which}.json"
    out.write_text(json.dumps({"weights": "random-init, pipeline-only", "cfg": cfg, "rows": rows}, indent=1))
    assert all(r["onnx_stream_vs_offline"] < 1e-3 and r["argmax_ids_equal"] for r in rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny")
