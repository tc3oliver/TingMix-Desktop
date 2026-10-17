"""Export the streaming step graph of a random-init SmallNet to ONNX and check it against torch.

    uv run python export.py tiny|prod [--out out]

Writes out/<name>.onnx (+ <name>.json with the config and state shapes). Not committed: converted models stay out of git."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import PROD, TINY, StepNet, make_net, zero_state

NAMES = ["mel", "c1", "c2", "h"]


def export(name: str, cfg: dict, out: Path) -> Path:
    net = make_net(cfg)
    step = StepNet(net).eval()
    mel = torch.randn(1, 8, 80)
    args = (mel, *zero_state(cfg))
    path = out / f"{name}.onnx"
    torch.onnx.export(
        step, args, str(path),
        input_names=NAMES, output_names=["logp", "c1_out", "c2_out", "h_out"],
        dynamic_axes={"mel": {1: "K"}, "logp": {1: "T"}},
        opset_version=17, dynamo=False,
    )
    (out / f"{name}.json").write_text(json.dumps({"config": cfg, "inputs": NAMES, "seed": 0, "weights": "random-init, pipeline-only"}, indent=1))
    # parity: whole-utterance and chunked ORT runs against torch
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    mel = torch.randn(1, 96, 80) * 2 - 8
    ref, *_ = step(mel, *zero_state(cfg))
    st = [s.numpy() for s in zero_state(cfg)]
    got = []
    for a in range(0, 96, 24):
        lp, *st = sess.run(None, dict(zip(NAMES, [mel[:, a : a + 24].numpy(), *st])))
        got.append(lp)
    err = np.abs(np.concatenate(got, 1) - ref.detach().numpy()).max()
    print(f"{name}: {path} ({path.stat().st_size / 1e6:.2f} MB), chunked-ORT vs torch max|dlogp| = {err:.2e}")
    assert err < 1e-4, err
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["tiny", "prod"])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    export(a.which, TINY if a.which == "tiny" else PROD, a.out)
