"""Export TingMix-Large (audio tower + Reader, LoRA merged) mel -> CTC logits to ONNX with a dynamic frame count.

    uv run python export.py [--out out/tingmix-large.onnx] [--no-prescale]

Input  mel    [1, F, 128] float32  log-mel of the VALID frames only (the Rust Gemma4 frontend's `valid`), F >= 8
Output logits [1, T, 5025] float32 T = ceil(ceil(F/2)/2); CTC blank = 0
Weights: google/embeddinggemma-2@914f7f8 audio tower + TingMix-Large@75071a1 (see common.py); output is not committed."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common import OUT, load_reader, load_tower
from modules import EncoderReaderDyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "tingmix-large.onnx")
    ap.add_argument("--no-prescale", action="store_true", help="plain RMSNorm (fp32 only)")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tower, (reader, itos) = load_tower(), load_reader()
    model = EncoderReaderDyn(tower, reader, rms_prescale=not a.no_prescale).eval()
    mel = torch.randn(1, 751, 128)
    with torch.inference_mode():
        torch.onnx.export(
            model, (mel,), str(a.out), input_names=["mel"], output_names=["logits"],
            dynamic_axes={"mel": {1: "F"}, "logits": {1: "T"}}, opset_version=17, dynamo=False,
        )
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    meta = {"file": a.out.name, "bytes": a.out.stat().st_size, "sha256": h, "opset": 17, "torch": torch.__version__,
            "vocab": len(itos), "weights": "google/embeddinggemma-2@914f7f8 + TingMix-Large@75071a1 (LoRA merged, fp32)"}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
