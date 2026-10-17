"""Small goldens for the Rust tests (tests/fixtures/small/*.npy, < 1 MB total) and the tiny test model (out/tiny.onnx).

    uv run python make_fixtures.py

mel_*.npy       torch LogMel of a fixture wav (frontend parity)
tiny_logp_*.npy offline log-probs of the seeded random-init tiny SmallNet (ONNX-vs-torch; PIPELINE-ONLY weights)"""

from pathlib import Path

import numpy as np
import torch

from common import TINY, load_wav, make_net
from export import export

ROOT = Path(__file__).resolve().parents[2]
AUDIO, OUT = ROOT / "tests/fixtures/audio", ROOT / "tests/fixtures/small"

if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    export("tiny", TINY, out)
    net = make_net(TINY)
    for name in ["en_02s", "mix_04s", "zh_08s"]:
        wav = load_wav(AUDIO / f"{name}.wav")[None]
        with torch.inference_mode():
            mel = net.mel(wav)[0]
            np.save(OUT / f"mel_{name}.npy", np.ascontiguousarray(mel.numpy(), dtype=np.float32))
            if name != "zh_08s":
                lp, lens = net(wav, torch.tensor([wav.shape[1]]))
                np.save(OUT / f"tiny_logp_{name}.npy", np.ascontiguousarray(lp[0, : int(lens[0])].numpy(), dtype=np.float32))
        print(name, "mel", tuple(mel.shape))
