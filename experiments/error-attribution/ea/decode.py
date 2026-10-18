"""Run the TingMix-Large encoder + Reader once per audio unit and keep what every decoder needs.

    uv run python -m ea.decode NAME [--variant orig|vocals|slow] [--device cpu]

Units: every clip (out/NAME/<variant>/clips/*.wav, or the original clips for `orig`), and for `orig` also every region
under two fixed chunkings: `fixed25` (consecutive 25 s windows, as a file transcription with a 25 s cap) and `overlap`
(25 s windows every 20 s; each frame's logits are taken from the window where that frame is furthest from an edge).
Writes out/NAME/logits/<variant>/<unit>.npz with `lp` (log-softmax, f32 [T,V]) and, for clips, `h` (Reader states
[T,768], the attention decoder's memory). Encoder and Reader weights are identical in v2 and v3 (same file hashes on the
Hub), so one pass serves both."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import LARGE_ID, SR, V3_REV, load_json, read_wav, video_dir

FPS = 25


def load(device):
    from tingmix.models.large.model import LargeModel

    m = LargeModel.from_pretrained(LARGE_ID, revision=V3_REV, device=device, att_weight=0.0)
    return m.asr


@torch.inference_mode()
def run(asr, wav):
    from tingmix.models.large.attention import reader_states

    feats, lens = asr.encoder([wav])
    logits, h = reader_states(asr.reader, feats.to(asr.device).float(), lens)
    n = int(lens[0])
    return logits[0, :n].float().log_softmax(-1).cpu().numpy(), h[0, :n].float().cpu().numpy()


def windows(n_samples, win_s, hop_s):
    w, hop = int(win_s * SR), int(hop_s * SR)
    starts = list(range(0, max(n_samples - w, 0) + 1, hop))
    if starts[-1] + w < n_samples:
        starts.append(starts[-1] + hop if hop < w else starts[-1] + w)
    return [(s, min(s + w, n_samples)) for s in starts]


def stitched(asr, x, win_s, hop_s):
    """Logits of a long recording from windowed passes; overlapping frames come from the window where they are most central."""
    T = int(len(x) / SR * FPS) + 1
    best = np.full(T, -1.0)
    out = None
    for a, b in windows(len(x), win_s, hop_s):
        lp, _ = run(asr, x[a:b])
        f0 = int(round(a / SR * FPS))
        for i in range(lp.shape[0]):
            t = f0 + i
            if t >= T:
                break
            margin = min(i, lp.shape[0] - 1 - i)
            if margin > best[t]:
                if out is None:
                    out = np.zeros((T, lp.shape[1]), dtype=np.float32)
                out[t], best[t] = lp[i], margin
    return out[best >= 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--variant", default="orig")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    d = video_dir(a.name)
    meta = load_json(d / "regions.json")
    asr = load(a.device)
    od = d / "logits" / a.variant
    od.mkdir(parents=True, exist_ok=True)
    src = d / "clips" if a.variant == "orig" else d / a.variant / "clips"
    for r in meta["regions"]:
        for c in r["clips"]:
            lp, h = run(asr, read_wav(src / f"{c['id']}.wav"))
            np.savez(od / f"{c['id']}.npz", lp=lp, h=h)
        if a.variant == "orig":
            x = read_wav(d / "regions" / f"{r['id']}.wav")
            np.savez(od / f"{r['id']}_fixed25.npz", lp=stitched(asr, x, 25.0, 25.0))
            np.savez(od / f"{r['id']}_overlap.npz", lp=stitched(asr, x, 25.0, 20.0))
        print(r["id"], "done", flush=True)


if __name__ == "__main__":
    main()
