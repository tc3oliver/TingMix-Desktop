"""Counterfactual audio for every clip: what changes if the background or the speaking rate is taken away?

    uv run python -m ea.counterfactual [--variants vocals,slow,shift] NAME [NAME ...]

- `vocals`: Demucs htdemucs source separation, vocals stem only (music and effects removed). Also records the per-clip
  background level as 10*log10(E_vocals / E_rest) in out/NAME/clip_features.json.
- `slow`: ffmpeg atempo=0.8 (pitch kept, 25% more time per unit).
- `shift`: the clip delayed by 10 ms of silence. Inaudible, but it moves every 40 ms frame boundary, so it measures how often
  a span changes by chance under any perturbation (the null baseline for the two tests above).
Writes out/NAME/<variant>/clips/*.wav; then run ea.decode and ea.decoders with --variant."""

import argparse
import subprocess

import numpy as np
import torch

from .common import SR, load_json, read_wav, save_json, video_dir, write_wav


def main():
    import torchaudio.functional as AF
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--variants", default="vocals,slow,shift")
    a = p.parse_args()
    want = set(a.variants.split(","))
    model = get_model("htdemucs").eval()
    vi = model.sources.index("vocals")
    for name in a.names:
        d = video_dir(name)
        meta = load_json(d / "regions.json")
        feats = load_json(d / "clip_features.json") if (d / "clip_features.json").exists() else {}
        for r in meta["regions"]:
            for c in r["clips"]:
                src = d / "clips" / f"{c['id']}.wav"
                if "shift" in want:
                    write_wav(d / "shift" / "clips" / f"{c['id']}.wav", np.concatenate([np.zeros(SR // 100, np.float32), read_wav(src)]))
                if "vocals" in want:
                    x = torch.from_numpy(read_wav(src))
                    y = AF.resample(x, SR, model.samplerate)[None].repeat(2, 1)
                    with torch.inference_mode():
                        stems = apply_model(model, y[None], device="cpu", progress=False)[0]  # [S, 2, N]
                    voc = stems[vi].mean(0)
                    rest = (stems.sum(0) - stems[vi]).mean(0)
                    feats[c["id"]] = dict(vocal_to_rest_db=float(10 * np.log10((voc**2).mean() / max(float((rest**2).mean()), 1e-12))))
                    write_wav(d / "vocals" / "clips" / f"{c['id']}.wav", AF.resample(voc, model.samplerate, SR).numpy())
                if "slow" in want:
                    out = d / "slow" / "clips" / f"{c['id']}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af", "atempo=0.8", "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
            print(name, r["id"], flush=True)
        save_json(d / "clip_features.json", feats)


if __name__ == "__main__":
    main()
