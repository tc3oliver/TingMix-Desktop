"""Is the app's live-path loss the backend or the segmentation? Re-decode every live segment with the PyTorch reference.

    uv run python -m ea.liveparity NAME [NAME ...]

Cuts each region at the app's own final boundaries (out/NAME/app_live.json, MLX), decodes the pieces with the PyTorch
CPU fp32 reference and the same LM decoder, and compares the normalized texts. Writes out/liveparity.json."""

import sys

from embed2asr.lm import beam_decode

from .common import ALPHA, BEAM, BETA, OUT, SR, load_json, norm, read_wav, save_json, video_dir
from .decode import load, run
from .decoders import _G, _init


def main():
    asr = load("cpu")
    _init()
    tok, lm, units = _G["tok"], _G["lm"], _G["units"]
    rows = []
    for name in sys.argv[1:]:
        meta, live = load_json(video_dir(name) / "regions.json"), load_json(video_dir(name) / "app_live.json")
        for r in meta["regions"]:
            x = read_wav(video_dir(name) / "regions" / f"{r['id']}.wav")
            for a, b, mlx in live[r["id"]]:
                lp, _ = run(asr, x[int(a * SR) : int(b * SR)])
                pt = beam_decode(lp, tok, lm, ALPHA, BETA, BEAM, units=units)
                rows.append(dict(video=name, region=r["id"], start=a, end=b, identical=norm(pt) == norm(mlx)))
    same = sum(r["identical"] for r in rows)
    save_json(OUT / "liveparity.json", dict(identical=same, segments=len(rows), rows=rows))
    print(f"live segments with identical text (MLX app vs PyTorch reference): {same} of {len(rows)}")


if __name__ == "__main__":
    main()
