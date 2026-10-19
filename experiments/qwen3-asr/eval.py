"""Qwen3-ASR (mlx-qwen3-asr, fp16, language auto) on the error-attribution clips -> out/qwen_<size>_<video>.json in the
format `ea.other` reads ({"texts": {clip: text}, timing}). Weights are downloaded at pinned revisions into $HF_HOME.

    uv run python eval.py <error-attribution out dir> 0.6B 1.7B
"""

import json
import sys
import time
from pathlib import Path

REVISIONS = {
    "0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
}


def main() -> None:
    from huggingface_hub import snapshot_download
    import mlx_qwen3_asr as q
    import soundfile as sf

    ea = Path(sys.argv[1])
    for size in sys.argv[2:]:
        local = snapshot_download(f"Qwen/Qwen3-ASR-{size}", revision=REVISIONS[size])
        t = time.perf_counter()
        s = q.Session(local)
        load_s = time.perf_counter() - t
        for video in ("product", "finance"):
            texts, audio_s, decode_s = {}, 0.0, 0.0
            for wav in sorted((ea / video / "clips").glob("*.wav")):
                audio_s += sf.info(wav).duration
                t = time.perf_counter()
                r = s.transcribe(str(wav))
                decode_s += time.perf_counter() - t
                texts[wav.stem] = r.text
            doc = {"backend": f"Qwen3-ASR-{size} mlx-qwen3-asr fp16", "revision": REVISIONS[size], "load_s": load_s,
                   "clips": len(texts), "audio_s": audio_s, "decode_s": decode_s, "rtf": decode_s / audio_s, "texts": texts}
            Path("out").mkdir(exist_ok=True)
            Path(f"out/qwen_{size}_{video}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            print(size, video, len(texts), f"RTF {decode_s / audio_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
