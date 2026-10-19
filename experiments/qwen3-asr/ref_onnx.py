"""Oracle for the Rust Qwen3-ASR backend (tingmix-runtime-onnx `qwen3`): the export repository's own Python ONNX path
(`src/mel.py`, `src/prompt.py`, `src/inference.py` from andrewleech/qwen3-asr-onnx at 4aee2ae, cloned into out/export-src),
on the same int4 graphs. Writes out/qwen_ref.json (per clip: generated ids, text after <asr_text>, mel frames) and
out/<clip>_mel.npy.

    git clone https://github.com/andrewleech/qwen3-asr-onnx.git out/export-src && git -C out/export-src checkout 4aee2ae
    uv run python ref_onnx.py <export dir> <wav>...
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("out/export-src").resolve()))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402
from src.inference import greedy_decode_onnx  # noqa: E402
from src.mel import log_mel_spectrogram  # noqa: E402
from src.prompt import ASR_TEXT_TOKEN_ID, build_prompt_ids  # noqa: E402


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    d, wavs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    q = lambda stem: str(d / f"{stem}.int4.onnx") if (d / f"{stem}.int4.onnx").exists() else str(d / f"{stem}.onnx")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    enc = ort.InferenceSession(q("encoder"), opt, providers=["CPUExecutionProvider"])
    sessions = {s: ort.InferenceSession(q(s), opt, providers=["CPUExecutionProvider"]) for s in ("decoder_init", "decoder_step")}
    embed = np.fromfile(d / "embed_tokens.bin", dtype=np.float16).reshape(151936, -1).astype(np.float32)
    vocab = {i: t for t, i in json.loads((d / "vocab.json").read_text()).items()}
    bd = byte_decoder()
    out = {"onnxruntime": ort.__version__, "clips": {}}
    for p in wavs:
        wav, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == 16000
        mel = log_mel_spectrogram(wav[:, 0]).numpy()
        np.save(f"out/{p.stem}_mel.npy", mel[0])
        feats = enc.run(None, {enc.get_inputs()[0].name: mel})[0]
        ids = greedy_decode_onnx(sessions, embed, feats, build_prompt_ids(feats.shape[1]), max_tokens=32 + len(wav) * 15 // 16000)
        gen = [i for i in ids if i not in (151643, 151645)]
        text_ids = gen[gen.index(ASR_TEXT_TOKEN_ID) + 1 :] if ASR_TEXT_TOKEN_ID in gen else []
        text = bytes(b for i in text_ids for ch in vocab.get(i, "") for b in [bd[ch]]).decode("utf-8", "replace").strip()
        out["clips"][p.stem] = {"ids": gen, "text": text, "mel_frames": int(mel.shape[2])}
        print(p.stem, mel.shape, gen[:6], text, flush=True)
    Path("out/qwen_ref.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
