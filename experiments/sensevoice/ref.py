"""sherpa-onnx oracle for the Rust SenseVoice backend (tingmix-runtime-onnx `sensevoice`).

Writes out/reference.json: per clip the text sherpa-onnx decodes (greedy CTC, language auto, ITN on), plus the first LFR+CMVN
feature frames of each clip computed with kaldi-native-fbank under sherpa's SenseVoice settings, so the Rust frontend can be
compared number by number. Usage: uv run python ref.py <model dir> <wav>...
"""

import json
import sys
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import sherpa_onnx
import soundfile as sf


def fbank(wav: np.ndarray) -> np.ndarray:
    # offline-recognizer-sense-voice-impl.h InitFeatConfig + features.h defaults
    o = knf.FbankOptions()
    o.frame_opts.samp_freq = 16000
    o.frame_opts.dither = 0
    o.frame_opts.snip_edges = True
    o.frame_opts.window_type = "hamming"
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20
    o.mel_opts.high_freq = 0
    f = knf.OnlineFbank(o)
    f.accept_waveform(16000, (wav * 32768).tolist())  # normalize_samples = 0 in the model metadata
    f.input_finished()
    return np.array([f.get_frame(i) for i in range(f.num_frames_ready)], dtype=np.float32)


def lfr_cmvn(x: np.ndarray, meta: dict) -> np.ndarray:
    # sherpa-onnx lfr.cc (FunASR apply_lfr): first frame repeated (m-1)//2 times on the left, last frame past the end
    m, n = int(meta["lfr_window_size"]), int(meta["lfr_window_shift"])
    out_n = (len(x) + n - 1) // n
    padded = np.concatenate([np.repeat(x[:1], (m - 1) // 2, 0), x, np.repeat(x[-1:], m, 0)])
    out = [padded[i * n : i * n + m].reshape(-1) for i in range(out_n)]
    neg_mean = np.array([float(v) for v in meta["neg_mean"].split(",")], dtype=np.float32)
    inv = np.array([float(v) for v in meta["inv_stddev"].split(",")], dtype=np.float32)
    return (np.stack(out) + neg_mean) * inv


def main() -> None:
    model_dir, wavs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=4,
        use_itn=True,
        language="",
    )
    import onnx  # metadata only

    meta = {p.key: p.value for p in onnx.load(str(model_dir / "model.int8.onnx"), load_external_data=False).metadata_props}
    Path("out").mkdir(exist_ok=True)
    out = {"sherpa_onnx": sherpa_onnx.__version__, "kaldi_native_fbank": knf.__version__ if hasattr(knf, "__version__") else "", "clips": {}}
    for p in wavs:
        wav, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == 16000, p
        wav = wav[:, 0]
        s = rec.create_stream()
        s.accept_waveform(16000, wav)
        rec.decode_stream(s)
        feats = lfr_cmvn(fbank(wav), meta)
        np.save(f"out/{p.stem}_lfr.npy", feats)
        out["clips"][p.stem] = {
            "text": s.result.text,
            "lang": s.result.lang,
            "lfr_frames": int(len(feats)),
            "lfr_head": feats[:3].round(5).tolist(),
        }
        print(p.stem, s.result.lang, s.result.text)
    Path("out").mkdir(exist_ok=True)
    Path("out/reference.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
