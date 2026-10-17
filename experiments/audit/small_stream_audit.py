"""Targeted streaming-correctness checks for TingMix-Small (tingmix @ f44cbdb, src/tingmix/models/small/model.py).
Random weights only: this proves structural/streaming properties, never accuracy. Writes out/small_stream_audit.json.

  uv run python small_stream_audit.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from embed2asr.ctc import collapse_ctc
from embed2asr.tokenizer import Tok
from tingmix.models.small import SmallModel
from tingmix.models.small.model import HOP, N_FFT, LogMel, SmallNet, Stream

torch.set_num_threads(4)
R = {"torch": torch.__version__, "checks": {}}


def check(name, ok, **detail):
    R["checks"][name] = {"pass": bool(ok), **detail}
    print(("PASS " if ok else "FAIL ") + name, detail)


def net(vocab=5025, d=512, hidden=640, layers=5, seed=0):
    torch.manual_seed(seed)
    return SmallNet(vocab_size=vocab, d=d, hidden=hidden, layers=layers, dropout=0.0).eval()


def offline(n, wav):
    with torch.inference_mode():
        lp, lens = n(torch.from_numpy(wav)[None], torch.tensor([len(wav)]))
    return lp[0, : int(lens[0])]


def stream(n, wav, chunks):
    s, i = Stream(n, lambda ids: ""), 0
    for c in chunks:
        s.accept(wav[i : i + c])
        i += c
    return s


def split(total, sizes):
    out, i = [], 0
    while i < total:
        c = min(sizes[len(out) % len(sizes)], total - i)
        out.append(c)
        i += c
    return out


rng = np.random.default_rng(1)
wav = (rng.standard_normal(16000 * 7) * 0.1).astype(np.float32)
prod = net()  # production-size defaults of training/small/train.py (d 512, hidden 640, 5 GRU layers); vocab = Large's 5025 as a stand-in

# 1. frame count / output shape: closed form vs implementation, streaming vs offline, lengths around every boundary
bad, rows = [], []
small = net(vocab=40, d=16, hidden=24, layers=2)
for n_s in list(range(0, 2600, 1)) + [4000, 16000, 16001, 16159, 16160, 16161, 33333]:
    x = (rng.standard_normal(n_s) * 0.1).astype(np.float32)
    frames = 0 if n_s < N_FFT else (n_s - N_FFT) // HOP + 1
    expect = frames // 4
    s = stream(small, x, split(n_s, [333])) if n_s else Stream(small, lambda i: "")
    got_stream = sum(len(l) for l in s.logp)
    try:
        got_off = int(offline(small, x).shape[0]) if n_s >= 1 else None
    except Exception as e:  # noqa: BLE001
        got_off = f"{type(e).__name__}"
    rows.append((n_s, expect, got_stream, got_off))
    if got_stream != expect or (got_off is not None and got_off != expect):
        bad.append((n_s, expect, got_stream, got_off))
crash = [b for b in bad if isinstance(b[3], str)]
mismatch = [b for b in bad if not isinstance(b[3], str)]
check("frame_count_stream_equals_closed_form", all(r[2] == r[1] for r in rows), checked=len(rows))
check("frame_count_offline_equals_closed_form_when_it_runs", not mismatch, mismatches=mismatch[:5])
R["offline_forward_crash_samples"] = {"count": len(crash), "min": min([c[0] for c in crash], default=None),
                                      "max": max([c[0] for c in crash], default=None), "error": crash[0][3] if crash else None}
check("offline_forward_runs_for_every_length_ge_1", not crash, crashing_lengths=f"{len(crash)} lengths, max {R['offline_forward_crash_samples']['max']}",
      error=R["offline_forward_crash_samples"]["error"])

# 2. streaming == offline at production size, many chunkings (1 sample, STFT/hop/conv boundaries, random)
worst = {}
chunkings = {"1": [1], "7": [7], "159": [159], "160": [160], "161": [161], "511": [511], "512": [512], "513": [513],
             "639": [639], "640": [640], "641": [641], "1600": [1600], "16000": [16000], "random": [int(x) for x in rng.integers(1, 5000, 400)],
             "whole": [len(wav)]}
off = offline(prod, wav)
for name, sz in chunkings.items():
    if name == "1":
        w = wav[: 16000 * 2]  # one sample at a time is slow
        o = offline(prod, w)
        s = stream(prod, w, split(len(w), sz))
    else:
        o = off
        s = stream(prod, wav, split(len(wav), sz))
    st = torch.cat(s.logp)
    worst[name] = {"frames": [int(st.shape[0]), int(o.shape[0])], "max_abs_diff": float((st - o).abs().max()) if st.shape == o.shape else None,
                   "argmax_equal": bool((st.argmax(-1) == o.argmax(-1)).all()) if st.shape == o.shape else False}
R["stream_vs_offline_prod_size"] = worst
check("stream_equals_offline_prod_size_all_chunkings", all(v["max_abs_diff"] is not None and v["max_abs_diff"] < 1e-4 and v["argmax_equal"] for v in worst.values()),
      worst_diff=max(v["max_abs_diff"] for v in worst.values() if v["max_abs_diff"] is not None))

# 3. sensitivity: the check above must fail if state is NOT carried (proves the test can detect a GRU/conv-context bug)
class NoGruCarry(Stream):
    def accept(self, pcm):
        self.state = None
        return super().accept(pcm)
s = NoGruCarry(prod, lambda i: "")
for i in range(0, len(wav), 3200):
    s.accept(wav[i : i + 3200])
d_nocarry = float((torch.cat(s.logp) - off).abs().max())
check("sensitivity_gru_reset_is_detected", d_nocarry > 1e-2, max_abs_diff_without_carry=d_nocarry)

# 4. causality / padding invariance: prefix of the audio and padded batch give the same frames
full = offline(prod, wav)
pre = offline(prod, wav[: 16000 * 3 + 123])
check("offline_prefix_stable", torch.allclose(full[: len(pre)], pre, atol=1e-4), prefix_frames=len(pre),
      max_abs_diff=float((full[: len(pre)] - pre).abs().max()))
a, b = wav[:16000 * 2 + 77], wav[: 16000 * 5 + 5]
x = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a), torch.from_numpy(b)], batch_first=True)
with torch.inference_mode():
    lp, fl = prod(x, torch.tensor([len(a), len(b)]))
da = float((lp[0, : fl[0]] - offline(prod, a)).abs().max())
db = float((lp[1, : fl[1]] - offline(prod, b)).abs().max())
check("batch_padding_invariant", max(da, db) < 1e-4, diffs=[da, db])

# 5. tail: audio samples never decoded after finalize (no flush / right padding)
unc = []
for n_s in range(992, 20000):  # below 992 samples no output frame exists at all (see check 1)
    k = ((n_s - N_FFT) // HOP + 1) // 4
    covered = 640 * (k - 1) + 3 * HOP + N_FFT  # last sample used by output frame k-1 (exclusive)
    unc.append(n_s - covered)
R["tail_uncovered_samples"] = {"max": max(unc), "max_ms": max(unc) / 16, "min": min(unc)}
check("finalize_drops_tail_up_to_40ms", max(unc) <= 640, max_uncovered_samples=max(unc), ms=max(unc) / 16)
first_frame_audio = 3 * HOP + N_FFT
R["first_output_after_samples"] = first_frame_audio + 1 * 0  # output 0 needs mel frames 0..3 => samples 0..991
check("min_audio_for_first_frame", Stream(prod, lambda i: "").accept(wav[:991]) == "" and len(stream(prod, wav[:992], [992]).logp) == 1,
      samples=992)

# 6. CTC across chunk boundaries: collapse over the full id history == offline collapse, incl. boundaries between equal ids
ids_off = off.argmax(-1).tolist()
ok_all, n_equal_boundary = True, 0
for sz in ([640], [1280], [641], [2000, 777]):
    s = stream(prod, wav, split(len(wav), sz))
    dec_stream = collapse_ctc(s.ids)
    ok_all &= dec_stream == collapse_ctc(ids_off) and s.ids == ids_off
    pos = np.cumsum([len(l) for l in s.logp])[:-1]
    n_equal_boundary += int(sum(s.ids[p - 1] == s.ids[p] for p in pos))
check("ctc_collapse_across_chunks", ok_all, boundaries_with_equal_ids=n_equal_boundary,
      note="random weights give mostly non-blank repeats; real blank/repeat behaviour is unmeasured")

# 7. resource growth of Stream: per-frame log-probs and ids are retained forever
V = prod.cfg["vocab_size"]
s = stream(prod, wav, split(len(wav), [1600]))
R["stream_growth"] = {"frames_kept": sum(len(l) for l in s.logp), "bytes_per_frame": V * 4,
                      "mb_per_hour_at_25fps": 25 * 3600 * V * 4 / 1e6}
check("stream_logp_retained_unbounded", True, mb_per_hour=round(R["stream_growth"]["mb_per_hour_at_25fps"]),
      note="Stream.logp keeps every frame (diagnostics); a port must not copy this")
# decode cost per accept: Stream._decode re-collapses the whole id list (O(n)) every call
dec = lambda ids: collapse_ctc(ids)  # noqa: E731
times = []
for secs in (10, 60, 300):
    s = Stream(prod, dec)
    s.ids = [int(i) for i in rng.integers(0, 5025, 25 * secs)]
    t0 = time.perf_counter()
    for _ in range(100):
        s.decode(s.ids)
    times.append((secs, (time.perf_counter() - t0) / 100 * 1e6))
R["decode_us_per_accept"] = times

# 8. SmallModel wrapper on short / mixed inputs
tok = Tok(["<blank>"] + [chr(0x4e00 + i) for i in range(39)])
m = SmallModel(small, tok)
res = {}
for n_s in (0, 100, 511, 512, 700, 991, 992, 1119, 1120, 4000):
    try:
        out = m.transcribe_batch([np.zeros(n_s, np.float32)])
        res[n_s] = "ok"
    except Exception as e:  # noqa: BLE001
        res[n_s] = f"{type(e).__name__}: {str(e)[:80]}"
R["transcribe_batch_short_single"] = res
try:
    m.transcribe_batch([np.zeros(100, np.float32), (rng.standard_normal(16000) * 0.1).astype(np.float32)])
    R["transcribe_batch_short_mixed"] = "ok"
except Exception as e:  # noqa: BLE001
    R["transcribe_batch_short_mixed"] = f"{type(e).__name__}: {str(e)[:80]}"
check("transcribe_batch_survives_audio_shorter_than_992_samples", all(v == "ok" for v in res.values()), results=res)

# 9. numerics of the front-end a port must match
mel = LogMel()
R["frontend"] = {"window": "torch.hann_window(400) periodic", "n_fft": N_FFT, "hop": HOP, "center": False,
                 "power": "abs^2", "floor": 1e-6, "mel_scale": "HTK (docstring says Slaney-style)",
                 "fb_shape": list(mel.fb.shape), "fb_row_sum_min_max": [float(mel.fb.sum(1).min()), float(mel.fb.sum(1).max())]}

# 10. front-end definition a port must copy: torch.stft centres the 400-sample window inside the 512 FFT frame
from tingmix.models.small.model import WIN  # noqa: E402
w = torch.from_numpy((rng.standard_normal(2000) * 0.1).astype(np.float32))
spec = torch.stft(w, N_FFT, HOP, WIN, mel.window, center=False, return_complex=True)[:, 0]
frame = torch.zeros(N_FFT)
off = (N_FFT - WIN) // 2
frame[off : off + WIN] = w[off : off + WIN] * mel.window
check("stft_window_centred_in_512_frame", torch.allclose(torch.fft.rfft(frame), spec, atol=1e-4), window_offset=off,
      note="frame t = samples [t*160, t*160+512); Hann(400, periodic) multiplies samples [t*160+56, t*160+456)")
Path(__file__).parent.joinpath("out/small_stream_audit.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: v for k, v in R.items() if k not in ("checks", "stream_vs_offline_prod_size")}, indent=1, default=str))
