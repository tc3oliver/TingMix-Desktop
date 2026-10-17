"""MLX vs PyTorch CPU fp32 oracle (ref_torch.py): features, logits, greedy and beam+LM text, per precision.
Usage: uv run python parity.py  -> benchmarks/apple/mlx/parity.{json,md}"""
import json

import mlx.core as mx
import numpy as np
import torch
from common import *

import tingmix_mlx as T
from embed2asr.ctc import collapse_ctc
from embed2asr.lm import CharLM, lm_decoder
from embed2asr.tokenizer import Tok

DEST = ROOT / "benchmarks" / "apple" / "mlx"
DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def cer(h, r):
    return ed(h, r) / max(len(r), 1)


def main():
    itos = json.loads((LARGE_DIR / "units.json").read_text())
    tok = Tok(itos, (LARGE_DIR / "tokenizer_bpe.model").read_bytes())
    beam = lm_decoder(CharLM.load(LM_FILE), LM_ALPHA, LM_BETA, BEAM)
    texts = json.loads((OUT / "ref" / "texts.json").read_text())
    fx = {r["id"]: r for r in fixtures()}
    rows = []
    for dname, dt in DTYPES.items():
        m = T.load(dt)
        for fid, ref in texts.items():
            z = np.load(OUT / "ref" / f"{fid}.npz")
            n = ref["frames"]
            f, lg = m(mx.array(z["mel"]), mx.array(z["mask"]).astype(mx.bool_), n)
            f, lg = np.array(f.astype(mx.float32))[0], np.array(lg.astype(mx.float32))[0]
            rf, rl = z["feats"], z["logits"]
            cos = (f * rf).sum(1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(rf, axis=1))
            ids = lg.argmax(-1).tolist()
            greedy = tok.decode(collapse_ctc(ids, 0))
            b = beam(torch.from_numpy(lg)[None], torch.tensor([n]), tok)[0]
            rows.append(dict(
                dtype=dname, id=fid, frames=n,
                feat_max_abs=float(np.abs(f - rf).max()), feat_ref_max_abs=float(np.abs(rf).max()),
                feat_rel_l2=float(np.linalg.norm(f - rf) / np.linalg.norm(rf)),
                cos_mean=float(cos.mean()), cos_min=float(cos.min()),
                logit_max_abs=float(np.abs(lg - rl).max()),
                argmax_agree=float((lg.argmax(-1) == rl.argmax(-1)).mean()),
                greedy=greedy, greedy_match=greedy == ref["greedy"], greedy_cer_vs_oracle=cer(greedy, ref["greedy"]),
                beam=b, beam_match=b == ref["beam"], beam_cer_vs_oracle=cer(b, ref["beam"]),
                cer_vs_truth=cer(b, fx[fid]["text"]), oracle_cer_vs_truth=cer(ref["beam"], fx[fid]["text"]),
            ))
            print(dname, fid, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items() if k.startswith(("feat_max", "cos_min", "logit", "greedy_m", "beam_m"))}, flush=True)
        del m
    DEST.mkdir(parents=True, exist_ok=True)
    meta = dict(mlx=mx.__version__ if hasattr(mx, "__version__") else "0.32.4", oracle="PyTorch 2.14.1 CPU fp32 (ref_torch.py), embed2asr b582757 / tingmix f44cbdb", artifact_sha256=(OUT / "tingmix-large-mlx-fp32.sha256").read_text().split()[0], large_rev=LARGE_REV, base_rev=BASE_REV)
    (DEST / "parity.json").write_text(json.dumps(dict(meta=meta, rows=rows), ensure_ascii=False, indent=1))
    L = ["| dtype | id | feat max-abs | feat rel-L2 | cos min | logit max-abs | argmax agree | greedy == oracle | beam == oracle | beam CER vs truth (oracle) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dtype']} | {r['id']} | {r['feat_max_abs']:.2e} | {r['feat_rel_l2']:.2e} | {r['cos_min']:.6f} | {r['logit_max_abs']:.2e} | {r['argmax_agree']:.4f} | {r['greedy_match']} | {r['beam_match']} | {r['cer_vs_truth']:.3f} ({r['oracle_cer_vs_truth']:.3f}) |")
    (DEST / "parity.md").write_text("\n".join(L) + "\n")


main()
