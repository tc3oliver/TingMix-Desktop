"""MLX fp32 vs the reference task's oracle dumps (experiments/macos/reference/out), and texts vs expected_texts.json."""
import json
import numpy as np, mlx.core as mx, torch
from common import *
import tingmix_mlx as T
from embed2asr.ctc import collapse_ctc
from embed2asr.lm import CharLM, lm_decoder
from embed2asr.tokenizer import Tok

R = ROOT / "experiments/macos/reference/out"
exp = json.loads((ROOT / "tests/fixtures/large/expected_texts.json").read_text())["clips"]
tok = Tok(json.loads((LARGE_DIR / "units.json").read_text()), (LARGE_DIR / "tokenizer_bpe.model").read_bytes())
beam = lm_decoder(CharLM.load(LM_FILE), LM_ALPHA, LM_BETA, BEAM)
m = T.load()
for cid, e in exp.items():
    z = np.load(R / f"{cid}.npz"); g, b, mf, ml = [], [], 0.0, 0.0
    for k in range(e["n_chunks"]):
        n = int(z[f"c{k}_n_frames"])
        f, lg = m(mx.array(z[f"c{k}_mel"]), mx.array(z[f"c{k}_mel_mask"]), n)
        f, lg = np.array(f)[0], np.array(lg)[0]
        mf, ml = max(mf, np.abs(f - z[f"c{k}_features"]).max()), max(ml, np.abs(lg - z[f"c{k}_logits"]).max())
        g.append(tok.decode(collapse_ctc(lg.argmax(-1).tolist(), 0)))
        b.append(beam(torch.from_numpy(lg)[None], torch.tensor([n]), tok)[0])
    j = lambda p: ("" if " " not in "".join(p) else " ").join(p)
    print(cid, f"feat {mf:.1e} logit {ml:.1e} greedy {j(g) == e['greedy']} beam {j(b) == e['beam_lm']}")
