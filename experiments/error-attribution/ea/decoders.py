"""All decoders over the saved logits (out/NAME/logits/<variant>/*.npz) -> out/NAME/hyps_<variant>.json.

    uv run python -m ea.decoders NAME [--variant orig]

Per clip: `greedy` (v2 greedy CTC), `lm` (v2: CTC prefix beam 16 + char 5-gram LM, alpha 0.35 beta 1.5, the app's decoder),
`v3` (the LM beam's 8-best rescored by the v3 attention decoder, weight 1.0, as tingmix.json "rescore"), and N-best lists
for N in 8/16/32 from the LM beam (`nbest_lm`, beam = max(16, N)) and from a plain CTC beam (`nbest_ac`, no LM).
Per region (orig only): `greedy` and `lm` over the fixed25 and overlap logits. N-best lists are kept for offline oracle
diagnosis only; nothing here selects an output with the reference."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import ALPHA, BEAM, BETA, LARGE_ID, LM_FILE, RESCORE, V3_REV, load_json, save_json, video_dir

NS = (8, 16, 32)
_G = {}


def _init():
    import torch
    from embed2asr.lm import CharLM, lm_units
    from embed2asr.pipeline import load_release
    from embed2asr.tokenizer import Tok
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tingmix.models.large.attention import AttDecoder

    torch.set_num_threads(1)
    path = Path(snapshot_download(LARGE_ID, revision=V3_REV))
    _, _, itos, sp, _ = load_release(path)
    meta = json.loads((path / "tingmix.json").read_text(encoding="utf-8"))
    cfg = dict(meta["attention_decoder"])
    cfg.pop("mem")
    dec = AttDecoder(**cfg)
    dec.load_state_dict(load_file(str(path / meta["attention_decoder_file"])))
    _G.update(tok=Tok(itos, sp), lm=CharLM.load(path / LM_FILE), dec=dec.eval(), units=lm_units(itos))


def _clip(job):
    import torch
    from embed2asr.lm import beam_decode
    from tingmix.models.large.attention import ZeroLM, beam_nbest

    path, full = job
    z = np.load(path)
    lp = z["lp"]
    tok, lm, units = _G["tok"], _G["lm"], _G["units"]
    ids = lp.argmax(-1)
    keep = [int(i) for k, i in enumerate(ids) if i != 0 and (k == 0 or i != ids[k - 1])]
    out = {"greedy": tok.decode(keep).replace("臺", "台"), "lm": beam_decode(lp, tok, lm, ALPHA, BETA, BEAM, units=units)}
    if full:
        h8 = beam_nbest(lp, tok, lm, ALPHA, BETA, BEAM, RESCORE["nbest"], units)
        att = _G["dec"].score(torch.from_numpy(z["h"]), lp.shape[0], [x["ids"] for x in h8])
        best = max(zip(h8, att), key=lambda p: p[0]["beam"] + RESCORE["att_weight"] * p[1] + RESCORE["att_mu"] * p[0]["units"])
        out["v3"] = best[0]["text"]
        for n in NS:
            out[f"nbest_lm{n}"] = [x["text"] for x in beam_nbest(lp, tok, lm, ALPHA, BETA, max(BEAM, n), n, units)]
            out[f"nbest_ac{n}"] = [x["text"] for x in beam_nbest(lp, tok, ZeroLM(), 0.0, 0.0, max(BEAM, n), n, units)]
    return Path(path).stem, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--variant", default="orig")
    p.add_argument("--jobs", type=int, default=10)
    a = p.parse_args()
    d = video_dir(a.name)
    meta = load_json(d / "regions.json")
    ld = d / "logits" / a.variant
    jobs = [(str(ld / f"{c['id']}.npz"), True) for r in meta["regions"] for c in r["clips"]]
    if a.variant == "orig":
        jobs += [(str(ld / f"{r['id']}_{k}.npz"), False) for r in meta["regions"] for k in ("fixed25", "overlap")]
    with Pool(a.jobs, initializer=_init) as pool:
        res = dict(pool.map(_clip, jobs))
    save_json(d / f"hyps_{a.variant}.json", res)
    print(a.name, a.variant, len(res), "units")


if __name__ == "__main__":
    main()
