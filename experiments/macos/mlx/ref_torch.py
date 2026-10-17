"""PyTorch CPU fp32 oracle (own run): mel, encoder features, reader logits, greedy and beam+LM text per fixture."""
import json
import time

import numpy as np
import torch
from common import *

from embed2asr.ctc import collapse_ctc
from embed2asr.lm import CharLM, lm_decoder
from tingmix.models.large.model import LargeModel

torch.set_num_threads(os.cpu_count())
m = LargeModel.from_pretrained(str(LARGE_DIR), device="cpu")
asr = m.asr
assert asr.encoder.dtype == torch.float32
beam = lm_decoder(CharLM.load(LM_FILE), LM_ALPHA, LM_BETA, BEAM)
(OUT / "ref").mkdir(parents=True, exist_ok=True)
res = {}
with torch.inference_mode():
    for r in fixtures():
        wav = r["wav"]
        inp = asr.encoder.feature_extractor([wav], sampling_rate=16000, return_tensors="pt")
        mel, mask = inp["input_features"], inp["input_features_mask"]
        t = time.perf_counter()
        o = asr.encoder.tower(mel, mask, return_dict=True)
        feats, lens = o.last_hidden_state, o.attention_mask.sum(1)
        t_enc = time.perf_counter() - t
        logits = asr.reader(feats.float(), lens)
        ids = logits.argmax(-1)[0, : int(lens[0])].tolist()
        greedy = asr.tok.decode(collapse_ctc(ids, 0))
        b = beam(logits, lens, asr.tok)[0]
        n = int(lens[0])
        np.savez(OUT / "ref" / f"{r['id']}.npz", mel=mel.numpy(), mask=mask.numpy(), feats=feats[0, :n].numpy(), logits=logits[0, :n].numpy())
        res[r["id"]] = dict(ref=r["text"], greedy=greedy, beam=b, frames=n, mel_frames=int(mask.sum()), enc_s=round(t_enc, 3))
        print(r["id"], res[r["id"]], flush=True)
(OUT / "ref" / "texts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
