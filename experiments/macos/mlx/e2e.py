"""Raw wav -> numpy mel (mlx-vlm port, no torch) -> MLX tower+reader -> greedy text vs oracle greedy. -> benchmarks/apple/mlx/e2e.json"""
import json

import mlx.core as mx
import numpy as np
from common import *

import tingmix_mlx as T
from embed2asr.ctc import collapse_ctc
from embed2asr.tokenizer import Tok
from mlx_vlm.models.gemma4.audio_feature_extractor import Gemma4AudioFeatureExtractor

tok = Tok(json.loads((LARGE_DIR / "units.json").read_text()), (LARGE_DIR / "tokenizer_bpe.model").read_bytes())
texts = json.loads((OUT / "ref" / "texts.json").read_text())
ext, out = Gemma4AudioFeatureExtractor(), []
for dname, dt in (("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)):
    m = T.load(dt)
    for r in fixtures():
        o = ext(r["wav"][: 30 * 16000])
        mel, mk = o["input_features"][0], o["input_features_mask"][0]
        n = int(texts[r["id"]]["frames"])
        _, lg = m(mx.array(mel)[None], mx.array(mk)[None], n)
        t = tok.decode(collapse_ctc(np.array(mx.argmax(lg, -1))[0].tolist(), 0))
        out.append(dict(dtype=dname, id=r["id"], text=t, oracle_greedy=texts[r["id"]]["greedy"], match=t == texts[r["id"]]["greedy"]))
        print(dname, r["id"], out[-1]["match"])
(ROOT / "benchmarks/apple/mlx/e2e.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
