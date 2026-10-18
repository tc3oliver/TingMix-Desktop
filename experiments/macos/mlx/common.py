"""Shared paths, pins and fixture helpers for the MLX experiment."""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
FIX = ROOT / "tests" / "fixtures" / "audio"
HF = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
LARGE_REV = "75071a1260fb21874f150df6ebd53716ac423f13"
BASE_REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LARGE_DIR = HF / "models--tc3oliver--TingMix-Large" / "snapshots" / LARGE_REV
BASE_DIR = HF / "models--google--embeddinggemma-2" / "snapshots" / BASE_REV
LM_FILE = LARGE_DIR / "lm" / "lmCN_cv_parl_ascend_ntuml.charlm.json.gz"
LM_ALPHA, LM_BETA, BEAM = 0.35, 1.5, 16  # tingmix.json recommended decoding


def fixtures():
    rows = [json.loads(l) for l in (FIX / "manifest.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        w, sr = sf.read(FIX / r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, (r["id"], sr)
        r["wav"] = w.mean(1).astype(np.float32)
    return rows
