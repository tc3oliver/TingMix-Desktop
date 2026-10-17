"""Shared loader for the TingMix-Large PyTorch reference and the MPS harness (research only)."""

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
LARGE_ID, LARGE_REV = "tc3oliver/TingMix-Large", "75071a1260fb21874f150df6ebd53716ac423f13"
LM_FILE = "lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz"
ALPHA, BETA, BEAM = 0.35, 1.5, 16
CHUNK = int(29.5 * 16000)
MIN_CHUNK = 1600  # embed2asr drops shorter tails


def manifest():
    d = REPO / "tests/fixtures/audio"
    return [dict(json.loads(l), path=d / json.loads(l)["audio"]) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]


def load_wav(path):
    from embed2asr.audio import load_audio

    return load_audio(path)


def chunks(wav):
    return [wav[i : i + CHUNK] for i in range(0, len(wav), CHUNK) if len(wav[i : i + CHUNK]) >= MIN_CHUNK]


def join(parts):
    return " ".join(parts) if " " in "".join(parts) else "".join(parts)


def load_model(device="cpu", dtype=torch.float32):
    """-> (tower, feature_extractor, reader, tok, lm_decode) with the TingMix-Large LoRA delta applied to the tower.
    The tower is built in fp32 (LoRA needs it), then cast to `dtype` and moved to `device`; the reader stays fp32."""
    from embed2asr.lm import CharLM, lm_decoder
    from embed2asr.pipeline import resolve_release
    from tingmix.models.large.model import LargeModel

    path = resolve_release(LARGE_ID, LARGE_REV)
    m = LargeModel.from_pretrained(str(path), device="cpu")
    asr = m.asr
    asr.encoder.tower.to(device=device, dtype=dtype)
    asr.reader.to(device)
    lm = lm_decoder(CharLM.load(path / LM_FILE), ALPHA, BETA, BEAM)
    return asr.encoder.tower, asr.encoder.feature_extractor, asr.reader, asr.tok, lm, path


def mel(fe, wavs):
    inp = fe(wavs, sampling_rate=16000, return_tensors="pt")
    return inp["input_features"], inp["input_features_mask"]
