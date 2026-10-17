"""Shared loading for the Core ML study: fp32 CPU TingMix-Large (LoRA merged into the base weights), fixtures, helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HOME", "$HF_HOME")

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
FIXTURES = REPO / "tests/fixtures/audio"
ENC_ID, ENC_REV = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
TM_ID, TM_REV = "tc3oliver/TingMix-Large", "75071a1260fb21874f150df6ebd53716ac423f13"


def tm_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(TM_ID, revision=TM_REV, allow_patterns=["*.json", "*.safetensors", "*.model"]))


def load_tower(merge: bool = True):
    """fp32 audio_tower with the TingMix-Large LoRA delta applied; merge=True folds W += (alpha/r) B A into the base."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tingmix.models.large.encoder import LoRALinear, load_delta
    from transformers import AutoModel

    snapshot_download(ENC_ID, revision=ENC_REV)  # no-op when cached
    full = AutoModel.from_pretrained(ENC_ID, revision=ENC_REV, dtype=torch.float32)
    tower = full.audio_tower
    del full
    d = tm_dir()
    meta = json.loads((d / "tingmix.json").read_text())
    load_delta(tower, meta["encoder_delta"], load_file(str(d / meta["encoder_delta_file"])))
    if merge:
        with torch.no_grad():
            for mod in list(tower.modules()):
                lin = getattr(mod, "linear", None)
                if isinstance(lin, LoRALinear):
                    lin.base.weight += lin.scale * (lin.lora_b @ lin.lora_a)
                    mod.linear = lin.base
    return tower.requires_grad_(False).eval()


def load_reader():
    from embed2asr.pipeline import load_release
    from embed2asr.reader import Reader

    sd, rcfg, itos, _, _ = load_release(tm_dir())
    r = Reader(**rcfg)
    r.load_state_dict(sd)
    return r.requires_grad_(False).eval(), itos


def mel(name: str):
    """Fixture wav -> (input_features [1,F,128] f32, mask [1,F] bool) from the pinned feature extractor."""
    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(ENC_ID, revision=ENC_REV)
    w, sr = sf.read(FIXTURES / name, dtype="float32")
    assert sr == 16000
    i = fe([w], sampling_rate=16000, return_tensors="pt")
    return i["input_features"], i["input_features_mask"]


def rel_err(a, b) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), rel_l2=float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)))
