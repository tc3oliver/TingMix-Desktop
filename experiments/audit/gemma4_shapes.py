"""Instantiate the EmbeddingGemma-2 audio tower from the pinned config.json (random weights, fp32, CPU), compare its
state_dict keys/shapes with the real safetensors header (no weight download), trace per-stage tensor shapes and measure
the input->output dependency (receptive field). Writes out/gemma4_shapes.json.

  uv run python gemma4_shapes.py            # needs out/header_embeddinggemma-2_model.safetensors.json (st_headers.py)
"""
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
OUT = Path(__file__).parent / "out"
cache = OUT / "eg2_cfg"
cache.mkdir(parents=True, exist_ok=True)
for f in ("config.json", "preprocessor_config.json"):
    if not (cache / f).exists():
        urllib.request.urlretrieve(f"https://huggingface.co/google/embeddinggemma-2/resolve/{REV}/{f}", cache / f)

cfg = AutoConfig.from_pretrained(cache).audio_config
fe = AutoFeatureExtractor.from_pretrained(cache)
torch.manual_seed(0)
tower = AutoModel.from_config(cfg).float().eval()
res = {"torch": torch.__version__, "class": type(tower).__name__}

# 1. structure vs the real header
hdr = json.loads((OUT / "header_embeddinggemma-2_model.safetensors.json").read_text())["header"]
real = {k.removeprefix("audio_tower."): tuple(v["shape"]) for k, v in hdr.items() if k.startswith("audio_tower.")}
mine = {k: tuple(v.shape) for k, v in tower.state_dict().items() if "softcap" not in k}
res["params"] = sum(p.numel() for p in tower.parameters())
res["buffers_persistent"] = sum(b.numel() for k, b in tower.named_buffers() if k in mine)
res["missing_in_module"] = sorted(set(real) - set(mine))
res["extra_in_module"] = sorted(set(mine) - set(real))
res["shape_mismatch"] = sorted(k for k in set(real) & set(mine) if real[k] != mine[k])
res["n_tensors_header"] = len(real)

# 2. shape trace on a real feature-extractor output
trace = []
def hook(name):
    def f(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        trace.append((name, [tuple(x.shape) for x in i if torch.is_tensor(x)], tuple(o.shape)))
    return f
for n in ("subsample_conv_projection.layer0", "subsample_conv_projection.layer1", "subsample_conv_projection.input_proj_linear",
          "layers.0", "layers.0.feed_forward1", "layers.0.self_attn", "layers.0.lconv1d", "layers.0.feed_forward2", "layers.11", "output_proj"):
    tower.get_submodule(n).register_forward_hook(hook(n))
res["shape_traces"] = {}
for secs in (1.0, 3.0, 29.5, 30.0, 35.0):
    n = int(secs * 16000)
    x = (np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32)
    inp = fe([x], sampling_rate=16000, return_tensors="pt")
    trace.clear()
    with torch.inference_mode():
        o = tower(inp["input_features"], inp["input_features_mask"], return_dict=True)
    res["shape_traces"][str(secs)] = {
        "samples": n, "input_features": list(inp["input_features"].shape), "mask_true": int(inp["input_features_mask"].sum()),
        "last_hidden_state": list(o.last_hidden_state.shape), "out_mask_true": int(o.attention_mask.sum()),
        "stages": [[a, b, list(c)] for a, b, c in (trace[:10] if secs == 29.5 else [])],
    }

# 3. frame-count formula check against many lengths
bad = []
for n in list(range(800, 20000, 97)) + [16000, 160000, 472000, 480000]:
    x = np.zeros(n, np.float32) + 0.01
    inp = fe([x], sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        o = tower(inp["input_features"], inp["input_features_mask"], return_dict=True)
    npad = math.ceil(min(n, 480000) / 128) * 128
    mel_valid = int(inp["input_features_mask"].sum())
    if o.last_hidden_state.shape[1] != math.ceil(math.ceil(inp["input_features"].shape[1] / 2) / 2):
        bad.append(n)
res["frame_formula_violations"] = bad

# 4. receptive field: perturb one mel frame, find affected output frames (random weights; structural property)
T = 400
mel = torch.randn(1, T, 128)
mask = torch.ones(1, T, dtype=torch.bool)
with torch.inference_mode():
    base = tower(mel, mask, return_dict=True).last_hidden_state
    rf = {}
    for j in (200, 201, 203):
        m2 = mel.clone(); m2[0, j] += 1.0
        d = (tower(m2, mask, return_dict=True).last_hidden_state - base).abs().amax(-1)[0]
        idx = (d > 1e-6).nonzero().flatten()
        rf[str(j)] = [int(idx.min()), int(idx.max())]
res["perturb_mel_frame_to_output_frames"] = rf
res["output_frames_for_400_mel"] = int(base.shape[1])
(OUT / "gemma4_shapes.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "shape_traces"}, indent=1))
for k, v in res["shape_traces"].items():
    print(k, {a: b for a, b in v.items() if a != "stages"})
for s in res["shape_traces"]["29.5"]["stages"]:
    print(s)
