"""Build the single-file MLX artifact: audio_tower (base + LoRA merged in fp32, MLX layouts) + reader. No text/vision towers.

Output keys: `audio_tower.*` (mlx-vlm gemma4 AudioEncoder names) and `reader.*` (tingmix_mlx.Reader names), all fp32.
Lower precisions are produced at load time (tingmix_mlx.load), so the file stays the fp32 source of truth."""
import hashlib
import json

import numpy as np
from common import *
from safetensors import safe_open
from safetensors.numpy import save_file

SCALE = 32 / 16  # lora_alpha / lora_r from tingmix.json


def f32(t):
    return t.float().numpy() if hasattr(t, "float") else t.astype(np.float32)


def main():
    base = safe_open(str(BASE_DIR / "model.safetensors"), "pt")
    delta = safe_open(str(LARGE_DIR / "encoder_delta.safetensors"), "pt")
    cfg = json.loads((LARGE_DIR / "tingmix.json").read_text())["encoder_delta"]
    assert (cfg["mode"], cfg["lora_r"], cfg["lora_alpha"]) == ("lora", 16, 32) and SCALE == 2
    out, merged = {}, set()
    for k in base.keys():
        if not k.startswith("audio_tower."):
            continue
        t = f32(base.get_tensor(k))
        if k.endswith(".linear.weight") and (dk := k.removeprefix("audio_tower.").removesuffix("weight") + "lora_a") in delta.keys():
            a, b = f32(delta.get_tensor(dk)), f32(delta.get_tensor(dk.replace("lora_a", "lora_b")))
            t = t + SCALE * (b @ a)  # fp32 merge: W' = W + (alpha/r) * B A
            merged.add(dk)
        if "subsample_conv_projection" in k and k.endswith("conv.weight"):
            t = t.transpose(0, 2, 3, 1)  # OIHW -> OHWI
        if k.endswith("depthwise_conv1d.weight"):
            t = t.transpose(0, 2, 1)  # (C,1,K) -> (C,K,1)
        out[k] = np.array(t, order="C")
    assert len(merged) * 2 == len(list(delta.keys())) == 192, (len(merged), len(list(delta.keys())))
    rd = safe_open(str(LARGE_DIR / "model.safetensors"), "pt")
    for k in rd.keys():
        t = f32(rd.get_tensor(k))
        if k.startswith("blocks.") and k.endswith("conv.weight"):
            t = t.transpose(0, 2, 1)  # (O,I,K) -> (O,K,I)
        if k.startswith("rnn."):
            continue
        out["reader." + k.replace("pre.0.", "pre_norm.").replace("pre.1.", "pre_lin.")] = np.array(t, order="C")
    # GRU -> mlx layout per layer/direction: Wx=w_ih, Wh=w_hh, b=b_ih+[b_hh_r,b_hh_z,0], bhn=b_hh_n (gate order r,z,n)
    for l in (0, 1):
        for d, sfx in (("f", ""), ("b", "_reverse")):
            g = lambda n: f32(rd.get_tensor(f"rnn.{n}_l{l}{sfx}"))
            bhh = g("bias_hh")
            H = bhh.shape[0] // 3
            b = g("bias_ih").copy()
            b[: 2 * H] += bhh[: 2 * H]
            p = f"reader.rnn{l}_{d}."
            out[p + "Wx"], out[p + "Wh"], out[p + "b"], out[p + "bhn"] = g("weight_ih"), g("weight_hh"), b, bhh[2 * H :].copy()
    OUT.mkdir(exist_ok=True)
    meta = {"format": "tingmix-large-mlx", "large_rev": LARGE_REV, "base_rev": BASE_REV, "lora": "merged fp32 scale 2.0",
            "tensors": str(len(out))}
    dst = OUT / "tingmix-large-mlx-fp32.safetensors"
    save_file(out, str(dst), metadata=meta)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    (OUT / "tingmix-large-mlx-fp32.sha256").write_text(f"{sha}  {dst.name}\n")
    n_tower = sum(v.size for k, v in out.items() if k.startswith("audio_tower."))
    n_read = sum(v.size for k, v in out.items() if k.startswith("reader."))
    print(f"{dst} {dst.stat().st_size/1e6:.1f} MB sha256={sha}\ntower params {n_tower:,} reader params {n_read:,} tensors {len(out)}")


main()
