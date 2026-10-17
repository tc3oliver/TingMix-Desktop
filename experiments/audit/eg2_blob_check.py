"""Download google/embeddinggemma-2 model.safetensors at the pinned revision (into $HF_HOME), verify its sha256 against the
Hub LFS oid, check the audio_tower tensors form one contiguous byte range, and dump the clipping scalars. Writes out/eg2_blob.json."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
OID = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
p = Path(hf_hub_download("google/embeddinggemma-2", "model.safetensors", revision=REV))
h = hashlib.sha256()
with p.open("rb") as f:
    while b := f.read(1 << 24):
        h.update(b)
n = struct.unpack("<Q", p.open("rb").read(8))[0]
hdr = json.loads(p.open("rb").read(8 + n)[8:])
a = {k: v for k, v in hdr.items() if k.startswith("audio_tower.")}
lo, hi = min(v["data_offsets"][0] for v in a.values()), max(v["data_offsets"][1] for v in a.values())
others_in_range = [k for k, v in hdr.items() if k != "__metadata__" and not k.startswith("audio_tower.") and v["data_offsets"][0] < hi and v["data_offsets"][1] > lo]
clip = {}
with p.open("rb") as f:
    for k, v in a.items():
        if k.endswith(("input_min", "input_max", "output_min", "output_max")):
            f.seek(8 + n + v["data_offsets"][0])
            raw = np.frombuffer(f.read(2), dtype=np.uint16).astype(np.uint32) << 16  # bf16 -> f32
            clip[k] = float(raw.view(np.float32)[0])
vals = np.array(list(clip.values()))
res = {
    "path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest(), "sha256_matches_hub_lfs_oid": h.hexdigest() == OID,
    "header_bytes": n, "data_start": 8 + n,
    "audio_tower_byte_range_in_data": [lo, hi], "audio_tower_bytes": hi - lo, "audio_tower_tensor_bytes_sum": sum(v["data_offsets"][1] - v["data_offsets"][0] for v in a.values()),
    "other_tensors_overlapping_range": others_in_range,
    "n_clip_scalars": len(clip), "clip_all_finite": bool(np.isfinite(vals).all()), "clip_min": float(vals.min()), "clip_max": float(vals.max()),
    "clip_abs_min_nonzero": float(np.abs(vals[vals != 0]).min()) if (vals != 0).any() else None,
}
Path(__file__).parent.joinpath("out/eg2_blob.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
