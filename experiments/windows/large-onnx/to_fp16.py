"""fp32 tingmix-large.onnx -> fp16 weights/activations (graph inputs and outputs stay fp32).

    uv run python to_fp16.py [out/tingmix-large.onnx] [out/tingmix-large-fp16.onnx]

The export already uses the RMSNorm pre-scale (x/16) so activations of ~300 do not overflow; the softcap-tanh attention logits and the
softmax stay finite in fp16 (mask fill -3e4). Nodes fp16 cannot represent are kept in fp32 by the converter's block list."""
import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

from common import OUT

src = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "tingmix-large.onnx"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "tingmix-large-fp16.onnx"
m = onnx.load(str(src))
m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
onnx.save(m16, str(dst))
h = hashlib.sha256(dst.read_bytes()).hexdigest()
meta = {"file": dst.name, "bytes": dst.stat().st_size, "sha256": h, "from": src.name, "from_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
dst.with_suffix(".json").write_text(json.dumps(meta, indent=1))
print(meta)
