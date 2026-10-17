"""CPU fp32 oracle: dump mel, mel mask, encoder features, reader logits and texts for every fixture clip.

Usage: uv run python dump.py   (writes out/<id>.npz, out/results.json, out/MANIFEST.sha256)"""

import hashlib
import json
import platform
import sys

import numpy as np
import torch
import embed2asr, transformers

import common as C
from embed2asr.ctc import collapse_ctc

OUT = C.Path(__file__).parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())
    tower, fe, reader, tok, lm, path = C.load_model("cpu", torch.float32)
    greedy_ids_all, res = {}, {}
    for clip in C.manifest():
        arrs, g_parts, b_parts = {}, [], []
        wav = C.load_wav(clip["path"])
        for k, w in enumerate(C.chunks(wav)):
            m, mm = C.mel(fe, [w])
            with torch.inference_mode():
                o = tower(m, mm, return_dict=True)
                feats, lens = o.last_hidden_state, o.attention_mask.sum(1)
                logits = reader(feats.float(), lens)
            n = int(lens[0])
            ids = collapse_ctc(logits[0, :n].argmax(-1).tolist(), 0)
            g_parts.append(tok.decode(ids))
            b_parts.append(lm(logits, lens, tok)[0])
            p = f"c{k}_"
            arrs |= {p + "mel": m.numpy(), p + "mel_mask": mm.numpy(), p + "features": feats[0, :n].numpy(),
                     p + "logits": logits[0, :n].numpy(), p + "greedy_ids": np.array(ids, dtype=np.int64),
                     p + "n_frames": np.array(n)}
        np.savez(OUT / f"{clip['id']}.npz", **arrs)
        res[clip["id"]] = {"ref": clip["text"], "greedy": C.join(g_parts), "beam_lm": C.join(b_parts), "n_chunks": len(g_parts)}
        print(clip["id"], res[clip["id"]], flush=True)
    meta = {"device": "cpu fp32", "torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0],
            "machine": platform.platform(), "threads": torch.get_num_threads(), "lm": {"alpha": C.ALPHA, "beta": C.BETA, "beam": C.BEAM},
            "model": f"{C.LARGE_ID}@{C.LARGE_REV}", "encoder": "google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b",
            "command": "cd experiments/macos/reference && uv run python dump.py", "texts": res}
    (OUT / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in sorted(OUT.glob("*.npz")) + [OUT / "results.json"]]
    (OUT / "MANIFEST.sha256").write_text("\n".join(sums) + "\n")


main()
