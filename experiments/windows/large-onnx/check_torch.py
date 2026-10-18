"""Torch-level check of the dynamic modules against the CPU fp32 oracle dumps (before exporting)."""
import sys
import numpy as np
import torch
from common import REF, load_reader, load_tower, rel_err
from modules import EncoderReaderDyn, blocked_mask

tower, (reader, itos) = load_tower(), load_reader()
# the in-graph mask must equal the HF mask for an all-valid input
from tingmix.models.large.encoder import _mask_and_pos
for T in (1, 5, 12, 13, 24, 25, 188, 738):
    am, _ = _mask_and_pos(tower, torch.zeros(1, T, tower.config.hidden_size), torch.tensor([T]))
    nb = (T + 11) // 12
    mine = blocked_mask(T, nb, 12, "cpu")
    q = (torch.arange(nb)[:, None] * 12 + torch.arange(12)[None]) < T  # padding query rows are sliced off; HF masks them out
    ok = am is not None and bool(((am == mine) | ~q[None, None, :, :, None]).all())
    print("mask T=%d equal=%s" % (T, ok))
    assert ok
model = EncoderReaderDyn(tower, reader).eval()
names = sys.argv[1:] or ["zh_02s", "en_08s", "mix_08s"]
for n in names:
    d = np.load(REF / f"{n}.npz")
    k = 0
    while f"c{k}_mel" in d.files:
        mel = torch.from_numpy(d[f"c{k}_mel"])[:, d[f"c{k}_mel_mask"][0]]
        with torch.inference_mode():
            lg = model(mel).numpy()[0]
        ref = d[f"c{k}_logits"]
        e = rel_err(lg, ref)
        print(n, k, tuple(mel.shape), "ids equal", bool((lg.argmax(-1) == ref.argmax(-1)).all()), e)
        k += 1
