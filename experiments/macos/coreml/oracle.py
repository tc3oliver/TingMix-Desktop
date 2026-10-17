"""Own PyTorch CPU fp32 oracle (until reference's dumps land): per-stage tensors for fixtures."""
import sys, time, numpy as np, torch
from common import OUT, load_tower, load_reader, mel

names = sys.argv[1:] or ["zh_02s.wav", "mix_08s.wav", "mix_29s.wav"]
tower, (reader, itos) = load_tower(), load_reader()
OUT.mkdir(exist_ok=True)
for n in names:
    x, m = mel(n)
    with torch.inference_mode():
        h, om = tower.subsample_conv_projection(x, m)
        t = time.time(); o = tower(x, m); dt = time.time() - t
        feats, lens = o.last_hidden_state, o.attention_mask.sum(1)
        logits = reader(feats, lens)
    ids = logits[0, : lens[0]].argmax(-1).tolist()
    txt = "".join(itos[i] for i, p in zip(ids, [-1] + ids[:-1]) if i != 0 and i != p)
    np.savez(OUT / f"oracle_{n[:-4]}.npz", mel=x.numpy(), mask=m.numpy(), sub=h.numpy(), sub_mask=om.numpy(), feats=feats.numpy(), lens=lens.numpy(), logits=logits.numpy())
    print(n, "mel", tuple(x.shape), "valid", int(m.sum()), "sub", tuple(h.shape), "feats", tuple(feats.shape), "lens", lens.tolist(), f"{dt:.2f}s", repr(txt[:80]))
