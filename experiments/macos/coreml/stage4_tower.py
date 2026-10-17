"""Stage 4: full audio tower from log-mel, one model per frame-count bucket; plus token agreement through the torch Reader."""
import sys, numpy as np, torch
from common import OUT, load_tower, load_reader
from modules import TowerFull
from stages import drive
from harness import record
tower = load_tower(); reader, itos = load_reader()
names = sys.argv[1:] or ["mix_08s", "mix_29s"]
allres = {}
for n in names:
    o = np.load(OUT / f"oracle_{n}.npz")
    mel, feats, logits = o["mel"], o["feats"], o["logits"]
    ref_ids = logits.argmax(-1)
    def extra(out, feats=feats, ref_ids=ref_ids):  # Core ML features -> torch Reader -> greedy ids vs oracle ids
        f = torch.from_numpy(np.asarray(next(iter(out.values())), np.float32))
        with torch.no_grad():
            lg = reader(f, torch.tensor([f.shape[1]]))
        return dict(reader_greedy_id_agree=float((lg.argmax(-1).numpy() == ref_ids).mean()))
    mod = TowerFull(tower, mel.shape[1])
    allres[n] = drive("stage4", mod, f"s4_tower_F{mel.shape[1]}", {"mel": mel}, feats, extra=extra, n=5)
record("stage4_tower", dict(input_fixtures=names, results=allres))
