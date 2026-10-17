"""Stage 5: complete Encoder + Reader (mel -> logits) as one Core ML model; greedy CTC text vs oracle."""
import sys, numpy as np, torch
from common import OUT, load_tower, load_reader
from modules import EncoderReader
from stages import drive
from harness import record
tower = load_tower(); reader, itos = load_reader()
names = sys.argv[1:] or ["mix_08s", "mix_29s"]

def text(ids):
    out, prev = [], -1
    for i in ids:
        if i != 0 and i != prev: out.append(itos[i])
        prev = i
    return "".join(out)

allres = {}
for n in names:
    o = np.load(OUT / f"oracle_{n}.npz")
    mel, logits = o["mel"], o["logits"]
    ref_ids, ref_text = logits.argmax(-1), text(logits.argmax(-1)[0])
    def extra(out, ref_ids=ref_ids, ref_text=ref_text):
        lg = np.asarray(next(iter(out.values())))
        return dict(greedy_id_agree=float((lg.argmax(-1) == ref_ids).mean()), text_equal=text(lg.argmax(-1)[0]) == ref_text)
    allres[n] = drive("stage5", EncoderReader(tower, reader, mel.shape[1]), f"s5_encreader_F{mel.shape[1]}", {"mel": mel}, logits, extra=extra, n=5, methods=("trace",))
    allres[n]["oracle_text"] = ref_text
record("stage5_encoder_reader", dict(input_fixtures=names, results=allres))
