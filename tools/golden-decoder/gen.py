"""Writes tests/fixtures/decoder/* from the embed2asr reference (pinned in pyproject.toml).

    uv run gen.py --release DIR [--large-lm FILE]

DIR holds units.json and tokenizer_bpe.model of TingMix-Large@75071a1. All input text below is written for this test
(no third-party text). Log-prob matrices are stored sparsely: every row is one `floor` value plus overrides, as the
IEEE bit patterns of the float32 values, so Rust rebuilds exactly the array the reference decoded.
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from embed2asr.ctc import collapse_ctc
from embed2asr.lm import PH, CharLM, beam_decode, lm_text, lm_units
from embed2asr.normalize import norm
from embed2asr.tokenizer import Tok

OUT = Path(__file__).resolve().parents[2] / "tests/fixtures/decoder"
CONFIGS = [  # alpha, beta, beam
    (0.35, 1.5, 16),  # model card
    (0.0, 0.0, 16),  # pure CTC prefix beam search
    (0.6, 2.0, 8),
    (0.35, 1.5, 4),
    (1.0, 0.5, 32),
]
CASES = {
    "zh1": ("zh", "今天天氣很好我們去公園散步"),
    "zh2": ("zh", "台灣的臺北市有很多好吃的夜市"),
    "zh3": ("zh", "請問火車站要怎麼走"),
    "zh4": ("zh", "這個問題需要仔細思考之後再回答你"),
    "zh5": ("zh", "我們明天早上九點在學校門口見面然後一起去圖書館"),
    "en1": ("en", "the quick brown fox jumps over the lazy dog"),
    "en2": ("en", "please turn on the lights in the kitchen"),
    "en3": ("en", "it's hard to recognize speech on a small device"),
    "mix1": ("mixed", "我今天要用 python 寫一個 decoder 程式"),
    "mix2": ("mixed", "這個 model 的 accuracy 還不錯"),
    "mix3": ("mixed", "明天開會討論 project schedule 和 budget"),
    "mix4": ("mixed", "hello 大家好 welcome to the 語音辨識 demo"),
}
CORPUS = """今天天氣很好我們去公園散步
今天天氣不好我們留在家裡看書
台灣的臺北市有很多好吃的夜市
請問火車站要怎麼走
請問捷運站在哪裡
這個問題需要仔細思考之後再回答
我們再討論一次這個問題
他在學校做功課
我明天要去台北開會
我今天要用 python 寫一個 程式
這個 model 的準確率還不錯
明天開會討論 project 的進度
大家好 歡迎來到語音辨識的展示
我們明天早上九點在學校門口見面
然後一起去圖書館讀書
他的中文說得很好
你做得很好
作業已經寫完了
這家店的東西很好吃
我想要一杯咖啡
今天晚上我們一起吃飯
這是一個很有意思的問題
請你再說一次
語音辨識的速度很快""".splitlines()
CONFUSABLE = ["在再", "做作", "的得地", "他她它", "台臺", "已以", "像象", "座坐", "需須", "仔子", "思司", "答達"]
rng = random.Random(20261017)


def bits(a: np.ndarray) -> list[int]:
    return [int(x) for x in a.astype(np.float32).view(np.uint32)]


def make_rows(tok: Tok, ids: list[int], conf: dict[int, list[int]], cjk_ids: list[int]) -> np.ndarray:
    """Returns float32 log-probs [T, V] for a unit sequence: peaked frames, confusers, blanks, noise, a few flat frames."""
    V = len(tok.itos)
    frames: list[dict[int, float]] = []

    def noise(row, k=None):
        for j in rng.sample(range(1, V), k if k is not None else rng.randint(2, 6)):
            row.setdefault(j, rng.uniform(0.0, 4.0))

    def blank_frame():
        r = {0: rng.uniform(9, 12)}
        noise(r)
        frames.append(r)

    blank_frame()
    for i, u in enumerate(ids):
        for _ in range(rng.randint(1, 3)):
            r = {u: rng.uniform(8, 11)}
            if rng.random() < 0.4:
                alts = conf.get(u) or [rng.choice(cjk_ids)]
                r[rng.choice(alts)] = r[u] + rng.uniform(-1.5, 1.2)
            if rng.random() < 0.15:
                r[0] = r[u] + rng.uniform(-2, 2)
            noise(r)
            frames.append(r)
        nxt = ids[i + 1] if i + 1 < len(ids) else None
        for _ in range(rng.randint(1 if nxt == u else 0, 2)):
            blank_frame()
        if rng.random() < 0.1:  # flat frame: 14 near-equal candidates exercise the top-8 pruning
            r = {0: 3.0}
            for j in rng.sample(range(1, V), 14):
                r[j] = rng.uniform(2.0, 3.5)
            frames.append(r)
    blank_frame()
    x = torch.full((len(frames), V), -30.0)
    for t, r in enumerate(frames):
        for j, val in r.items():
            x[t, j] = val
    return x.log_softmax(-1).numpy()


def sparse(lp: np.ndarray) -> list[dict]:
    rows = []
    for row in lp:
        floor = row[np.argmin(row)]
        idx = np.flatnonzero(row != floor)
        assert floor < -14 and len(idx) <= 40
        rows.append({"floor": bits(np.array([floor]))[0], "idx": idx.tolist(), "val": bits(row[idx])})
    return rows


def check_no_boundary_tie(lp: np.ndarray, topk=8):
    """np.argpartition picks arbitrarily among equal values at the top-k boundary; make sure no fixture row has one."""
    for row in lp:
        s = np.sort(row)[::-1]
        assert not (s[topk - 1] == s[topk] and s[topk - 1] > -14), "tie at top-k boundary"


def decode_ref(lp, tok, lm, units, a, b, beam):
    seen = []

    class Rec(Tok):
        def decode(self, ids):
            seen.append(list(ids))
            return super().decode(ids)

    rec = Rec(tok.itos, tok.sp_bytes)
    text = beam_decode(lp, rec, lm, a, b, beam, units=units)
    assert len(seen) == 1
    return {"ids": seen[0], "text": text}


def lm_probes(lm: CharLM, tok: Tok, n=300) -> list[dict]:
    """(prefix, w) -> log P from the reference; prefixes are prefixes of LM-side test sentences plus random CJK chars."""
    r = random.Random(7)
    cjk = [u for u in lm_units(tok.itos) if u]
    texts = [lm_text(norm(t)) for _, t in CASES.values()]
    out = []
    for _ in range(n):
        s = r.choice(texts)
        k = r.randint(0, len(s))
        prefix = s[:k] if r.random() < 0.7 else "".join(r.choice(cjk) for _ in range(r.randint(0, 6)))
        w = r.choice([s[k] if k < len(s) else "\x03", r.choice(cjk), PH, "\x03"])
        out.append({"prefix": prefix, "w": w, "logp": lm.logp(prefix, w)})
    return out


def lm_expected(lm, tok, units, cases, sha, extra):
    res = {}
    for name, c in cases.items():
        res[name] = [decode_ref(c["lp"], tok, lm, units, a, b, bm) for a, b, bm in CONFIGS]
    return {"lm_sha256": sha, **extra, "configs": CONFIGS, "results": res, "logp_probes": lm_probes(lm, tok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--large-lm", type=Path)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    itos = json.loads((a.release / "units.json").read_text(encoding="utf-8"))
    sp = (a.release / "tokenizer_bpe.model").read_bytes()
    tok = Tok(itos, sp)
    (OUT / "units.json").write_bytes((a.release / "units.json").read_bytes())
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    units = lm_units(itos)
    cjk_ids = [i for i, u in enumerate(units) if u]
    conf = {}
    for grp in CONFUSABLE:
        gi = [tok.stoi[c] for c in grp if c in tok.stoi]
        for i in gi:
            conf[i] = [j for j in gi if j != i]

    cases = {}
    for name, (kind, text) in CASES.items():
        ids = tok.encode(norm(text))
        lp = make_rows(tok, ids, conf, cjk_ids)
        check_no_boundary_tie(lp)
        am = lp.argmax(-1)
        g = collapse_ctc(am.tolist(), 0)
        cases[name] = {"kind": kind, "text": text, "target_ids": ids, "lp": lp, "greedy_ids": g, "greedy_text": tok.decode(g)}

    rs = random.Random(11)
    decode_cases = []
    for _ in range(300):
        seq = [rs.randrange(1, len(itos)) for _ in range(rs.randint(0, 12))]
        decode_cases.append({"ids": seq, "text": tok.decode(seq)})
    # the corners of _GAP / split: spaces beside non-ASCII letters, empty and blank-only input
    for pieces in (["▁erkoç", "▁and"], ["▁a", "你", "▁b"], ["你", "▁b", "▁c", "好"]):
        seq = [tok.stoi[p] for p in pieces if p in tok.stoi]
        decode_cases.append({"ids": seq, "text": tok.decode(seq)})

    cj = {
        "format": "tingmix-decoder-golden/1 (row = floor bits + sparse overrides, all float32 bit patterns)",
        "reference": "embed2asr@b58275785501c084610b1570f1a446a31da5f933",
        "units_sha256": sha((a.release / "units.json").read_bytes()),
        "vocab_size": len(itos),
        "lm_units": units,
        "decode_cases": decode_cases,
        "cases": {
            n: {"kind": c["kind"], "text": c["text"], "T": int(c["lp"].shape[0]), "rows": sparse(c["lp"]),
                "greedy_ids": c["greedy_ids"], "greedy_text": c["greedy_text"]}
            for n, c in cases.items()
        },
    }
    # round trip: dense rebuild from the sparse form is bit-identical
    for n, c in cases.items():
        dense = np.empty_like(c["lp"])
        for t, r in enumerate(cj["cases"][n]["rows"]):
            dense[t] = np.array([r["floor"]], dtype=np.uint32).view(np.float32)[0]
            dense[t, r["idx"]] = np.array(r["val"], dtype=np.uint32).view(np.float32)
        assert dense.tobytes() == c["lp"].tobytes()
    (OUT / "cases.json").write_text(json.dumps(cj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # small LM: order-5 KN over CORPUS, same file format as the release LM
    small = CharLM([lm_text(norm(s)) for s in CORPUS], 5)
    sha_small = small.save(OUT / "lm_small.charlm.json.gz")
    small = CharLM.load(OUT / "lm_small.charlm.json.gz")
    exp = lm_expected(small, tok, units, cases, sha_small, {})
    (OUT / "expected_small_lm.json").write_text(json.dumps(exp, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if a.large_lm:
        t0 = time.time()
        lm = CharLM.load(a.large_lm)
        load_s = time.time() - t0
        exp = lm_expected(lm, tok, units, cases, sha(a.large_lm.read_bytes()), {"python_load_seconds": round(load_s, 1)})
        (OUT / "expected_large_lm.json").write_text(json.dumps(exp, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for n, c in cases.items():
        print(n, c["lp"].shape[0], "greedy:", c["greedy_text"])


if __name__ == "__main__":
    main()
