"""Shared paths, scoring and alignment for the error-attribution harness (research only)."""

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SR = 16000
LARGE_ID = "tc3oliver/TingMix-Large"
V3_REV = "aae0ca252bf4a4cf725cd43861799c92acaf7b08"  # v3: v2 encoder/Reader/LM + attention decoder
V2_REV = "75071a1260fb21874f150df6ebd53716ac423f13"  # the revision the app ships
LM_FILE = "lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz"
ALPHA, BETA, BEAM = 0.35, 1.5, 16  # tingmix.json recommended decoding
RESCORE = dict(att_weight=1.0, att_mu=0.0, nbest=8)  # tingmix.json "rescore" of the v3 revision

_LATIN = re.compile(r"[a-z0-9']")


def out_file(name: str) -> Path:
    """out/<name>, or out/<stem>-subtitles<ext> when EA_REF=subtitles (sensitivity runs never overwrite the main results)."""
    import os

    p = OUT / name
    return p.with_name(f"{p.stem}-subtitles{p.suffix}") if os.environ.get("EA_REF") == "subtitles" else p


def video_dir(name: str) -> Path:
    return OUT / name


def read_wav(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    assert sr == SR and x.ndim == 1, f"{path}: need 16 kHz mono"
    return x


def write_wav(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, SR, subtype="PCM_16")  # 16-bit plays in every browser (review page)


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


_cc = None


def to_trad(s: str) -> str:
    """Simplified -> Traditional characters only (no Taiwan phrase rewrites), applied to references and hypotheses alike."""
    global _cc
    if _cc is None:
        import opencc

        _cc = opencc.OpenCC("s2t")
    return _cc.convert(s)


def norm(s: str) -> str:
    """Scoring normalization: s2t, digits to spoken zh (embed2asr numnorm), punctuation to spaces (so "Mac、iPad" stays two
    words), then embed2asr norm (NFKC, lowercase, 臺->台, no spaces next to CJK)."""
    import unicodedata

    from embed2asr.normalize import norm as e2a_norm
    from embed2asr.numnorm import numnorm

    t = numnorm(to_trad(s), "zh")[0]
    t = "".join(" " if unicodedata.category(c)[0] in "PS" and c != "'" else c for c in t)
    return e2a_norm(t)


def tokens(s: str) -> list[str]:
    """MER units: each CJK character and each Latin word (embed2asr.metrics.mer_tokens) of the normalized text."""
    from embed2asr.metrics import mer_tokens

    return mer_tokens(norm(s))


def is_latin(t: str) -> bool:
    return bool(_LATIN.match(t))


def lang_class(ref_tokens) -> str:
    lat = sum(is_latin(t) for t in ref_tokens)
    return "en" if lat == len(ref_tokens) else "zh" if lat == 0 else "mixed"


def align(ref, hyp):
    """Levenshtein alignment -> list of (op, ref_token|None, hyp_token|None), op in {=, S, D, I}."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def counts(ref, hyp) -> dict:
    """Per-category S/D/I and reference sizes; category = zh (CJK character) or en (Latin word) of the reference token
    (S, D) or of the hypothesis token (I)."""
    c = {f"{k}_{l}": 0 for k in ("S", "D", "I", "N") for l in ("zh", "en")}
    for op, r, h in align(ref, hyp):
        if r is not None:
            c[f"N_{'en' if is_latin(r) else 'zh'}"] += 1
        if op in "SD":
            c[f"{op}_{'en' if is_latin(r) else 'zh'}"] += 1
        elif op == "I":
            c[f"I_{'en' if is_latin(h) else 'zh'}"] += 1
    return c


def summarize(rows) -> dict:
    """rows: dicts from counts() -> totals and rates (MER over all units, CER over CJK, WER over Latin words)."""
    t = {}
    for r in rows:
        for k, v in r.items():
            t[k] = t.get(k, 0) + v
    out = dict(t)
    for l in ("zh", "en"):
        n = t.get(f"N_{l}", 0)
        e = sum(t.get(f"{k}_{l}", 0) for k in "SDI")
        out[f"err_{l}"] = e / n if n else None
    n = t.get("N_zh", 0) + t.get("N_en", 0)
    e = sum(t.get(f"{k}_{l}", 0) for k in "SDI" for l in ("zh", "en"))
    out["mer"] = e / n if n else None
    for k in "SDI":
        out[f"{k}_rate"] = sum(t.get(f"{k}_{l}", 0) for l in ("zh", "en")) / n if n else None
    return out


def edits(ref, hyp) -> int:
    from embed2asr.metrics import edit_distance

    return edit_distance(ref, hyp)


if __name__ == "__main__":  # self-check of the scoring helpers
    r, h = tokens("今天天氣很好 OK"), tokens("今天天汽很 okay 好")
    assert r == ["今", "天", "天", "氣", "很", "好", "ok"], r
    c = counts(r, h)
    assert c["N_zh"] == 6 and c["N_en"] == 1 and c["S_zh"] + c["D_zh"] + c["I_zh"] + c["S_en"] + c["D_en"] + c["I_en"] == edits(r, h), c
    assert lang_class(r) == "mixed" and lang_class(tokens("hello world")) == "en"
    assert norm("百分之3.7") == norm("百分之三點七"), (norm("百分之3.7"), norm("百分之三點七"))
    assert to_trad("软件") == "軟件"
    assert tokens("Mac、iPad、Android、PC 裝置") == ["mac", "ipad", "android", "pc", "裝", "置"]
    assert tokens("NVIDIA" + " " + "NVIDIA") == ["nvidia", "nvidia"]
    print("ok", c)
