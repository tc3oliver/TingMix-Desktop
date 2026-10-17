"""Size/shape statistics of the TingMix-Large character 5-gram LM (gzip JSON count tables). Writes out/lm_stats.json.

  uv run python lm_stats.py /path/to/lmCN_cv_parl_ascend_ntuml.charlm.json.gz
"""
import gzip
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

p = Path(sys.argv[1])
t0 = time.time()
raw = gzip.decompress(p.read_bytes())
d = json.loads(raw)
cnt = d["counts"]
st = {
    "file": p.name,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "gz_bytes": p.stat().st_size,
    "json_bytes": len(raw),
    "format": d["format"],
    "order": d["order"],
    "D": d["D"],
    "special_codepoints": [ord(d[k]) for k in ("BOS", "EOS", "PH")],
    "entries_per_order": [len(c) for c in cnt],
    "total_entries": sum(len(c) for c in cnt),
    "unigram_vocab": len(cnt[0]),
    "count_max_per_order": [max(c.values()) for c in cnt],
    "count_values_all_int": all(isinstance(v, int) for c in cnt for v in c.values()),
    "load_seconds": round(time.time() - t0, 1),
    "python_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0),
}
Path(__file__).parent.joinpath("out/lm_stats.json").write_text(json.dumps(st, indent=1))
print(json.dumps(st, indent=1))
