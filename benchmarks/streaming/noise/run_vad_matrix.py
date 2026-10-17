"""Model-free VAD matrix: Pipeline segmentation vs known speech spans, energy vs adaptive VAD, over synthetic noise.
Usage: python3 run_vad_matrix.py OUT.json [extra tingmix-bench args]  (run from the repo root, release build present)"""
import json, subprocess, sys, tempfile, os, itertools

BIN = os.environ.get("BENCH", "target/release/tingmix-bench")
# macOS strips DYLD_* when /usr/bin/python3 starts, so the MLX build gets its library path from MLXLIB.
ENV = {**os.environ, **({"DYLD_LIBRARY_PATH": os.environ["MLXLIB"]} if "MLXLIB" in os.environ else {})}
out = sys.argv[1]
extra = sys.argv[2:]
rows = []
conds = [("none", None)] + [(n, s) for n in ("white", "pink", "fan", "babble") for s in (40, 30, 20, 15, 10, 5)]
for (noise, snr), vad, speech in itertools.product(conds, ("energy", "adaptive"), (True, False)):
    if noise == "none" and not speech:
        continue
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cmd = [BIN, "--vad-only", "--vad", vad, "--duration-s", "600", "--seed", "11", "--out", path, *extra]
    cmd += ["--noise", noise] + (["--snr-db", str(snr)] if snr is not None else [])
    if not speech:
        cmd += ["--no-speech", "--gap-min", "5", "--gap-max", "20"]
    subprocess.run(cmd, check=True, capture_output=True, env=ENV)
    r = json.load(open(path))
    os.unlink(path)
    rows.append({"noise": noise, "snr_db": snr, "vad": vad, "speech": speech, **r["result"]})
    print(noise, snr, vad, speech, {k: round(v, 3) if isinstance(v, float) else v for k, v in r["result"].items() if k in ("false_segments_per_min", "missed_utterances", "speech_coverage", "merged_segments", "longest_segment_s")}, flush=True)
json.dump({"binary": BIN, "extra_args": extra, "env": r["env"], "rows": rows}, open(out, "w"), indent=1)
