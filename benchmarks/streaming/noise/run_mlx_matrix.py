"""Real MLX Large replay over synthetic speech+noise: energy vs adaptive VAD, suppression off/on.
Usage: python3 run_mlx_matrix.py OUTDIR SUPPRESS(on|off) [extra args]   (repo root, MLXLIB/H/A env set by the caller script)
Each condition writes OUTDIR/<name>.json (full tingmix-bench report); a summary table is printed."""
import json, os, subprocess, sys, itertools

out, suppress, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
H, A = os.environ["H"], os.environ["A"]
ENV = {**os.environ, "DYLD_LIBRARY_PATH": os.environ["MLXLIB"]}
BIN = os.environ.get("BENCH", "target/release/tingmix-bench")
base = [BIN, "--backend", "mlx", "--artifact", A, "--units", H + "units.json", "--lm", H + "lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz",
        "--dtype", "f32", "--speed", "2", "--seed", "5", "--suppress", suppress, *extra]
conds = [("clean", ["--noise", "none"], True)]
conds += [(f"{n}20", ["--noise", n, "--snr-db", "20"], True) for n in ("white", "pink", "fan", "babble")]
conds += [(f"white10", ["--noise", "white", "--snr-db", "10"], True), ("music-35dB", ["--noise", "music", "--noise-dbfs", "-35"], True)]
conds += [(f"only-{n}20", ["--noise", n, "--snr-db", "20", "--no-speech", "--gap-min", "10", "--gap-max", "20"], False) for n in ("white", "fan", "babble")]
conds += [("only-music-35dB", ["--noise", "music", "--noise-dbfs", "-35", "--no-speech", "--gap-min", "10", "--gap-max", "20"], False)]
only = os.environ.get("ONLY")  # comma-separated condition names to (re)run
for (name, args, speech), vad in itertools.product(conds, ("energy", "adaptive")):
    if only and name not in only.split(","):
        continue
    path = f"{out}/{name}-{vad}.json"
    dur = "120" if speech else "90"
    subprocess.run([*base, "--vad", vad, "--duration-s", dur, "--scenario", f"{name}-{vad}-suppress-{suppress}", "--out", path, *args], check=True, capture_output=True, env=ENV)
    r = json.load(open(path))
    print(f"{name:12s} {vad:8s} falseSeg/min {r['false_segments_per_min']:.2f} missed {r['missed_utterances']} streamCER {r['stream_cer'] if r['stream_cer'] is None else round(r['stream_cer'], 3)} "
          f"fp95 {r['first_partial_from_onset_s']['p95']:.2f} fin95 {r['final_from_speech_end_s']['p95']:.2f} errors {len(r['errors'])}", flush=True)
