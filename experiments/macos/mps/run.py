"""PyTorch MPS experiment for TingMix-Large (goals.md section 5, Experiment A). Research only.

Usage: uv run python run.py [--out ../../../benchmarks/apple/mps/results.json] [--runs 15]

MPS CPU fallback is disabled; every op failure is recorded verbatim. Stages are teacher-forced from the CPU fp32
oracle (experiments/macos/reference/out) so one stage's error does not hide behind another's. Feature extraction
(HF AutoFeatureExtractor) is numpy on the CPU in every runtime and is timed, not run on MPS."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # before torch is imported

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import common as C  # noqa: E402
from embed2asr.ctc import collapse_ctc  # noqa: E402
from tingmix.models.large.encoder import _mask_and_pos  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# Proposed gates: relative RMS error vs CPU fp32 (rms(x-ref)/rms(ref)) per stage output.
TOL = {"fp32": 1e-3, "fp16": 2e-2, "bf16": 5e-2}
ORACLE = C.REPO / "experiments/macos/reference/out"
FAILS = []
FE = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def err(x, ref):
    x, ref = x.detach().float().cpu(), ref.detach().float().cpu()
    d = x - ref
    return {"max_abs": d.abs().max().item(), "mean_abs": d.abs().mean().item(),
            "rel_rms": (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item(),
            "nan_or_inf": bool((~torch.isfinite(x)).any())}


def attempt(name, fn, ctx):
    """Run fn; on failure record the exception verbatim (fallback is off, so unsupported ops raise)."""
    try:
        r = fn()
        torch.mps.synchronize()
        return r
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        FAILS.append({**ctx, "stage": name, "error": msg, "traceback_tail": traceback.format_exc().splitlines()[-6:]})
        print("FAIL", ctx, name, msg[:200], flush=True)
        return None


def cpu_stages(tower, reader, m, mm):
    """CPU fp32 intermediates for one chunk."""
    with torch.inference_mode():
        h, om = tower.subsample_conv_projection(m, mm)
        lens = om.sum(1)
        am, pe = _mask_and_pos(tower, h, lens)
        h1 = tower.layers[0](h, attention_mask=am, position_embeddings=pe)
        x = h
        for layer in tower.layers:
            x = layer(x, attention_mask=am, position_embeddings=pe)
        feats = tower.output_proj(x)
        logits = reader(feats, lens)
    return dict(h=h, om=om, lens=lens, am=am, pe=pe, h1=h1, feats=feats, logits=logits)


def stage_tests(tower_cpu, reader_cpu, wav, dname, dt, dev, clip):
    """Each MPS stage fed the CPU fp32 input cast to dt; error vs the CPU fp32 output of the same stage."""
    m, mm = C.mel(FE, [wav])
    ref = cpu_stages(tower_cpu, reader_cpu, m, mm)
    tower = copy.deepcopy(tower_cpu).to(dev, dt)
    ctx = {"clip": clip, "dtype": dname}
    out = {}
    with torch.inference_mode():
        r = attempt("subsample_conv", lambda: tower.subsample_conv_projection(m.to(dev, dt), mm.to(dev)), ctx)
        if r:
            out["subsample_conv"] = {**err(r[0], ref["h"]), "mask_equal": bool((r[1].cpu() == ref["om"]).all())}
        h = ref["h"].to(dev, dt)
        lens = ref["lens"].to(dev)
        r = attempt("mask5d_and_relpos", lambda: _mask_and_pos(tower, h, lens), ctx)
        am = pe = None
        if r:
            am, pe = r
            out["mask5d"] = {"shape": list(am.shape), "equal_to_cpu": bool((am.cpu() == ref["am"]).all())}
            out["rel_pos_enc"] = err(pe, ref["pe"])
        if am is not None:
            r = attempt("layer0_relpos_attention", lambda: tower.layers[0](h, attention_mask=am, position_embeddings=pe), ctx)
            if r is not None:
                out["layer0"] = err(r, ref["h1"])

            def all_layers():
                x = h
                for layer in tower.layers:
                    x = layer(x, attention_mask=am, position_embeddings=pe)
                return tower.output_proj(x)

            r = attempt("layers12_output_proj", all_layers, ctx)
            if r is not None:
                out["layers12_output_proj"] = err(r, ref["feats"])
        f_in = ref["feats"].to(dev)  # reader input is fp32 in production
        for rname, rdt in [("reader_fp32", torch.float32)] + ([] if dname == "fp32" else [(f"reader_{dname}", dt)]):
            rd = copy.deepcopy(reader_cpu).to(dev, rdt)
            r = attempt(rname + "_bigru_ctc_logits", lambda: rd(f_in.to(rdt), ref["lens"]), {**ctx, "reader": rname})
            if r is not None:
                out[rname] = err(r, ref["logits"])
    return out


def e2e(tower, reader, tok, lm, dev, dname, dt):
    """Full pipeline on MPS (tower in dt, reader fp32) vs the oracle dump: features, logits, greedy and beam text."""
    texts = json.loads((ORACLE / "results.json").read_text())["texts"]
    res = {}
    for clip in C.manifest():
        ctx = {"clip": clip["id"], "dtype": dname}
        o = np.load(ORACLE / f"{clip['id']}.npz")
        g, b, ef, el = [], [], [], []
        ok = True
        for k, w in enumerate(C.chunks(C.load_wav(clip["path"]))):
            m, mm = C.mel(FE, [w])

            def run():
                with torch.inference_mode():
                    r = tower(m.to(dev, dt), mm.to(dev), return_dict=True)
                    f, ln = r.last_hidden_state, r.attention_mask.sum(1)
                    return f, ln, reader(f.float(), ln.cpu())

            r = attempt("e2e_forward", run, ctx)
            if r is None:
                ok = False
                break
            f, ln, lg = r
            n = int(ln[0])
            ef.append(err(f[0, :n], torch.from_numpy(o[f"c{k}_features"])))
            el.append(err(lg[0, :n], torch.from_numpy(o[f"c{k}_logits"])))
            g.append(tok.decode(collapse_ctc(lg[0, :n].argmax(-1).tolist(), 0)))
            b.append(lm(lg.cpu(), ln.cpu(), tok)[0])
        if not ok:
            res[clip["id"]] = {"status": "FAILED"}
            continue
        ref = texts[clip["id"]]
        res[clip["id"]] = {"features": max(ef, key=lambda e: e["rel_rms"]), "logits": max(el, key=lambda e: e["rel_rms"]),
                           "greedy": C.join(g), "greedy_equal_cpu": C.join(g) == ref["greedy"],
                           "beam_lm": C.join(b), "beam_lm_equal_cpu": C.join(b) == ref["beam_lm"]}
    return res


def make_audio(seconds):
    """Deterministic: concatenate all fixture clips in manifest order, tile if needed, crop."""
    w = np.concatenate([C.load_wav(c["path"]) for c in C.manifest()])
    n = int(seconds * 16000)
    return np.tile(w, -(-n // len(w)))[:n]


def pct(a, q):
    return float(np.percentile(a, q))


def bench(tower, reader, tok, lm, dev, dname, dt, secs, runs, warm=3):
    wav = make_audio(secs)
    ctx = {"dtype": dname, "audio_s": secs}
    peak = {"cur": 0, "stop": False}

    def poll():
        while not peak["stop"]:
            peak["cur"] = max(peak["cur"], torch.mps.current_allocated_memory())
            time.sleep(0.002)

    def one():
        t0 = time.perf_counter()
        m, mm = C.mel(FE, [wav])
        t1 = time.perf_counter()
        with torch.inference_mode():
            r = tower(m.to(dev, dt), mm.to(dev), return_dict=True)
            f, ln = r.last_hidden_state, r.attention_mask.sum(1)
            lg = reader(f.float(), ln.cpu())
        torch.mps.synchronize()
        t2 = time.perf_counter()
        lgc = lg.cpu()
        collapse_ctc(lgc[0, : int(ln[0])].argmax(-1).tolist(), 0)
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, lgc, ln.cpu()

    cold = attempt("bench_first_call", one, ctx)
    if cold is None:
        return {"status": "FAILED"}
    first = sum(cold[:3])
    for _ in range(warm - 1):
        one()
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    rows = [one() for _ in range(runs)]
    peak["stop"] = True
    th.join()
    t_fe, t_gpu, t_gr = (np.array([r[i] for r in rows]) for i in range(3))
    tot = t_fe + t_gpu + t_gr
    t0 = time.perf_counter()
    lm(rows[-1][3], rows[-1][4], tok)
    beam = time.perf_counter() - t0
    return {"first_call_s": first, "warmup_runs": warm, "runs": runs,
            "latency_s_fe_cpu": {"p50": pct(t_fe, 50)},
            "latency_s_tower_reader_mps": {"p50": pct(t_gpu, 50), "p95": pct(t_gpu, 95), "max": float(t_gpu.max())},
            "latency_s_total_greedy": {"p50": pct(tot, 50), "p95": pct(tot, 95), "p99": pct(tot, 99), "max": float(tot.max())},
            "rtf_p50": pct(tot, 50) / secs, "rtf_p95": pct(tot, 95) / secs, "beam_lm_decode_s_once": beam,
            "mps_current_alloc_peak_MB": peak["cur"] / 2**20, "mps_driver_alloc_MB": torch.mps.driver_allocated_memory() / 2**20,
            "rss_MB": psutil.Process().memory_info().rss / 2**20}


def hardware():
    sp = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    gpu = [l.strip() for l in sp.splitlines() if "Chipset Model" in l or "Total Number of Cores" in l]
    return {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "gpu": gpu, "ram_GB": psutil.virtual_memory().total / 2**30, "os": platform.platform(), "macos": platform.mac_ver()[0]}


def main():
    global FE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.REPO / "benchmarks/apple/mps/results.json"))
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--stage-clips", default="zh_08s,mix_08s")
    a = ap.parse_args()
    dev = torch.device("mps")
    tower0, FE, reader0, tok, lm, path = C.load_model("cpu", torch.float32)
    wavs = {c["id"]: C.load_wav(c["path"])[: C.CHUNK] for c in C.manifest()}
    load_start = os.getloadavg()
    stages, e2e_res, bench_res = {}, {}, {}
    for dname, dt in DTYPES.items():
        tower = copy.deepcopy(tower0)
        if attempt("move_tower_to_mps", lambda: tower.to(dev, dt), {"dtype": dname}) is None:
            continue
        reader = copy.deepcopy(reader0).to(dev)
        print("==", dname, flush=True)
        e2e_res[dname] = e2e(tower, reader, tok, lm, dev, dname, dt)
        stages[dname] = {c: stage_tests(tower0, reader0, wavs[c], dname, dt, dev, c) for c in a.stage_clips.split(",")}
        bench_res[dname] = {}
        for secs in (2, 4, 8, 16, 29):
            bench_res[dname][str(secs)] = bench(tower, reader, tok, lm, dev, dname, dt, secs, a.runs)
            print(dname, secs, bench_res[dname][str(secs)].get("rtf_p50"), flush=True)
        del tower, reader
    res = {"command": f"cd experiments/macos/mps && uv run python run.py --runs {a.runs}", "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
           "hardware": hardware(), "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
           "artifacts_sha256": {f: sha(path / f) for f in ("model.safetensors", "encoder_delta.safetensors", "config.json", "tingmix.json", "units.json", C.LM_FILE)},
           "revisions": {"model": f"{C.LARGE_ID}@{C.LARGE_REV}", "encoder": "google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b"},
           "oracle_manifest_sha256": sha(ORACLE / "MANIFEST.sha256"), "tolerance_rel_rms_proposed": TOL,
           "caveat": "Other processes ran GPU/CPU work concurrently; timings are indicative. loadavg recorded.",
           "loadavg_start": load_start, "loadavg_end": os.getloadavg(), "failures": FAILS, "stages": stages, "e2e": e2e_res, "bench": bench_res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", a.out, "failures:", len(FAILS))


main()
