"""Generic stage driver: convert (trace/export x fp32/fp16), run on every compute unit, parity, placement -> JSON."""
import sys, numpy as np, torch, coremltools as ct
from harness import convert, run_units, placement, record

def drive(stage, mod, name, feed: dict, ref: np.ndarray, tag="", methods=("trace", "export"), precs=("fp32", "fp16"), extra=None, n=10):
    res = {}
    with torch.no_grad():
        t = mod(*[torch.from_numpy(v) for v in feed.values()])
    res["torch_wrapper_vs_oracle"] = dict(max_abs=float(np.abs(t.numpy() - ref).max()))
    print(stage, "wrapper vs oracle max_abs", res["torch_wrapper_vs_oracle"])
    for method in methods:
        for prec in precs:
            key = f"{method}_{prec}"
            c = convert(mod, tuple(torch.from_numpy(v) for v in feed.values()), f"{name}{tag}", method=method, precision=prec,
                        inputs=[ct.TensorType(name=k, shape=v.shape) for k, v in feed.items()])
            res[key] = dict(convert=c)
            print(key, {k: v for k, v in c.items() if k != "error"}, ("\n" + c["error"][-700:]) if not c["ok"] else "")
            if not c["ok"]:
                continue
            out_name = list(ct.models.MLModel(c["path"], compute_units=ct.ComputeUnit.CPU_ONLY).predict(feed).keys())[0]
            res[key]["run"] = run_units(c["path"], feed, {out_name: ref}, n=n, extra=extra)
            res[key]["placement"] = placement(c["path"])
            for u, r in res[key]["run"].items():
                print("  ", u, r.get("median_ms"), {k: round(v["rel_l2"], 6) for k, v in r.get("parity", {}).items()}, (r.get("error") or "")[-300:])
            print("   placement ALL:", {k: v for k, v in res[key]["placement"].get("ALL", {}).items() if k != "not_ne_supported_by_type"})
    return res
