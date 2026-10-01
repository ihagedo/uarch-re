#!/usr/bin/env python3
"""P4 non-classic mechanisms on the toy model; each writes results/toy/mech-<poc>/data.json.

p4_1  retire groups (48 rows x 7, one failable op per row) vs the classic toy  -> rule A4
p4_2  LSQ allocated at dispatch vs at issue, with and without early release    -> rule A5
p4_3  load address / value prediction on random, strided and constant chains
p4_4  two-level store queue (16 fast + 48 slow) vs one 64-entry SQ             -> rule A1

  taskset -c 4-19 python3 validation/toy/mech_p4.py [p4_1 p4_2 p4_3 p4_4] [--jobs 6]
"""

import argparse
from dataclasses import replace
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from tests.fixtures.anomaly_cases import offsets_table
from uarchre.analysis.anomaly import Curve, RunSet, evaluate, probe_params, transitions
from uarchre.analysis.protocol import run_protocol
from uarchre.backends.platform.toy import run_toy, ToyConfig
from uarchre.kernels import Kernel, Instr, Op
from uarchre.kernels.window import window_probe

RES = os.path.join(ROOT, "results", "toy")
BIG = dict(int_prf=32 + 480, fp_prf=32 + 480, lq=512, sq=512)
CTX = {"emitter": "abstract", "platform": "toy"}

P4_1 = {
    "classic": dict(BIG, rob=336, width=8, miss_latency=300),
    "retire_groups": dict(BIG, rob=48, retire_group=7, failable_per_group=1, width=8, miss_latency=300),
}
LSQ = dict(rob=128, int_prf=512, fp_prf=512, lq=32, sq=24, miss_latency_jitter=0.1)
P4_2 = {
    "dispatch": dict(LSQ),
    "issue": dict(LSQ, lsq_alloc="issue", ls_sched=16),
    "issue_early": dict(LSQ, lsq_alloc="issue", ls_sched=16, lq_release_early=True, sq_release_early=True),
}
P4_2_CURVES = [
    ("nop", 1, "miss"),
    ("load", 1, "miss"),
    ("store", 1, "miss"),
    ("load", 8, "fp"),
    ("store", 8, "fp"),
]
P4_4 = {"one_level": dict(BIG, rob=256, sq=64), "two_level": dict(BIG, rob=256, sq_levels=(16, 48))}


def pin():
    try:
        os.sched_setaffinity(0, set(range(4, 20)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass


def _cfg(d):
    d = dict(d)
    if "sq_levels" in d:
        d["sq_levels"] = tuple(d["sq_levels"])
    return ToyConfig(**d)


def sweep(job):
    """(tag, cfg dict, filler, depth, delay, coarse ns, outer reps) -> (tag, key, reps)."""
    tag, cfgd, f, d, dl, ns, outer = job
    cfg = _cfg(cfgd)
    mj, st = probe_params(d, dl)

    def measure(key, n, rep):
        return run_toy(window_probe(f, n, depth=d, delay=dl), replace(cfg, seed=cfg.seed + rep))

    _, reps = run_protocol(measure, ["k"], ns, outer_reps=outer, seed=0, min_jump=mj, expected_steps=st)
    return tag, (f, d, dl), {int(n): v for n, v in reps["k"].items()}


def run_sweeps(jobs, workers):
    out = {}
    with Pool(workers) as pool:
        for tag, key, reps in pool.imap_unordered(sweep, jobs):
            out.setdefault(tag, {})[key] = reps
    return out


def runset(tag, reps_by_key, cfgd, n_boot=1000):
    rs = RunSet("toy", tag, ctx=dict(CTX), widths={"default": cfgd.get("width", 4)}, offsets=offsets_table())
    for (f, d, dl), reps in sorted(reps_by_key.items()):
        rs.add(Curve.from_reps(f, reps, depth=d, delay=dl, n_boot=n_boot))
    return rs


def summarize(rs, cfgd):
    fs = evaluate(rs)
    curves = {}
    for k, c in rs.curves.items():
        x, y = c.median_curve()
        est = c.estimate()
        curves["/".join(map(str, k))] = {
            "filler": k[0],
            "depth": k[1],
            "delay": k[2],
            "n": x.tolist(),
            "y": y.tolist(),
            "reps": {str(n): v for n, v in c.reps.items()},
            "knee": c.knee.knee,
            "ci": c.knee.ci,
            "ci_method": c.knee.ci_method,
            "offset": c.offset and c.offset["value"],
            "estimate": est and est[0],
            "estimate_ci": est and est[1],
        }
    return {
        "cfg": cfgd,
        "truth": _cfg(cfgd).headroom(),
        "curves": curves,
        "findings": [f.to_dict() for f in fs],
        "fired": sorted({f.rule for f in fs}),
    }


def write(poc, data):
    d = os.path.join(RES, f"mech-{poc.replace('_', '')}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "data.json"), "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {os.path.relpath(d, ROOT)}/data.json")


def rng(a, b, s):
    return list(range(a, b + 1, s))


def p4_1(workers):
    jobs = [
        (t, cfg, f, 1, "miss", rng(8, 420, 8), 1)
        for t, cfg in P4_1.items()
        for f in ("nop", "load", "store", "branch")
    ]
    sw = run_sweeps(jobs, workers)
    data = {t: summarize(runset(t, sw[t], P4_1[t]), P4_1[t]) for t in P4_1}
    for t, v in data.items():
        k = {c["filler"]: c["knee"] for c in v["curves"].values()}
        ratio = k["nop"] / min(k["load"], k["store"], k["branch"])
        print(f"P4.1 {t:14s} knees {k}  NOP/min(failable) = {ratio:.2f}  fired {v['fired']}")
    write("p4_1", data)


def p4_2(workers):
    jobs = [(t, cfg, f, d, dl, rng(8, 176, 8), 5) for t, cfg in P4_2.items() for f, d, dl in P4_2_CURVES]
    sw = run_sweeps(jobs, workers)
    data = {t: summarize(runset(t, sw[t], P4_2[t]), P4_2[t]) for t in P4_2}
    for t, v in data.items():
        e = {k: (c["estimate"], c["estimate_ci"]) for k, c in v["curves"].items()}
        print(f"P4.2 {t:12s} {e}\n      fired {v['fired']}")
    write("p4_2", data)


def chase(pattern, d=8):
    return Kernel(
        f"chase_{pattern}_{d}",
        [Instr(Op.MISS_LOAD, 0) for _ in range(d)],
        n_chains=1,
        params={"chain": pattern},
    )


PRED = {"none": {}, "address": {"load_addr_predict": True}, "value": {"load_value_predict": True}}


def _chase_job(job):
    pred, pattern = job
    return pred, pattern, run_toy(chase(pattern), ToyConfig(**PRED[pred]), iters=30, warm=10) / 8


def _probe_job(job):
    pattern, n = job
    k = window_probe("nop", n)
    k.params["chain"] = pattern
    return pattern, n, run_toy(k, ToyConfig(rob=128, load_addr_predict=True))


def p4_3(workers):
    with Pool(workers) as pool:
        ch = pool.map(_chase_job, [(p, c) for p in PRED for c in ("random", "stride", "constant")])
        pr = pool.map(_probe_job, [(c, n) for c in ("random", "stride") for n in rng(8, 176, 4)])
    data = {
        "chase": {p: {c: v for pp, c, v in ch if pp == p} for p in PRED},
        "depth": 8,
        "miss_latency": ToyConfig().miss_latency,
        "probe": {
            c: {"n": [n for cc, n, _ in pr if cc == c], "y": [y for cc, _, y in pr if cc == c]}
            for c in ("random", "stride")
        },
    }
    det = {}
    for p, row in data["chase"].items():
        det[p] = {c: round(row["random"] / v, 1) for c, v in row.items()}
    data["speedup_vs_random"] = det
    print("P4.3 cycles per load", data["chase"], "\n     speedup vs random", det)
    write("p4_3", data)


def p4_4(workers):
    jobs = [(t, cfg, f, 1, "miss", rng(2, 96, 2), 1) for t, cfg in P4_4.items() for f in ("nop", "store")]
    jobs = [j if j[2] == "store" else j[:5] + (rng(128, 320, 8), 1) for j in jobs]
    sw = run_sweeps(jobs, workers)
    data = {t: summarize(runset(t, sw[t], P4_4[t]), P4_4[t]) for t in P4_4}
    for t, v in data.items():
        c = v["curves"]["store/1/miss"]
        tr = transitions(c["n"], c["y"])
        v["store_transitions"] = tr
        print(f"P4.4 {t:10s} store knee {c['knee']} transitions {tr} fired {v['fired']}")
    write("p4_4", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["p4_1", "p4_2", "p4_3", "p4_4"])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    pin()
    for w in args.which:
        globals()[w](args.jobs)


if __name__ == "__main__":
    main()
