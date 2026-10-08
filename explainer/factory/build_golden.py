#!/usr/bin/env python3
"""Write the factory's golden suite from the Python toy model, and the JSON specs the browser reads.

    python3 explainer/factory/build_golden.py           # rewrite stale files
    python3 explainer/factory/build_golden.py --check   # exit 1 if anything is stale

specs/*.json mirror spec/cores/*.yaml and offsets.yaml (the engine has no YAML parser). golden/
holds short event traces (each checked against run_toy), the R10K sweep, knee curves and resolver cases.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.kernels.window import window_probe
from uarchre.backends.platform.toy import ToyConfig, run_toy
from uarchre.backends.platform.toy_trace import trace_toy, resolve_spec, CLASSIC_FIELDS, UnsupportedSpec
from uarchre.analysis.knee import find_knee, attribute

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(ROOT, "spec", "cores")
OFFSETS = os.path.join(ROOT, "uarchre", "analysis", "offsets.yaml")

# the R10K spec has no DRAM latency, which resolve_spec refuses; 200 is the toy default
R10K_OVERRIDES = {"miss_latency": 200}
SMALL = dict(
    rob=48,
    int_prf=72,
    fp_prf=70,
    lq=12,
    sq=10,
    width=3,
    miss_latency=60,
    alu_latency=2,
    fp_latency=5,
    load_latency=3,
    alu_delay_latency=17,
)

# slow FP fillers: an older chain link and a younger, earlier-issued filler complete in one cycle
SKEW = dict(SMALL, width=2, fp_latency=9, alu_delay_latency=6)

# (case id, config name, filler, n, depth, delay); every Op and every stall reason appears
CASES = [
    ("r10k-nop-n30-d1", "r10k", "nop", 30, 1, "miss"),  # last N that fits the 32-entry active list
    ("r10k-nop-n31-d1", "r10k", "nop", 31, 1, "miss"),  # first N past it: the misses serialize
    ("r10k-int-n29-d1", "r10k", "int", 29, 1, "miss"),  # integer headroom 31
    ("r10k-int-n30-d1", "r10k", "int", 30, 1, "miss"),
    ("r10k-fp-n30-d1", "r10k", "fp", 30, 1, "miss"),  # FP headroom 32 hides behind the active list
    ("r10k-load-n14-d1", "r10k", "load", 14, 1, "miss"),
    ("r10k-store-n16-d1", "r10k", "store", 16, 1, "miss"),
    ("r10k-mix-n30-d1", "r10k", "mix", 30, 1, "miss"),
    ("small-nop-n22-d2", "small", "nop", 22, 2, "miss"),
    ("small-int-n20-d2-fp", "small", "int", 20, 2, "fp"),  # FSQRT-like delay chain
    ("small-fp-n16-d3-int", "small", "fp", 16, 3, "int"),  # integer delay chain
    ("small-branch-n24-d1", "small", "branch", 24, 1, "miss"),
    ("small-flags-n12-d1", "small", "flags", 12, 1, "miss"),
    ("small-load-n12-d1", "small", "load", 12, 1, "miss"),
    ("small-store-n10-d2", "small", "store", 10, 2, "miss"),
    ("skew-fp-n6-d2-int", "skew", "fp", 6, 2, "int"),
]
GOLDEN_ITERS, GOLDEN_WARM = 3, 1
SWEEP_NS = list(range(2, 61))
SWEEP_FILLERS = ["nop", "int", "fp"]
JOBS = 6  # at most 6 workers, off CPUs 0-3 (kept quiet for native timing)


def pool():
    """A Pool of JOBS workers pinned off CPUs 0-3 (children inherit the affinity)."""
    try:
        cpus = {c for c in os.sched_getaffinity(0) if c >= 4}
        if cpus:
            os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        pass
    return Pool(JOBS)


def load_spec(core):
    with open(os.path.join(SPEC_DIR, f"{core}.yaml")) as fh:
        return yaml.safe_load(fh)


def configs():
    default = ToyConfig()
    return {
        "r10k": resolve_spec(load_spec("r10000"), R10K_OVERRIDES)["cfg"],
        "small": {**{k: getattr(default, k) for k in CLASSIC_FIELDS}, **SMALL},
        "skew": {**{k: getattr(default, k) for k in CLASSIC_FIELDS}, **SKEW},
    }


def dumps(o):
    return json.dumps(o, separators=(",", ":"))


def golden_case(case):
    cid, cname, filler, n, depth, delay = case
    cfg = configs()[cname]
    k = window_probe(filler, n, depth=depth, delay=delay)
    tr = trace_toy(k, ToyConfig(**cfg), iters=GOLDEN_ITERS, warm=GOLDEN_WARM)
    ref = run_toy(k, ToyConfig(**cfg), iters=GOLDEN_ITERS, warm=GOLDEN_WARM)
    if tr.cycles_per_iter != ref:
        raise SystemExit(
            f"{cid}: trace_toy {tr.cycles_per_iter} != run_toy {ref}: toy_trace.py is out of date"
        )
    head = {
        "golden": 1,
        "case": cid,
        "config": cname,
        "kernel": {"filler": filler, "n": n, "depth": depth, "delay": delay, "rotation": 8},
        "body": [[i.op.value, i.arg] for i in k.body],
        "cfg": cfg,
        "iters": GOLDEN_ITERS,
        "warm": GOLDEN_WARM,
        "cycles_per_iter": tr.cycles_per_iter,
        "iter_marks": tr.iter_marks,
        "final_cycle": tr.final_cycle,
        "dispatched": tr.dispatched,
        "n_events": len(tr.events),
    }
    return cid, dumps(head) + "\n" + "".join(dumps(e) + "\n" for e in tr.events), len(tr.events)


def _sweep_point(args):
    filler, n, depth, cfg = args
    return run_toy(window_probe(filler, n, depth=depth), ToyConfig(**cfg))


def knee_dict(k):
    return {
        "knee": k.knee,
        "ci": k.ci,
        "jump_ratio": float(k.jump_ratio),
        "quality": float(k.quality),
        "anomaly": k.anomaly,
        "flags": list(k.flags),
    }


def r10k_sweep(pool):
    cfg = configs()["r10k"]
    out = {
        "spec": "spec/cores/r10000.yaml",
        "overrides": R10K_OVERRIDES,
        "cfg": cfg,
        "iters": 60,
        "warm": 20,
        "min_jump": 1.4,
        "expected_steps": 1,
        "sweeps": {},
    }
    for depth in (1, 2):
        fillers = SWEEP_FILLERS if depth == 1 else ["nop"]
        mj, steps = 1 + 0.4 / depth, 1 if depth == 1 else 2
        knees = {}
        for f in fillers:
            ys = pool.map(_sweep_point, [(f, n, depth, cfg) for n in SWEEP_NS])
            kn = find_knee(SWEEP_NS, ys, mj, steps)
            knees[f] = kn
            out["sweeps"][f"{f}-d{depth}"] = {
                "filler": f,
                "depth": depth,
                "min_jump": mj,
                "expected_steps": steps,
                "ns": SWEEP_NS,
                "ys": ys,
                "knee": knee_dict(kn),
            }
        est = attribute(knees, {"depth": depth})
        out[f"attribute-d{depth}"] = {
            s: {
                "filler": e["filler"],
                "knee": e["knee"],
                "value": e["value"],
                "status": e["status"],
                "anomalies": e["anomalies"],
                "offset": e["offset"]["value"],
            }
            for s, e in est.items()
        }
    return out


def knee_cases(sweep):
    """Curves for the JS find_knee port: the R10K sweeps plus synthetic shapes."""
    cases = []
    for name, s in sweep["sweeps"].items():
        cases.append(
            {
                "name": f"r10k-{name}",
                "ns": s["ns"],
                "ys": s["ys"],
                "min_jump": s["min_jump"],
                "expected_steps": s["expected_steps"],
            }
        )
    rng = np.random.default_rng(7)
    ns = list(range(10, 90, 2))
    step = [100.0 if n <= 50 else 200.0 for n in ns]
    cases.append(
        {
            "name": "synth-step-noisy",
            "ns": ns,
            "ys": [float(v * (1 + 0.02 * rng.standard_normal())) for v in step],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    cases.append(
        {
            "name": "synth-sigmoid",
            "ns": ns,
            "ys": [float(100 + 100 / (1 + np.exp(-(n - 50.5) / 2.0))) for n in ns],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    cases.append(
        {
            "name": "synth-staircase",
            "ns": ns,
            "ys": [100.0 if n <= 40 else (150.0 if n <= 46 else 200.0) for n in ns],
            "min_jump": 1.2,
            "expected_steps": 2,
        }
    )
    cases.append(
        {
            "name": "synth-extra-step",
            "ns": ns,
            "ys": [100.0 if n <= 30 else (200.0 if n <= 60 else 400.0) for n in ns],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    cases.append(
        {
            "name": "synth-flat",
            "ns": ns,
            "ys": [float(100 + 0.1 * (n % 3)) for n in ns],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    cases.append(
        {
            "name": "synth-ramp-poor-fit",
            "ns": ns,
            "ys": [float(100 + 3 * n + 40 * (n % 4 == 0)) for n in ns],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    cases.append(
        {"name": "synth-unsorted", "ns": ns[::-1], "ys": step[::-1], "min_jump": 1.3, "expected_steps": 1}
    )
    cases.append(
        {
            "name": "synth-too-few",
            "ns": [1, 2, 3],
            "ys": [1.0, 1.0, 2.0],
            "min_jump": 1.3,
            "expected_steps": 1,
        }
    )
    for c in cases:
        c["python"] = knee_dict(find_knee(c["ns"], c["ys"], c["min_jump"], c["expected_steps"]))
    return cases


RESOLVE_CASES = [
    ("r10000", R10K_OVERRIDES),
    ("r10000", {}),
    ("r10000", {"miss_latency": 300, "rob": 64}),
    ("firestorm", {"miss_latency": 300}),
]


def resolve_cases():
    out = []
    for core, ov in RESOLVE_CASES:
        c = {"spec": core, "overrides": ov}
        try:
            c["result"] = resolve_spec(load_spec(core), ov)
        except UnsupportedSpec as e:
            c["error"] = str(e)
        out.append(c)
    return out


def build():
    """{relative path: file content} for everything this script owns."""
    files = {}
    for fn in sorted(os.listdir(SPEC_DIR)):
        if fn.endswith(".yaml"):
            files[f"specs/{fn[:-5]}.json"] = json.dumps(load_spec(fn[:-5]), indent=1) + "\n"
    with open(OFFSETS) as fh:
        files["specs/offsets.json"] = json.dumps(yaml.safe_load(fh), indent=1) + "\n"
    with pool() as p:
        index = {
            "golden": 1,
            "generated_by": "explainer/factory/build_golden.py",
            "reference": "uarchre/backends/platform/toy_trace.py (= toy.py run_toy)",
            "cases": [],
        }
        for cid, text, nev in p.map(golden_case, CASES):
            files[f"golden/{cid}.jsonl"] = text
            index["cases"].append({"case": cid, "file": f"{cid}.jsonl", "events": nev})
        sweep = r10k_sweep(p)
    files["golden/r10k_sweep.json"] = json.dumps(sweep, indent=1) + "\n"
    files["golden/knee_cases.json"] = json.dumps(knee_cases(sweep), indent=1) + "\n"
    files["golden/resolve_cases.json"] = json.dumps(resolve_cases(), indent=1) + "\n"
    index["events_total"] = sum(c["events"] for c in index["cases"])
    index["sweep"], index["knee"], index["resolve"] = (
        "r10k_sweep.json",
        "knee_cases.json",
        "resolve_cases.json",
    )
    files["golden/index.json"] = json.dumps(index, indent=1) + "\n"
    return files


def stale(files):
    out = []
    for rel, text in files.items():
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            out.append(rel)
            continue
        with open(p) as fh:
            if fh.read() != text:
                out.append(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report stale files")
    args = ap.parse_args()
    files = build()
    bad = stale(files)
    if args.check:
        print("stale: " + ", ".join(bad) if bad else "golden suite up to date")
        sys.exit(1 if bad else 0)
    for rel in bad:
        p = os.path.join(HERE, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(files[rel])
        print(f"wrote {rel}")
    total = sum(len(t) for r, t in files.items() if r.startswith("golden/"))
    print(
        f"{len(CASES)} cases, {json.loads(files['golden/index.json'])['events_total']} events, "
        f"golden/ {total / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
