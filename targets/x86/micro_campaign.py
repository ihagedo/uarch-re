#!/usr/bin/env python3
"""Latency / throughput micro campaign on one pinned CPU (P1.6, P1.8).

Runs the uarchre/kernels/x86_micro.py tests with every repetition bracketed by the clock
probe, so results are in core cycles even while the clock moves. Writes
results/native/<tag>/micro_raw.csv and micro.json.

  targets/x86/micro_campaign.py --tag x86-micro-p --cpu 2
  targets/x86/micro_campaign.py --tag x86-micro-e --cpu 14
"""

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)

import numpy as np  # noqa: E402
from uarchre.kernels import x86_micro as xm  # noqa: E402
import hygiene  # noqa: E402

DRIVER_CPUS = set(range(4, 20)) - {14}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--tests", default="all", help="all | p16 | p18 | comma list of test names")
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--max-sibling", type=float, default=0.10)
    args = ap.parse_args()
    try:
        os.sched_setaffinity(0, DRIVER_CPUS)
    except OSError:
        pass
    tests = {"all": xm.P18 + xm.P16, "p16": xm.P16, "p18": xm.P18, "p18x": xm.P18X}.get(args.tests) or [
        xm.MICRO[t] for t in args.tests.split(",")
    ]
    outdir = os.path.join("results", "native", args.tag)
    os.makedirs(outdir, exist_ok=True)
    binpath = xm.compile_micro(tests)
    sib = hygiene.siblings(args.cpu)
    raw, launches = [], []
    L = 0
    for _ in range(args.launches * 6):
        if sum(x["clean"] for x in launches) >= args.launches:
            break
        waited = hygiene.wait_quiet(sib + [args.cpu], args.max_sibling, timeout=900 if sib else 0.5)
        load1 = os.getloadavg()[0]
        out, info = hygiene.run_monitored(
            ["taskset", "-c", str(args.cpu), binpath, str(args.reps)], sib + [args.cpu]
        )
        res = xm.parse_micro(out)
        sb = max([info["busy"].get(s, 0) for s in sib] or [0])
        clean = sb <= args.max_sibling and info["share"] >= 0.9
        launches.append(
            {
                "launch": L,
                "clean": clean,
                "waited_s": round(waited, 2),
                "load1": load1,
                "sibling_busy": sb,
                **{k: v for k, v in info.items() if k != "busy"},
            }
        )
        for name, samples in res.items():
            for r, (a, t, b) in enumerate(samples):
                raw.append(
                    {
                        "test": name,
                        "launch": L,
                        "clean": clean,
                        "rep": r,
                        "probe_pre": a,
                        "ticks_per_instr": t,
                        "probe_post": b,
                        "cycles": t / ((a + b) / 2),
                    }
                )
        print(f"launch {L}: {launches[-1]}", flush=True)
        L += 1
    good = {x["launch"] for x in launches if x["clean"]}
    if len(good) < 2:  # report everything, flagged, rather than nothing
        print("WARNING: fewer than 2 clean launches; summary uses all launches")
        good = {x["launch"] for x in launches}
    with open(os.path.join(outdir, "micro_raw.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw[0]))
        w.writeheader()
        w.writerows(raw)
    summ = {}
    for t in tests:
        c = np.array([r["cycles"] for r in raw if r["test"] == t.name and r["launch"] in good])
        per_launch = [
            float(np.median([r["cycles"] for r in raw if r["test"] == t.name and r["launch"] == L]))
            for L in sorted(good)
        ]
        all_launch = [
            float(np.median([r["cycles"] for r in raw if r["test"] == t.name and r["launch"] == x["launch"]]))
            for x in launches
        ]
        drift = [abs(r["probe_post"] / r["probe_pre"] - 1) for r in raw if r["test"] == t.name]
        summ[t.name] = {
            "kind": t.kind,
            "group": t.group,
            "label": t.label,
            "per_unit": t.per_unit,
            "median": float(np.median(c)),
            "q25": float(np.quantile(c, 0.25)),
            "q75": float(np.quantile(c, 0.75)),
            "min": float(c.min()),
            "max": float(c.max()),
            "per_launch": per_launch,
            "per_launch_all": all_launch,
            "probe_drift_median": float(np.median(drift)),
        }
        print(
            f"{t.name:16s} {summ[t.name]['median']:8.3f}  "
            f"[{summ[t.name]['q25']:.3f}, {summ[t.name]['q75']:.3f}]"
            f"  launches {[round(x, 3) for x in per_launch]}"
        )
    rep = {
        "tag": args.tag,
        "cpu": args.cpu,
        "launches": launches,
        "clean_launches": sorted(good),
        "reps": args.reps,
        "binary": binpath,
        "summary": summ,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(outdir, "micro.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"wrote {outdir}/")


if __name__ == "__main__":
    main()
