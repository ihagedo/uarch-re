#!/usr/bin/env python3
"""P0.3 anomaly rule suite: run every fixture and record expected vs fired rule ids.

The toy curves of tests/fixtures/toy_cases.yaml are cached in curves.json (--reuse skips
the runs). Writes results/toy/mech-fixtures/{curves.json, fixtures.json}.

  taskset -c 4-19 python3 validation/toy/mech_fixtures.py [--jobs 6] [--reuse]
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from tests.fixtures import anomaly_cases as AC
from uarchre.analysis.anomaly import evaluate, fired, RULE_IDS

OUT = os.path.join(ROOT, "results", "toy", "mech-fixtures")


def pin():
    try:
        os.sched_setaffinity(0, set(range(4, 20)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    pin()
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "curves.json")
    t0 = time.time()
    if args.reuse and os.path.exists(cache):
        curves = AC.load_curves(cache)
    else:
        curves = AC.run_toy_curves(jobs=args.jobs)
        AC.save_curves(curves, cache)
    print(f"toy curves: {sum(len(v) for v in curves.values())} in {time.time() - t0:.0f} s")

    cases = AC.all_fast_cases() + [("toy",) + c for c in AC.toy_runsets(curves)]
    rows, bad = [], 0
    for kind, name, rs, expect in cases:
        fs = evaluate(rs)
        got = fired(fs)
        ok = got == expect
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {kind:5s} {name:26s} expect {expect} fired {got}")
        if not ok or kind in ("toy", "pub"):
            for f in fs:
                print(f"       {f.code:3s} {f.rule:26s} {f.message[:170]}")
        rows.append(
            {
                "kind": kind,
                "name": name,
                "expect": expect,
                "fired": got,
                "ok": ok,
                "findings": [f.to_dict() for f in fs],
                "estimates": {
                    "/".join(map(str, k)): (c.estimate() and [c.estimate()[0], c.estimate()[1]])
                    for k, c in rs.curves.items()
                },
            }
        )
    with open(os.path.join(OUT, "fixtures.json"), "w") as fh:
        json.dump({"rules": RULE_IDS, "fixtures": rows}, fh, indent=1)
    print(
        f"\n{len(rows) - bad}/{len(rows)} fixtures fire exactly their expected rules; "
        f"wrote {os.path.relpath(OUT, ROOT)}/fixtures.json"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
