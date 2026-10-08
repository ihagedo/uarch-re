#!/usr/bin/env python3
"""P6.1 closed-loop round trip on gem5.

Measure a "mystery" core with all five window fillers (results/gem5/p61-a/spec.yaml),
export a gem5 config from that spec (exporters/gem5_from_spec.py), measure the export
(p61-b) and pass if every P1 structure's second reading falls inside the first's interval
and both match the hidden truth. Needs the gem5 offsets in uarchre/analysis/offsets.yaml.

  validation/gem5/p61_roundtrip.py
"""

import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from exporters.gem5_from_spec import export

MYSTERY = dict(
    numROBEntries=160, numPhysIntRegs=32 + 120, numPhysFloatRegs=32 + 100, LQEntries=40, SQEntries=28
)
FILLERS = "nop,int,fp,load,store"


def measure(tag, params):
    out = os.path.join(ROOT, "results", "gem5", tag)
    if not os.path.exists(os.path.join(out, "knees.json")):
        gem5 = ",".join(f"{k}={v}" for k, v in params.items())
        subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "sweep.py"),
                "gem5",
                "--tag",
                tag,
                "--gem5",
                gem5,
                "--fillers",
                FILLERS,
                "--n",
                "8:184:8",
                "--depth",
                "1",
                "--chain-mb",
                "4",
                "--iters",
                "40",
                "--outer-reps",
                "1",
                "--jobs",
                "12",
                "--no-plot",
            ],
            cwd=ROOT,
            check=True,
        )
    with open(os.path.join(out, "knees.json")) as fh:
        kj = json.load(fh)
    with open(os.path.join(out, "spec.yaml")) as fh:
        return kj, yaml.safe_load(fh)


def main():
    kj_a, spec_a = measure("p61-a", MYSTERY)
    exported = export(spec_a)
    print("exported from spec:", exported)
    kj_b, _ = measure("p61-b", exported)
    rows, ok = [], True
    for key in ["rob", "int_prf_headroom", "fp_prf_headroom", "lq", "sq"]:
        a, b, truth = kj_a["estimates"][key], kj_b["estimates"][key], kj_a["truth"][key]
        inside = a["ci"] is not None and b["value"] is not None and a["ci"][0] <= b["value"] <= a["ci"][1]
        exact = a["value"] is not None and abs(a["value"] - truth) <= max(2, 0.02 * truth)
        ok &= inside and exact
        rows.append(
            dict(
                structure=key,
                truth=truth,
                first=a["value"],
                first_ci=a["ci"],
                second=b["value"],
                second_ci=b["ci"],
                inside=inside,
                first_within_tolerance=exact,
                anomalies=sorted(set(a["anomalies"]) | set(b["anomalies"])),
            )
        )
        print(
            f"{key:18s} truth {truth:4}  first {a['value']} {a['ci']}  second {b['value']} {b['ci']}  "
            f"{'ok' if inside and exact else 'FAIL'}  {' '.join(rows[-1]['anomalies'])}"
        )
    with open(os.path.join(ROOT, "results", "gem5", "p61-summary.json"), "w") as fh:
        json.dump(dict(mystery=MYSTERY, exported=exported, rows=rows, ok=ok), fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
