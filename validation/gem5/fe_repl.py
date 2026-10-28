#!/usr/bin/env python3
"""P3.2-repl through targets/m1's runner on gem5 ARM O3, one L1i replacement policy per case.

LRU, FIFO, TreePLRU and Random at 8 ways (32 KiB); LRU, FIFO, Random and TreePLRU (heap layout, lopsided
at 6 and 12) at 6 and 12 ways of 4 KiB; BRRIP, second chance and MRU at 8 ways, which the kernel does not
model, as controls. Writes results/gem5/fe-repl/cases.json.

  validation/gem5/fe_repl.py [--only 8|6|12|controls]
"""

import argparse
import json
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "targets", "m1")]
import m1_fe  # noqa: E402
import native  # noqa: E402
from uarchre.backends.platform.gem5 import run_gem5_binary  # noqa: E402

GEO = {w: [f"cpu.icache.size={4 * w}kB", f"cpu.icache.assoc={w}"] for w in (6, 12)}
CASES = {
    "8": [([], p) for p in ("LRURP()", "FIFORP()", "TreePLRURP()", "RandomRP()")],
    "6": [(GEO[6], p) for p in ("LRURP()", "FIFORP()", "TreePLRURP()", "RandomRP()")],
    "12": [(GEO[12], p) for p in ("LRURP()", "FIFORP()", "TreePLRURP()", "RandomRP()")],
    "controls": [([], p) for p in ("BRRIPRP()", "SecondChanceRP()", "MRURP()")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(CASES))
    a = ap.parse_args()
    args = types.SimpleNamespace(
        gem5_smoke=True, cc=None, retries=0, max_drift=1, min_ghz=0, l1i_line=64, pin_cpu=None, force=True
    )
    out = []
    for group in [a.only] if a.only else CASES:
        for geo, pol in CASES[group]:
            sets = [*geo, f"cpu.icache.replacement_policy={pol}"]
            native.Runner._exec = lambda self, b, argv, timeout, sets=sets: run_gem5_binary(
                b, [str(x) for x in argv], sets=sets, isa="aarch64", timeout=timeout
            )[0]
            _, rows = m1_fe.run_repl(native.Runner(args), args)
            ways, repl = rows[0], rows[-1]
            print(
                f"{group:8s} {pol:16s} ways {ways['value']}  {repl['value']} {' '.join(repl['flags'])}",
                flush=True,
            )
            out.append({"group": group, "policy": pol, "sets": sets, "rows": rows})
    path = os.path.join(ROOT, "results", "gem5", "fe-repl")
    os.makedirs(path, exist_ok=True)
    json.dump(out, open(os.path.join(path, "cases.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
