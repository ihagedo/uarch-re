#!/usr/bin/env python3
"""P1.8 through targets/m1's runner on gem5 ARM O3: does it read gem5's configured FU latencies?

The same kernels and slopes as on the M1 (scheduler.micro_op, m1_sched.run_micro). Truth is gem5's
default FU pool for each instruction's op class (src/arch/arm/isa/insts). Writes
results/gem5/p1_8-arm/micro.json.

  validation/gem5/micro/micro_arm.py        # ~5 min
"""

import json
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "targets", "m1")]
import m1_sched  # noqa: E402
import native  # noqa: E402

ISSUE_WIDTH = 8
# instruction -> (op class, opLat, units, pipelined) in gem5's DefaultFUPool
TRUTH = {
    "add": ("IntAlu", 1, 6, True),
    "csel": ("IntAlu", 1, 6, True),
    "mul": ("IntMult", 3, 2, True),
    "madd": ("IntMult", 3, 2, True),
    "udiv": ("IntDiv", 20, 2, False),
    "fadd": ("FloatAdd", 2, 4, True),
    "fmul": ("FloatMult", 4, 2, True),
    "fmadd": ("FloatMultAcc", 5, 2, True),
    "fdiv": ("FloatDiv", 12, 2, False),
    "fsqrt": ("FloatSqrt", 24, 2, False),
    "fabs": ("FloatMisc", 3, 2, True),
    "vadd": ("SimdAdd", 1, 4, True),
}


def configured(kind, lat, units, piped):
    return lat if kind == "lat" else max(1 / ISSUE_WIDTH, (1 if piped else lat) / units)


def main():
    args = types.SimpleNamespace(
        gem5_smoke=True, cc=None, retries=0, max_drift=1, min_ghz=0, l1i_line=64, pin_cpu=None, force=True
    )
    _, rows = m1_sched.run_micro(native.Runner(args), args)
    out = []
    for row in rows:
        _, t, kind = row["structure"].split(".")
        opclass, *cfg = TRUTH[t]
        e = configured(kind, *cfg)
        ok = abs(row["value"] - e) <= max(0.1, 0.05 * e)
        out.append({"test": t, "kind": kind, "opclass": opclass, "measured": row["value"], "configured": e,
                    "ok": ok, "row": row})  # fmt: skip
        mark = "" if ok else "X"
        print(f"{t:6s} {kind:4s} {opclass:13s} measured {row['value']:7.3f}  configured {e:7.3f}  {mark}")
    print(f"{sum(o['ok'] for o in out)}/{len(out)} within max(0.1, 5 %)")
    path = os.path.join(ROOT, "results", "gem5", "p1_8-arm")
    os.makedirs(path, exist_ok=True)
    json.dump(out, open(os.path.join(path, "micro.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
