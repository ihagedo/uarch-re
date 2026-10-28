#!/usr/bin/env python3
"""SimEng M1 configs: configs.py OUTDIR SPECIAL_DIR.

Bristol's m1_firestorm.yaml with current key names and every data access 300 cycles (no caches without
SST, so the window probes need a slow miss), and one variant per disputed size. gen.yaml writes the
special-files directory once (run simeng on it); the others read it, so runs can go in parallel.
"""

import copy
import os
import sys

import yaml

SRC = os.path.join(
    os.environ.get("SIMENG_SRC", f"/var/tmp/{os.environ.get('USER')}/uarch-re/simeng/src"),
    "configs",
    "m1_firestorm.yaml",
)
RENAME = {
    "L1 Load Bandwidth": "Load-Bandwidth",
    "L1 Store Bandwidth": "Store-Bandwidth",
    "Permitted Requests-Per-Cycle": "Permitted-Requests-Per-Cycle",
}
INT, LS, FP = range(0, 6), range(6, 10), range(10, 14)


def base(special, generate=False):
    c = yaml.safe_load(open(SRC))
    c["LSQ-L1-Interface"] = {RENAME.get(k, k): v for k, v in c["LSQ-L1-Interface"].items()}
    c["LSQ-L1-Interface"]["Access-Latency"] = 300
    c["CPU-Info"]["Generate-Special-Dir"] = generate
    c["CPU-Info"]["Special-File-Dir-Path"] = special
    return c


def sized(c, ids, n):
    c = copy.deepcopy(c)
    for i in ids:
        c["Reservation-Stations"][i]["Size"] = n
    return c


def ls_unified(c):
    """One 48-entry station feeding the four load/store ports."""
    c = copy.deepcopy(c)
    rs = c["Reservation-Stations"]
    ports = [p for i in LS for p in rs[i]["Ports"]]
    rest = [rs[i] for i in sorted(rs) if i not in LS]
    merged = {"Size": 48, "Dispatch-Rate": rs[LS[0]]["Dispatch-Rate"], "Ports": ports}
    c["Reservation-Stations"] = dict(enumerate(rest[: len(INT)] + [merged] + rest[len(INT) :]))
    return c


NOCMP = (
    ["VECTOR"]
    + [
        f"SCALAR_{g}"
        for g in (
            "SIMPLE_ARTH",
            "SIMPLE_ARTH_NOSHIFT",
            "SIMPLE_LOGICAL",
            "SIMPLE_LOGICAL_NOSHIFT",
            "SIMPLE_CVT",
        )
    ]
    + ["SCALAR_MUL", "SCALAR_DIV_OR_SQRT"]
)


def fp_unified(c, n):
    """One n-entry station feeding the four FP ports; fcmp (SCALAR_SIMPLE_CMP) issues on the first only."""
    c = copy.deepcopy(c)
    rs = c["Reservation-Stations"]
    ports = [p for i in FP for p in rs[i]["Ports"]]
    for i in FP[1:]:
        c["Ports"][i]["Instruction-Group-Support"] = NOCMP
    rest = [rs[i] for i in sorted(rs) if i not in FP]
    c["Reservation-Stations"] = dict(enumerate(rest + [{"Size": n, "Dispatch-Rate": 16, "Ports": ports}]))
    c["Port-Allocator"] = {
        "Type": "Balanced"
    }  # the M1 allocator sends a shared station's ops to its first port
    return c


def fp_inflow1(c):
    """The four FP stations take one op a cycle each (the KB's queue inflow); fcmp on the first port."""
    c = copy.deepcopy(c)
    for i in FP:
        c["Reservation-Stations"][i]["Dispatch-Rate"] = 1
    for i in FP[1:]:
        c["Ports"][i]["Instruction-Group-Support"] = NOCMP
    return c


out, special = sys.argv[1], sys.argv[2]
b = base(special)
variants = {
    "gen": base(special, generate=True),
    "base": b,  # int 24/26/16/12/28/28, LS 4 x 12, FP 4 x 36
    "fp18": sized(b, FP, 18),
    "int13": sized(b, INT, 13),
    "ls1": ls_unified(b),
    "lq151": {**b, "Queue-Sizes": {**b["Queue-Sizes"], "Load": 151}},
    "lb0": {**b, "Fetch": {**b["Fetch"], "Loop-Buffer-Size": 0}},
    "lb24": {**b, "Fetch": {**b["Fetch"], "Loop-Buffer-Size": 24}},
    "fpu48": fp_unified(b, 48),
    "fpu24": fp_unified(b, 24),
    "fpin1": fp_inflow1(b),
    "fpin1-18": fp_inflow1(sized(b, FP, 18)),
}
for name, c in variants.items():
    with open(os.path.join(out, f"{name}.yaml"), "w") as fh:
        yaml.safe_dump(c, fh, sort_keys=False)
print(" ".join(sorted(variants)))
