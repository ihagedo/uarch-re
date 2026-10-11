#!/usr/bin/env python3
"""P3.2 L1i capacity and hit latency on gem5 O3.

Capacity: straight_nops(F), F bytes of NOP, NOP, fadd.d (fetch-bound; NOPs alone are
ALU-bound on gem5 and the window hides the first set's misses), --l1i 16-128 kB 8-way;
capacity_search() over F / line lines (`line` 64 B by default). Latency: jump_chain(512) with
icache tag and data latency L = 1..4; each taken jump should cost one more cycle per cycle of L.

    . tools/env.sh && python3 validation/gem5/fe_l1i.py [--isa aarch64] [l1i<kB> ...] [latency]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_common import run_fe, pmap, config_value, write_json, capacity_search, geom_grid, med
from fe_common import pop_isa, out_dir
from uarchre.kernels.frontend import straight_nops, jump_chain, SPELLING

SIZES_KB = [16, 32, 64, 128]
GRID = geom_grid(64, 4096, 4)  # lines: 4 kB .. 256 kB
REPS = 3


def laps_for(n):
    """Laps per rep for n lines: 256 up to 2048 lines, fewer far above any onset.
    The harness's own code shares L1i sets with the kernel and thrashes a full set
    once per rep; enough laps keep that well below tau (too few laps read low)."""
    return max(16, min(256, (1 << 19) // n))


LAT = [1, 2, 3, 4]
ARM = {"sizes": [16, 64, 128], "lat": [1, 3]}  # --isa aarch64: a reduced sweep


def run_size(kb, isa="riscv64", line=64):
    cargs = ["--l1i", f"{kb}kB", "--l2", "1MB"]  # every footprint fits twice in L2
    sp = SPELLING[isa]

    def measure(n):
        r = run_fe(
            straight_nops(line * n, isa=sp),
            laps=laps_for(n),
            reps=REPS,
            config_args=cargs,
            stats=["system.cpu.icache.overallMisses::total"],
        )
        return med(r["cpu"]) * line  # cycles per line

    r = run_fe(straight_nops(line * GRID[0], isa=sp), laps=laps_for(GRID[0]), reps=REPS, config_args=cargs)
    got = config_value(r["outdir"], "system.cpu.icache", "size")
    assert got == str(kb * 1024), (kb, got)
    res = capacity_search(measure, GRID, step=1, log=print, tag=f"l1i{kb}")
    out = {
        "name": f"l1i{kb}",
        "truth_bytes": kb * 1024,
        "truth_lines": kb * 1024 // line,
        "points": {str(k): v for k, v in sorted(res["points"].items())},
        "base": res["base"],
        "tau": res["tau"],
        "bracket": res["bracket"],
        "coarse_bracket": res["coarse_bracket"],
        "rounds": res["rounds"],
        "excess": {str(k): v for k, v in res.get("excess", {}).items()},
        "error": res.get("error"),
    }
    write_json(os.path.join(out_dir("fe-l1i", isa), f"l1i{kb}.json"), out)
    print(f"l1i {kb} kB: truth {kb * 1024 // line} lines, bracket {res['bracket']}", flush=True)
    return out


def run_latency(isa="riscv64"):
    def one(L):
        sets = [
            f"cpu.icache.tag_latency={L}",
            f"cpu.icache.data_latency={L}",
            "cpu.branchPred.instShiftAmt=2",
        ]
        r = run_fe(jump_chain(512, 0, isa=SPELLING[isa]), laps=64, reps=REPS, sets=sets)
        assert config_value(r["outdir"], "system.cpu.icache", "data_latency") == str(L)
        return {"latency": L, "cycles_per_jump": med(r["cpu"]), "reps": r["cpu"]}

    rows = pmap(one, LAT if isa == "riscv64" else ARM["lat"])
    write_json(os.path.join(out_dir("fe-l1i", isa), "latency.json"), rows)
    for r in rows:
        print(f"icache latency {r['latency']}: {r['cycles_per_jump']:.3f} cycles per taken jump", flush=True)
    return rows


def main(args):
    from concurrent.futures import ThreadPoolExecutor

    isa, args = pop_isa(args)
    sizes = SIZES_KB if isa == "riscv64" else ARM["sizes"]
    with ThreadPoolExecutor(len(sizes) + 1) as ex:
        futs = [ex.submit(run_size, kb, isa) for kb in sizes if not args or f"l1i{kb}" in args]
        if not args or "latency" in args:
            futs.append(ex.submit(run_latency, isa))
        for f in futs:
            f.result()


if __name__ == "__main__":
    main(sys.argv[1:])
