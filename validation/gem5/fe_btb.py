#!/usr/bin/env python3
"""P3.3 BTB capacity on gem5 O3: N jumps in consecutive 4-byte slots, visited in random order.

Past capacity a jump misses the BTB and decode redirects it (an extra bubble);
capacity_search() finds the onset to one entry and seeds 1 and 2 re-check it.
gem5's instShiftAmt defaults to 0, so 4-byte RISC-V instructions reach every fourth
BTB set; the main series sets 2, the shift0 series keeps 0 (truth numEntries / 4).

    . tools/env.sh && python3 validation/gem5/fe_btb.py [config-name ...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_common import RESULTS, run_fe, pmap, config_value, write_json, capacity_search, geom_grid, med
from uarchre.kernels.frontend import jump_chain, lap_excess

OUT = os.path.join(RESULTS, "fe-btb")
L1I = ["--l1i", "256kB", "--l2", "1MB"]  # code up to 64 kB stays in L1i: the BTB is the only limit
REPS = 3


def laps_for(n):
    """Laps per rep: 128 near any onset in the grid, fewer far above it. The harness's
    own taken branches collide with chain jumps once every set is full (~2-5 extra
    misses per rep); 128 laps keeps that well under tau (~1.5 cycles per lap)."""
    return max(16, min(128, (1 << 19) // n))


SEEDS = (0, 1, 2)
GRID = geom_grid(32, 12288, 4)

# name: (numEntries, associativity, instShiftAmt, effective truth)
CONFIGS = {f"dm{n}": (n, 1, 2, n) for n in (256, 512, 1024, 2048, 4096)}
CONFIGS.update({f"a4_{n}": (n, 4, 2, n) for n in (256, 1024, 4096)})
CONFIGS.update({f"shift0_{n}": (n, 1, 0, n // 4) for n in (1024, 4096)})


def sets_for(cfg):
    n, a, sh, _ = cfg
    return [
        f"cpu.branchPred.btb.numEntries={n}",
        f"cpu.branchPred.btb.associativity={a}",
        f"cpu.branchPred.instShiftAmt={sh}",
    ]


def run_config(name):
    cfg = CONFIGS[name]
    sets = sets_for(cfg)

    def measure_seed(n, seed):
        r = run_fe(
            jump_chain(n, seed),
            laps=laps_for(n),
            reps=REPS,
            sets=sets,
            config_args=L1I,
            stats=["system.cpu.branchPred.btb.misses::DirectUncond"],
        )
        return med(r["cpu"]), r

    def measure(n):
        return measure_seed(n, 0)[0]

    # confirm the overrides took
    _, r = measure_seed(GRID[0], 0)
    for key, want in (("numEntries", cfg[0]), ("associativity", cfg[1]), ("instShiftAmt", cfg[2])):
        got = config_value(r["outdir"], "system.cpu.branchPred.btb", key)
        assert got == str(want), (name, key, got, want)

    res = capacity_search(measure, GRID, step=1, log=print, tag=name)
    out = {
        "name": name,
        "numEntries": cfg[0],
        "assoc": cfg[1],
        "instShiftAmt": cfg[2],
        "truth": cfg[3],
        "points": {str(k): v for k, v in sorted(res["points"].items())},
        "base": res["base"],
        "tau": res["tau"],
        "bracket": res["bracket"],
        "coarse_bracket": res["coarse_bracket"],
        "rounds": res["rounds"],
        "excess": {str(k): v for k, v in res.get("excess", {}).items()},
        "error": res.get("error"),
    }
    # placement seeds: rerun the plateau points and the bracket ends
    if res["bracket"] and res["bracket"][0] is not None:
        lo, hi = res["bracket"]
        per_seed = {}
        for seed in SEEDS[1:]:
            ns = sorted(set(res["base"]) | {lo, hi})
            ys = pmap(lambda n: measure_seed(n, seed)[0], ns)  # noqa: B023
            exc = dict(zip(ns, lap_excess(ns, ys, res["base"])))
            per_seed[seed] = {
                "lo_excess": float(exc[lo]),
                "hi_excess": float(exc[hi]),
                "consistent": bool(exc[lo] <= res["tau"] < exc[hi]),
                "points": {str(n): y for n, y in zip(ns, ys)},
            }
        out["seeds"] = per_seed
    write_json(os.path.join(OUT, f"{name}.json"), out)
    b = out["bracket"]
    det = b[0] if b and b[0] is not None else None
    print(
        f"{name}: truth {cfg[3]}  detected {det}  bracket {b}  seeds consistent "
        f"{[s['consistent'] for s in out.get('seeds', {}).values()]}",
        flush=True,
    )
    return out


def main(names):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(names)) as ex:
        outs = list(ex.map(run_config, names))
    return outs


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))
