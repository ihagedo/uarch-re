#!/usr/bin/env python3
"""P3.9 fetch-to-decode queue depth on gem5 O3 (RISC-V).

fetchq_source(B): an fsqrt chain stalls the backend, a CSR read serializes rename, then
B NOPs and a jump to a line that misses in L1i and L2; the miss is hidden while the
frontend holds all B NOPs. The knee counts all buffering before rename: knee = Q + C,
C estimated leave-one-out. Three stall lengths check that the knee does not depend on it.

  . tools/env.sh && python3 validation/gem5/fp_fetchq.py [--analyze]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_common import RESULTS, run_points, config_value, pmap, save_json, load_json, log, g5
from uarchre.kernels.predictors import fetchq_source
from uarchre.analysis.knee import find_knee
from uarchre.analysis.bootstrap import bootstrap_knee
from uarchre.viz import tolerance

OUT = os.path.join(RESULTS, "fp-fetchq")
SIZES = [16, 24, 32, 48, 64]
CHAINS = [10, 12, 14]
# block k starts at page offset 64k: past B = 113 nine blocks share L1i sets and thrash
# the 8-way L1i (a Q-independent step at B ~ 120), so sweep below it
MAXB = 112
STEP = 8
PACK, STRIDE = 16, 64 << 10  # 16 B values per binary; landing lines 64 KiB apart (one L2 set per value)


def run_one(job):
    """One gem5 run: up to 8 B values for one (Q, chain)."""
    q, chain, bs = job
    pts, od = run_points(
        fetchq_source(list(bs), chain=chain, stride=STRIDE),
        f"fp_fq_c{chain}",
        sets=[f"cpu.fetchQueueSize={q}"],
    )
    took = int(config_value(od, "system.cpu", "fetchQueueSize"))
    misses = g5.stat(od, "system.cpu.icache.overallMisses::total")
    return [
        {
            "q": q,
            "chain": chain,
            "b": b,
            "cyc": pts[str(b)],
            "outdir": od,
            "config_fetchQueueSize": took,
            "icache_misses_run": misses,
        }
        for b in bs
    ]


def chunks(jobs):
    """Group (q, chain, b) into runs of <= PACK B values per (q, chain)."""
    out = []
    for key in sorted({(q, c) for q, c, _ in jobs}):
        bs = sorted(b for q, c, b in jobs if (q, c) == key)
        out += [(*key, tuple(bs[i : i + PACK])) for i in range(0, len(bs), PACK)]
    return out


def runs(jobs):
    return [r for rows in pmap(run_one, chunks(jobs)) for r in rows]


def knee_of(rows):
    bs = sorted({r["b"] for r in rows if r["b"] <= MAXB})
    y = [float(np.median(next(r["cyc"] for r in rows if r["b"] == b))) for b in bs]
    return find_knee(bs, y)


def campaign():
    raw_path = os.path.join(OUT, "raw.json")
    raw = load_json(raw_path) if os.path.exists(raw_path) else []
    done = {(r["q"], r["chain"], r["b"]) for r in raw}
    coarse = [
        (q, c, b) for q in SIZES for c in CHAINS for b in range(0, MAXB + 1, STEP) if (q, c, b) not in done
    ]
    raw += runs(coarse)
    save_json(raw_path, raw)
    fine = []
    for q in SIZES:
        for c in CHAINS:
            k = knee_of([r for r in raw if r["q"] == q and r["chain"] == c])
            if k.knee is None:
                log("fp-fetchq", f"Q={q} chain={c}: no coarse knee ({k.anomaly})")
                continue
            fine += [(q, c, b) for b in range(int(k.knee) + 1, int(k.knee) + STEP)]
    fine = sorted(set(fine) - {(r["q"], r["chain"], r["b"]) for r in raw})
    raw += runs(fine)
    save_json(raw_path, raw)
    return raw


def analyze(raw):
    out = []
    for q in sorted({r["q"] for r in raw}):
        rows = [r for r in raw if r["q"] == q and r["b"] <= MAXB]
        bs = sorted({r["b"] for r in rows})
        chains = sorted({r["chain"] for r in rows})
        per_chain = {}
        for c in chains:
            per_chain[c] = {r["b"]: float(np.median(r["cyc"])) for r in rows if r["chain"] == c}
        common = [b for b in bs if all(b in per_chain[c] for c in chains)]
        reps = {b: [per_chain[c][b] for c in chains] for b in common}
        kn = bootstrap_knee(reps)
        per_chain_knee = {c: knee_of([r for r in rows if r["chain"] == c]).knee for c in chains}
        out.append(
            {
                "q": q,
                "knee": kn.knee,
                "ci": kn.ci,
                "ci_method": kn.ci_method,
                "flags": kn.flags,
                "jump_ratio": kn.jump_ratio,
                "quality": kn.quality,
                "knee_per_chain": per_chain_knee,
                "curve": {str(c): {str(b): per_chain[c][b] for b in sorted(per_chain[c])} for c in chains},
                "config_fetchQueueSize": sorted({r["config_fetchQueueSize"] for r in rows}),
            }
        )
    # knee = Q + C: slope over all, and leave-one-out C for each Q
    qs = np.array([s["q"] for s in out], float)
    ks = np.array([s["knee"] for s in out], float)
    slope, icpt = np.polyfit(qs, ks, 1)
    for i, s in enumerate(out):
        others = [j for j in range(len(out)) if j != i]
        c_loo = float(np.median(ks[others] - qs[others]))
        s["offset_loo"] = c_loo
        s["detected"] = s["knee"] - c_loo
        s["ci_detected"] = [s["ci"][0] - c_loo, s["ci"][1] - c_loo]
        s["error"] = s["detected"] - s["q"]
        s["pass"] = bool(
            abs(s["error"]) <= float(tolerance(s["q"]))
            and s["ci_detected"][0] - 0.5 <= s["q"] <= s["ci_detected"][1] + 0.5
        )
    return {
        "points": out,
        "slope": float(slope),
        "intercept": float(icpt),
        "offset_all": float(np.median(ks - qs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    raw = load_json(os.path.join(OUT, "raw.json")) if a.analyze else campaign()
    summ = analyze(raw)
    save_json(os.path.join(OUT, "summary.json"), summ)
    log(
        "fp-fetchq",
        f"slope {summ['slope']:.3f} intercept {summ['intercept']:.2f} offset(all) {summ['offset_all']}",
    )
    for s in summ["points"]:
        log(
            "fp-fetchq",
            f"Q={s['q']:3d} knee={s['knee']} ci={s['ci']} per-chain={s['knee_per_chain']} "
            f"C_loo={s['offset_loo']:.1f} detected={s['detected']:.1f} err={s['error']:.1f} "
            f"pass={s['pass']} jump={s['jump_ratio']:.2f}",
        )


if __name__ == "__main__":
    main()
