#!/usr/bin/env python3
"""P2 memory probes on Apple silicon: caches, TLBs, MLP, reliable misses and prefetch, disambiguation.

    python3 targets/m1/m1_mem.py --tag m1 [P2.1 P2.2 P2.3 P2.4 P2.5]
    python3 targets/m1/m1_mem.py --gem5-smoke --tag smoke

Sweeps bracket spec/cores/firestorm.yaml with margin, each point timed in several clocked launches
(native.py) and read in core cycles. Writes results/m1/<tag>/mem/<PoC>/.
"""

import argparse
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native  # noqa: E402

sys.path.insert(0, os.path.join(native.ROOT, "validation", "gem5"))
from uarchre.kernels import memory as M  # noqa: E402
from uarchre.kernels.memory import plateau_onset  # noqa: E402
import mem_cache  # noqa: E402
import mem_memdep  # noqa: E402
import mem_mlp  # noqa: E402

FAMILY = "mem"
KB, MB = 1 << 10, 1 << 20
MD_NMAX = 508  # memdep buffer: the no-alias loads sit 8 KiB up in a 16 KiB buffer
DRAM_TOL = 0.05  # a window chain is a DRAM miss within 5 % of the random chain's latency
PREFETCHED = 0.5  # a dependent stride chain at under half the random chain's latency
MLP_K = list(range(1, 17)) + list(range(18, 33, 2)) + list(range(36, 57, 4))


def geom(lo, hi, per_octave, align=1):
    """lo * 2^(k / per_octave) up to hi, rounded to multiples of align."""
    out, k = set(), 0
    while lo * 2 ** (k / per_octave) <= hi * (1 + 1e-9):
        out.add(int(round(lo * 2 ** (k / per_octave) / align)) * align)
        k += 1
    return sorted(out)


def plan(smoke=False):
    """Every sweep's points and lengths; smoke: gem5 ARM's default O3 (32 KiB L1D, 256 KiB L2, 4 KiB pages)"""
    if smoke:
        return {
            "P2.1": dict(
                coarse=[([8 * KB, 16 * KB, 64 * KB, 128 * KB, 512 * KB, 1 * MB], 1000)],
                fine_cap=3,
                reps=2,
                lat=dict(l1d=[0, 16 * KB], l2=[64 * KB, 128 * KB], dram=[1 * MB, 1 * MB]),
                pair=[(128 * KB, 1000), (1 * MB, 1000)],
                slot=1024,
                pair_d=[8, 64, 128, 512],
                conflict_s=[16 * KB],
                conflict_k=list(range(1, 11)),
                conflict_nloads=1000,
            ),
            "P2.2": dict(page=[16, 64, 256], nloads=1000, code=[4, 16, 64], njumps=1000, fine_cap=3, reps=2),
            "P2.3": dict(F=4 * MB, iters=100, reps=1, x=None, K=[1, 2, 4, 8, 12, 16, 24]),
            "P2.4": dict(
                F=2 * MB,
                nloads=1000,
                indep_nloads=1000,
                wchain=[2 * MB],
                wchain_nloads=1000,
                seq=[64, 256],
                reps=2,
            ),
            "P2.5": dict(N=[4, 8, 16, 32], control=[4, 16], blocks=400, reps=2, fwd=500, fine_cap=3),
        }
    md_n = sorted({int(round(4 * 2 ** (k / 4))) for k in range(0, 28)} | {MD_NMAX})  # 4 .. 431, 508
    return {
        "P2.1": dict(
            coarse=[
                (geom(16 * KB, 16 * MB, 4, 64), 2_000_000),
                ([f for f in geom(16 * KB, 512 * MB, 4, 64) if f > 16 * MB], 200_000),
            ],
            fine_cap=48,
            reps=5,
            lat=dict(l1d=[0, 64 * KB], l2=[512 * KB, 2 * MB], dram=[256 * MB, 512 * MB]),
            pair=[(2 * MB, 1_000_000), (256 * MB, 100_000)],
            slot=1024,
            pair_d=mem_cache.PAIR_D,
            conflict_s=[16 * KB, 64 * KB],
            conflict_k=list(range(1, 33)),
            conflict_nloads=2_000_000,
        ),
        "P2.2": dict(
            page=geom(16, 16384, 8),
            nloads=2_000_000,
            code=geom(8, 4096, 6),
            njumps=2_000_000,
            fine_cap=24,
            reps=5,
        ),
        "P2.3": dict(F=512 * MB, iters=12000, reps=3, x=64 * MB, K=MLP_K),
        "P2.4": dict(
            F=256 * MB,
            nloads=1_000_000,
            indep_nloads=4_000_000,
            wchain=[64 * MB, 256 * MB],
            wchain_nloads=50_000,
            seq=[64, 128, 256, 512, 1024],
            reps=5,
        ),
        "P2.5": dict(N=md_n, control=md_n, blocks=3_000_000, reps=5, fwd=200_000, fine_cap=48),
    }


def convert(text, ratio):
    """One launch's stdout: every R value in core cycles (divided by ratio), the scalars, page size."""
    runs, sc = M.parse_runs(text)
    pb = re.search(r"^page_bytes (\d+)$", text, re.M)
    if pb:
        sc["page_bytes"] = int(pb.group(1))
    return {
        "ratio": ratio,
        "scalars": sc,
        "runs": {t: {str(k): [v / ratio for v in vs] for k, vs in d.items()} for t, d in runs.items()},
    }


def sweep(r, binpath, head, keys, label, n):
    """n launches of head + keys, keys shuffled per launch so slow drift is not read as a size effect."""
    out = []
    for i in range(n):
        ks = list(keys)
        random.Random(1000 * i + len(ks)).shuffle(ks)
        h = head(i) if callable(head) else head
        text, ratio = r.launch(binpath, [*h, *ks], label=f"{label}/{i}")
        out.append({**convert(text, ratio), "argv": [*h, *ks]})
    return out


def merge(sweeps):
    """Launch i of every sweep as one launch (their keys are disjoint)."""
    out = []
    for parts in zip(*sweeps):
        runs = defaultdict(dict)
        for p in parts:
            for t, d in p["runs"].items():
                runs[t].update(d)
        out.append({"ratio": [p["ratio"] for p in parts], "scalars": parts[0]["scalars"], "runs": dict(runs)})
    return out


def pooled(ls, tag):
    """(xs, ys): per key, the median over every rep of every launch."""
    acc = defaultdict(list)
    for lr in ls:
        for k, v in lr["runs"].get(tag, {}).items():
            acc[int(k)].extend(v)
    xs = sorted(acc)
    return xs, [statistics.median(acc[x]) for x in xs]


def each(ls, tag):
    return [pooled([lr], tag) for lr in ls]


def diff(a, b):
    """Common keys of two (xs, ys) curves and a - b there."""
    da, db = dict(zip(*a)), dict(zip(*b))
    xs = sorted(set(da) & set(db))
    return xs, [da[x] - db[x] for x in xs]


def region(ls, tag, lo, hi):
    """Median of the points with lo <= key <= hi over every launch, and the range of per-launch medians."""
    pick = [v for lr in ls for k, vs in lr["runs"].get(tag, {}).items() if lo <= int(k) <= hi for v in vs]
    per = [region([lr], tag, lo, hi)[0] for lr in ls] if len(ls) > 1 else []
    per = [v for v in per if v is not None]
    return (statistics.median(pick) if pick else None), ([min(per), max(per)] if per else None)


def onset(pool, curves, lo, hi, unit):
    """plateau_onset (10 % of the step) on the pooled curve; ci spans every curve's [last, first - unit]."""
    if len(pool[0]) < 2:
        return None, None
    value = plateau_onset(*pool, lo, hi, frac=0.1)[0]
    br = []
    for xs, ys in [pool, *curves]:
        last, first = plateau_onset(xs, ys, lo, hi, frac=0.1) if len(xs) > 1 else (None, None)
        if last is not None:
            br.append((last, (max(xs) + unit if first is None else first) - unit))
    return value, ([min(b[0] for b in br), max(b[1] for b in br)] if value is not None else None)


def steps(xs, ys, tol=1.0, min_rise=2.0):
    """Rises of a cost curve between plateaus (runs within tol + 15 %): [(last x, first x, lower, upper)]."""
    pts = sorted(zip(xs, ys))
    if not pts:
        return []
    groups = [[pts[0]]]
    for x, y in pts[1:]:
        g = groups[-1]
        if all(abs(y - gy) <= tol + 0.15 * abs(gy) for gy in (g[0][1], g[-1][1])):
            g.append((x, y))
        else:
            groups.append([(x, y)])
    plats = [g for g in groups if len(g) >= 2]
    out = []
    for a, b in zip(plats, plats[1:]):
        ya, yb = statistics.median(y for _, y in a), statistics.median(y for _, y in b)
        if yb - ya >= min_rise:
            out.append((a[-1][0], b[0][0], ya, yb))
    return out


def thin(g, cap):
    """At most about cap points of g, keeping both ends."""
    if len(g) <= cap:
        return list(g)
    k = math.ceil(len(g) / cap)
    return sorted(set(g[::k]) | {g[-1]})


def spread(vals):
    vals = [v for v in vals if v is not None]
    return [min(vals), max(vals)] if vals else None


def pub(path):
    """(published value or range, "path [sources]") from firestorm.yaml."""
    if not path:
        return None, ""
    node = native.published(path)
    parent = node if isinstance(node, dict) else native.published(path.rsplit(".", 1)[0])
    if isinstance(node, dict):
        node = node.get("range") or node.get("value")
    src = (parent or {}).get("sources") or []
    return node, f"{path} [{', '.join(src)}]" if src else path


def row(structure, value, ci, unit, published=None, flags=(), note=""):
    p, src = pub(published)
    rnd = lambda v: None if v is None else round(float(v), 3)  # noqa: E731
    return {
        "structure": structure,
        "value": rnd(value),
        "ci": None if ci is None else [rnd(c) for c in ci],
        "unit": unit,
        "published": p,
        "source": src,
        "flags": list(flags),
        "note": note,
    }


def scale(v, ci, k):
    return (None if v is None else v / k), (None if ci is None or not k else [c / k for c in ci])


def size(b):
    return f"{b // MB} MiB" if b >= MB else f"{b // KB} KiB"


def page_bytes(ls):
    return next((lr["scalars"]["page_bytes"] for lr in ls if "page_bytes" in lr["scalars"]), None)


def cache_transitions(xs, ys):
    """(L1D, L2) on the coarse random-chain curve: the first two transitions that at least double it."""
    lv = [t for t in mem_cache.levels(xs, ys) if t[3] >= 2 * t[2]] if len(xs) > 3 else []
    return (lv + [None, None])[:2]


def run_caches(r, args):
    p, n = plan(args.gem5_smoke)["P2.1"], launches(args)
    b = r.compile(M.chase_source("aarch64", r.counter), "memprobe")
    reps = p["reps"]
    raw = {"plan": p}
    raw["coarse"] = merge(
        [sweep(r, b, ["lat", "rand", 64, nl, reps, 0, 1], fs, "P2.1 coarse", n) for fs, nl in p["coarse"]]
    )
    raw["fine"] = {}
    for name, t in zip(("l1d", "l2"), cache_transitions(*pooled(raw["coarse"], "lat"))):
        if t is None:
            continue
        fs = thin(mem_cache.fine_grid(t[0], t[1]), p["fine_cap"])
        nl = next((nl for g, nl in p["coarse"] if max(fs) <= max(g)), p["coarse"][-1][1])
        ls = sweep(r, b, ["lat", "rand", 64, nl, reps, 0, 1], fs, f"P2.1 fine {name}", n)
        raw["fine"][name] = {"transition": t, "step": fs[1] - fs[0], "launches": ls}
    raw["pair"] = {
        str(f): sweep(r, b, ["pair", p["slot"], nl, reps, f], p["pair_d"], f"P2.1 pair {f}", n)
        for f, nl in p["pair"]
    }
    raw["conflict"] = {
        str(s): sweep(
            r, b, ["conflict", s, p["conflict_nloads"], reps, 0], p["conflict_k"], f"P2.1 conflict {s}", n
        )
        for s in p["conflict_s"]
    }
    return raw, analyze_caches(raw)


def analyze_caches(raw):
    p, rows = raw["plan"], []
    for name, unit, k, pubpath in (
        ("l1d", "KiB", KB, "memory.l1d.size_kib"),
        ("l2", "MiB", MB, "memory.l2.size_mib"),
    ):
        fi = raw["fine"].get(name)
        if fi is None:
            rows.append(row(f"{name.upper()} capacity", None, None, unit, pubpath, ["no transition"]))
            continue
        _, _, lo, hi = fi["transition"]
        ls = fi["launches"]
        v, ci = onset(pooled(ls, "lat"), each(ls, "lat"), lo, hi, 64)
        v, ci = scale(v, ci, k)
        note = f"random chain, 64 B nodes; levels {lo:.1f} -> {hi:.1f} cycles; fine step {fi['step']} B"
        rows.append(row(f"{name.upper()} capacity", v, ci, unit, pubpath, [] if v else ["no onset"], note))
    co = raw["coarse"]
    for name, key, pubpath in (
        ("L1D load-to-use", "l1d", "memory.l1d.latency_cycles"),
        ("L2 load-to-use", "l2", "memory.l2.latency_cycles"),
        ("DRAM load-to-use", "dram", "memory.dram_latency"),
    ):
        lo, hi = p["lat"][key]
        v, ci = region(co, "lat", lo, hi)
        rows.append(row(name, v, ci, "cycles", pubpath, note=f"random chain over {size(lo)} to {size(hi)}"))
    for (f, _), level, pubpath in zip(
        p["pair"], ("L1D", "L2"), ("memory.l1d.line_bytes", "memory.l2.line_bytes")
    ):
        ls = raw["pair"][str(f)]
        xs, ys = pooled(ls, "pair")
        v = mem_cache.line_size(xs, ys) if len(xs) > 1 else None
        per = [mem_cache.line_size(*c) for c in each(ls, "pair") if len(c[0]) > 1]
        flags = ["no step"] if v is None else ["launches disagree"] if len(set(per)) > 1 else []
        note = f"pair chase over {size(f)}: first D past half the step from D = {xs[0] if xs else '?'} B"
        rows.append(row(f"{level} line", v, spread(per), "B", pubpath, flags, note))
    reads = {}
    for s in p["conflict_s"]:
        ls = raw["conflict"][str(s)]
        xs, ys = pooled(ls, "conflict")
        reads[s] = (
            mem_cache.assoc_onset(xs, ys),
            spread(mem_cache.assoc_onset(*c) for c in each(ls, "conflict")),
        )
    s = max(reads)
    flags = ["stride-dependent"] if len({v for v, _ in reads.values()}) > 1 else []
    note = "K nodes S apart: " + ", ".join(f"S = {k // KB} KiB -> {v}" for k, (v, _) in sorted(reads.items()))
    rows.append(row("L1D associativity", reads[s][0], reads[s][1], "ways", "memory.l1d.ways", flags, note))
    return rows


def run_tlbs(r, args):
    p, n = plan(args.gem5_smoke)["P2.2"], launches(args)
    b = r.compile(M.chase_source("aarch64", r.counter), "memprobe")
    reps = p["reps"]

    def page(ps, label):
        return {
            "page": sweep(r, b, ["page", p["nloads"], reps, 0, 0], ps, f"P2.2 {label} page", n),
            "dense": sweep(r, b, ["page", p["nloads"], reps, 0, 1], ps, f"P2.2 {label} dense", n),
        }

    def code(ps, label):
        return {
            "code": sweep(r, b, ["code", p["njumps"], reps], ps, f"P2.2 {label} code", n),
            "dense": sweep(r, b, ["code", p["njumps"], reps, "d=1"], ps, f"P2.2 {label} code dense", n),
        }

    raw = {"plan": p, "data": page(p["page"], "coarse"), "code": code(p["code"], "coarse")}
    for kind, run in (("data", page), ("code", code)):
        raw[f"{kind}_fine"] = []
        for i, t in enumerate(tlb_rises(raw[kind])[:2]):
            st = max(1, t[0] // 64)
            ps = thin(list(range(max(2, t[0] - 2 * st), t[1] + st, st)), p["fine_cap"])
            raw[f"{kind}_fine"].append({"rise": t, "step": st, **run(ps, f"fine {kind} {i}")})
    return raw, analyze_tlbs(raw)


def tlb_curves(sw):
    """Page-spread minus packed chain (cycles per load or jump the pages add): pooled, per launch."""
    a, ta, tb = ("page", "page", "dense") if "page" in sw else ("code", "code", "code")
    pool = diff(pooled(sw[a], ta), pooled(sw["dense"], tb))
    return pool, [diff(x, y) for x, y in zip(each(sw[a], ta), each(sw["dense"], tb))]


def tlb_rises(sw):
    return steps(*tlb_curves(sw)[0])


def analyze_tlbs(raw):
    rows = []
    pb = page_bytes(raw["data"]["page"]) or page_bytes(raw["code"]["code"])
    names = {
        "data": [("L1 dTLB", "memory.dtlb_l1"), ("L2 TLB", "memory.dtlb_l2")],
        "code": [("L1 iTLB", None), ("L2 TLB, instruction side", None)],
    }
    for kind in ("data", "code"):
        rises = tlb_rises(raw[kind])
        for i, (name, pubpath) in enumerate(names[kind]):
            ent = pubpath and pubpath + ".entries"
            fi = raw[f"{kind}_fine"][i] if i < len(raw[f"{kind}_fine"]) else None
            if fi is None:
                if kind == "data" or i == 0:
                    rows.append(row(f"{name} entries", None, None, "pages", ent, ["no rise"]))
                continue
            _, _, lo, hi = fi["rise"]
            v, ci = onset(*tlb_curves(fi), lo, hi, 1)
            reach = f"{v * pb / MB:.3g} MiB" if v and pb else "?"
            note = f"page minus dense chain, {pb} B pages: reach {reach}; +{hi - lo:.1f} cycles past it"
            rows.append(row(f"{name} entries", v, ci, "pages", ent, [], note))
            if kind == "data":
                pen = pubpath + ".miss_penalty_cycles"
                rows.append(
                    row(
                        f"{name} miss penalty",
                        hi - lo,
                        None,
                        "cycles",
                        pen,
                        note="rise of the page-minus-dense curve",
                    )
                )
        if len(rises) > 2:
            rows[-1]["note"] += "; more rises at " + ", ".join(f"{t[0]}-{t[1]}" for t in rises[2:])
    return rows


def run_mlp(r, args):
    p, n = plan(args.gem5_smoke)["P2.3"], launches(args)
    nodes = p["F"] // 128
    if (p["reps"] + 1) * p["iters"] >= nodes // max(p["K"]):
        raise ValueError("P2.3: chains would wrap; raise F or lower iters")
    b = r.compile(M.chase_source("aarch64", r.counter), "memprobe")
    x = [f"x={p['x']}"] if p["x"] else []
    ks = list(p["K"])
    random.Random(0).shuffle(ks)
    # one launch per K, seed = launch index: every K's chains start at the same nodes, so a K run
    # in the same process would find the previous K's lines still cached
    head = lambda i: ["mlp", p["F"], p["iters"], p["reps"], i, *x]  # noqa: E731
    raw = {"plan": p, "launches": merge([sweep(r, b, head, [k], f"P2.3 K={k}", n) for k in ks])}
    return raw, analyze_mlp(raw)


def analyze_mlp(raw):
    reads = []
    for ks, ts in each(raw["launches"], "mlp"):
        if len(ks) >= 5:
            reads.append({**mem_mlp.reading(ks, ts), "t1": ts[0]})
    if not reads:
        return [row("misses in flight", None, None, "chains", "memory.mlp", ["no data"])]
    v = statistics.median(x["reading"] for x in reads)
    why = sorted({w for x in reads for w in x["why"]})
    t1, knee = (statistics.median(x[k] for x in reads) for k in ("t1", "knee"))
    note = f"K random chains over {size(raw['plan']['F'])}; T(1) {t1:.0f} cycles; hinge knee {knee:.0f}"
    note += "".join(f"; {w}" for w in why)
    flags = ["lower bound"] if why else []
    return [
        row("misses in flight", v, spread(x["reading"] for x in reads), "chains", "memory.mlp", flags, note)
    ]


def run_prefetch(r, args):
    p, n = plan(args.gem5_smoke)["P2.4"], launches(args)
    b = r.compile(M.chase_source("aarch64", r.counter), "memprobe")
    f, nl, reps = p["F"], p["nloads"], p["reps"]
    raw = {"plan": p, "rand": sweep(r, b, ["lat", "rand", 128, nl, reps, 0, 1], [f], "P2.4 rand", n)}
    raw["seq"] = {
        str(s): sweep(r, b, ["lat", "seq", s, nl, reps, 0, 1], [f], f"P2.4 seq {s}", n) for s in p["seq"]
    }
    raw["indep"] = sweep(r, b, ["indep", 128, p["indep_nloads"], reps, 0], [f], "P2.4 indep", n)
    raw["wchain"] = {
        str(w): sweep(r, b, ["wchain", w, p["wchain_nloads"], reps], [], f"P2.4 wchain {w}", n)
        for w in p["wchain"]
    }
    return raw, analyze_prefetch(raw)


def analyze_prefetch(raw):
    p, f, rows = raw["plan"], raw["plan"]["F"], []
    dram, dci = region(raw["rand"], "lat", f, f)
    rows.append(
        row(
            "DRAM latency",
            dram,
            dci,
            "cycles",
            "memory.dram_latency",
            note=f"random chain, 128 B nodes, {size(f)}",
        )
    )
    for w in p["wchain"]:
        ls = raw["wchain"][str(w)]
        a, aci = region(ls, "wchainA", w, w)
        ab, _ = region(ls, "wchainAB", w, w)
        ratio = a / dram if a and dram else None
        flags = [] if ratio is None or abs(ratio - 1) <= DRAM_TOL else ["not a DRAM miss"]
        note = f"chain A {a and round(a, 1)} cycles per load" + (
            f"; A+B {ab / a:.2f} x A per pair" if a and ab else ""
        )
        if w != f:
            note += f"; against the {size(f)} random chain"
        ci = scale(None, aci, dram)[1]
        rows.append(row(f"window chains, {size(w)}: latency / DRAM", ratio, ci, "x", None, flags, note))
    best = None
    for s in p["seq"]:
        v, ci = region(raw["seq"][str(s)], "lat", f, f)
        ratio = v / dram if v and dram else None
        hit = ratio is not None and ratio < PREFETCHED
        best = s if hit else best
        note = f"{v and round(v, 1)} cycles per load"
        rows.append(
            row(
                f"dependent {s} B-stride chain / random",
                ratio,
                scale(None, ci, dram)[1],
                "x",
                None,
                ["prefetched"] if hit else [],
                note,
            )
        )
    v, ci = region(raw["indep"], "indep", f, f)
    rows.append(row("independent 128 B-stride loads", v, ci, "cycles", note="cycles per load, summed"))
    note = f"under {PREFETCHED} x the random chain"
    rows.append(
        row(
            "stride prefetcher: largest dependent stride caught",
            best,
            None,
            "B",
            "memory.prefetch",
            note=note,
        )
    )
    return rows


def run_memdep(r, args):
    p, n = plan(args.gem5_smoke)["P2.5"], launches(args)
    if max(p["N"] + p["control"]) > MD_NMAX:
        raise ValueError(f"P2.5: N up to {MD_NMAX}")

    def blocks(ns, alias, label):
        b = r.compile(M.memdep_source(ns, alias, "aarch64", r.counter), f"memdep_{'a' if alias else 'n'}")
        octaves = defaultdict(list)
        for x in ns:
            octaves[x.bit_length()].append(x)
        return b, merge(
            [
                sweep(
                    r, b, ["memdep", max(1, p["blocks"] // min(g)), p["reps"]], g, f"P2.5 {label} {min(g)}", n
                )
                for g in octaves.values()
            ]
        )

    b, coarse = blocks(p["N"], True, "alias")
    raw = {"plan": p, "coarse": coarse, "control": blocks(p["control"], False, "control")[1]}
    raw["fwd"] = sweep(r, b, ["fwd", p["fwd"], p["reps"]], [], "P2.5 fwd", n)
    br = memdep_bracket(raw)
    if br:  # from one coarse point below: a gradual rise crosses 10 % well past its knee
        below = [x for x in pooled(coarse, "memdep")[0] if x < br[0]]
        ns = thin(list(range(max(2, below[-1] if below else br[0]), br[1] + 2)), p["fine_cap"])
        raw["fine"] = {"bracket": br, "launches": blocks(ns, True, "fine")[1]}
    return raw, analyze_memdep(raw)


def memdep_bracket(raw, min_rise=1.25):
    """mem_memdep.bracket on the pooled coarse curve, if the cost rises at least min_rise x."""
    ns, ys = pooled(raw["coarse"], "memdep")
    br = mem_memdep.bracket(ns, ys) if len(ns) > 3 else None
    return br if br and br[3] >= min_rise * br[2] else None


def analyze_memdep(raw):
    rows, br = [], memdep_bracket(raw)
    ns, ys = pooled(raw["coarse"], "memdep")
    _, cy = pooled(raw["control"], "memdep")
    flags = ["control rises"] if cy and max(cy) > 1.25 * min(cy[:3]) else []
    fi = raw.get("fine")
    if br and fi:
        fx, fy = pooled(fi["launches"], "memdep")
        c = mem_memdep.change_point(fx, fy) if len(fx) > 2 else None
        per = [mem_memdep.change_point(*cv) for cv in each(fi["launches"], "memdep") if len(cv[0]) > 2]
        above = [y for x, y in zip(ns, ys) if c and x >= 2 * c]
        note = (
            f"N store/load blocks, store address behind a divide; {br[2]:.1f} -> {br[3]:.1f} cycles per block"
        )
        rows.append(row("store-load pairs tracked", c, spread(per), "pairs", None, flags, note))
        cost = statistics.median(above) - br[2] if above else None
        rows.append(
            row(
                "ordering violation cost per block",
                cost,
                None,
                "cycles",
                note="above 2x the knee, over the flat level",
            )
        )
    else:
        top = max(ns) if ns else None
        rows.append(
            row("store-load pairs tracked", None, None, "pairs", None, flags + ["no rise"], f"at least {top}")
        )
    v, ci = region(raw["fwd"], "fwd", 0, 0)
    rows.append(
        row("store-to-load forwarding round trip", v, ci, "cycles", note="str; ldr through one address")
    )
    return rows


POCS = {"P2.1": run_caches, "P2.2": run_tlbs, "P2.3": run_mlp, "P2.4": run_prefetch, "P2.5": run_memdep}
EST_MIN = {"P2.1": 5, "P2.2": 2, "P2.3": 3, "P2.4": 1, "P2.5": 1}


def launches(args):
    return 1 if args.gem5_smoke else getattr(args, "launches", 3)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    native.add_args(ap)
    ap.add_argument("pocs", nargs="*", metavar="PoC", help=f"any of {' '.join(POCS)} (default all)")
    ap.add_argument("--launches", type=int, default=3, help="launches per sweep (1 under --gem5-smoke)")
    return ap


def show(poc, rows):
    for x in rows:
        ci = f"[{x['ci'][0]:g}, {x['ci'][1]:g}]" if x["ci"] else "-"
        v = "-" if x["value"] is None else f"{x['value']:g}"
        pub = f"published {x['published']}" if x["published"] is not None else ""
        print(f"  {poc} {x['structure']:<46} {v:>9} {ci:>18} {x['unit']:<7} {pub} {' '.join(x['flags'])}")


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    want = args.pocs or list(POCS)
    bad = [p for p in want if p not in POCS]
    if bad:
        ap.error(f"unknown PoC {' '.join(bad)}; choose from {' '.join(POCS)}")
    r = native.Runner(args)
    print(
        f"{FAMILY}: {' '.join(want)} (~{sum(EST_MIN[p] for p in want)} min on an M1), counter {r.counter}",
        flush=True,
    )
    for poc in want:
        t0 = time.time()
        raw, rows = POCS[poc](r, args)
        raw["elapsed_s"] = round(time.time() - t0, 1)
        ok, tried = sum(a["accepted"] for a in r.attempts), len(r.attempts)
        d = r.save(FAMILY, poc, raw, rows)
        print(f"{poc}: {raw['elapsed_s']} s, {ok}/{tried} launches accepted -> {os.path.relpath(d)}")
        show(poc, rows)


if __name__ == "__main__":
    main()
