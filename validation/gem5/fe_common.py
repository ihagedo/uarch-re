"""Shared runner for the frontend campaigns (validation/gem5/fe_*.py).

run_fe(kernel, ...) compiles a FEKernel's harness for RISC-V, runs it under
gem5 O3 (validation/gem5/config_o3.py), drops the warm-up rep and returns the
cycles per unit for every measured rep, plus the outdir for stats/config.ini.
"""

import csv
import hashlib
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.backends.platform.gem5 import compile_c, run_gem5_binary, stat
from uarchre.kernels.frontend import harness_c, parse_reps

RESULTS = os.path.join(ROOT, "results", "gem5")
CACHE = os.path.join(RESULTS, "fe-cache")  # stdout of every finished run, keyed by binary + args
LOGS = os.path.join(
    os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER', 'u')}/uarch-re"),
    "logs",
    "frontend-core",
)


# compile_source names its temp file by PID only, so threads compiling the same
# kernel collide; serialize compiles (they are cached)
_COMPILE = threading.Lock()
# cap this process's concurrent gem5 runs so the shared host slots stay fair
GATE = threading.BoundedSemaphore(int(os.environ.get("FE_JOBS", "8")))


def build(k):
    with _COMPILE:
        return compile_c(harness_c(k), k.name)


WATCHDOG_S = float(os.environ.get("FE_WATCHDOG_S", "900"))


def _watchdog():
    """Kill this process's gem5 children older than WATCHDOG_S; a gem5 livelock
    would otherwise block a campaign forever (run_fe then raises)."""
    import signal
    import time

    me = os.getpid()
    hz = os.sysconf("SC_CLK_TCK")
    while True:
        time.sleep(10)
        try:
            up = float(open("/proc/uptime").read().split()[0])
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                    cmd = open(f"/proc/{pid}/cmdline").read()
                except OSError:
                    continue
                if "gem5.opt" not in cmd:
                    continue
                ppid = int(st[1])
                try:  # gem5 runs under taskset, whose parent is this process
                    pp = int(open(f"/proc/{ppid}/stat").read().rsplit(")", 1)[1].split()[1])
                except OSError:
                    pp = -1
                if me not in (ppid, pp):
                    continue
                age = up - int(st[19]) / hz
                if age > WATCHDOG_S:
                    print(f"watchdog: killing gem5 pid {pid} after {age:.0f} s: {cmd[-300:]!r}", flush=True)
                    os.kill(int(pid), signal.SIGKILL)
        except Exception as e:  # never let the watchdog die
            print("watchdog error", e, flush=True)


threading.Thread(target=_watchdog, daemon=True).start()


def run_fe(k, laps=1, reps=3, seed=0, mode=0, params=None, sets=(), config_args=(), stats=()):
    """Returns dict(cpu=[cycles/unit per measured rep], ipc=[...], outdir, stats={name: value})."""
    b = build(k)
    argv = [laps, reps, seed, mode]
    key = hashlib.md5(
        json.dumps(
            [os.path.basename(b), argv, params or {}, list(sets), list(config_args)], sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    cpath = os.path.join(CACHE, f"{k.name}_{key}.json")
    hit = None
    if os.path.exists(cpath):
        with open(cpath) as fh:
            hit = json.load(fh)
        if not os.path.exists(os.path.join(hit["outdir"], "stats.txt")):
            hit = None
    if hit:
        out, od = hit["stdout"], hit["outdir"]
    else:
        with GATE:
            out, od = run_gem5_binary(
                b, argv, params=params or {}, sets=list(sets), config_args=list(config_args)
            )
        os.makedirs(CACHE, exist_ok=True)
        with open(cpath + ".tmp", "w") as fh:
            json.dump({"stdout": out, "outdir": od}, fh)
        os.replace(cpath + ".tmp", cpath)
    rr = parse_reps(out)
    if len(rr) != reps:
        raise RuntimeError(f"{k.name}: expected {reps} reps, got:\n{out[-1500:]}")
    meas = rr[1:] if reps > 1 else rr
    units = k.units_per_lap * laps
    return {
        "cpu": [c / units for c, _ in meas],
        "ipc": [i / c for c, i in meas],
        "cycles": [c for c, _ in meas],
        "instret": [i for _, i in meas],
        "outdir": od,
        "stats": {s: stat(od, s) for s in stats},
    }


def config_value(outdir, section, key):
    """Read one key of one section of config.ini (to confirm an override took)."""
    sec = None
    with open(os.path.join(outdir, "config.ini")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                sec = line[1:-1]
            elif sec == section and line.startswith(key + "="):
                return line.split("=", 1)[1]
    return None


def pmap(fn, items, jobs=16, log=None):
    """Run fn over items concurrently (gem5 slots queue them); results in order.
    `log(i, result)` is called as each finishes."""
    from concurrent.futures import as_completed

    items = list(items)
    res = [None] * len(items)
    with ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            i = futs[f]
            res[i] = f.result()
            if log:
                log(i, res[i])
    return res


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def med(xs):
    return statistics.median(xs)


def plateau(ns, y, tau, start=3):
    """Leading in-capacity points: the first `start` points, extended while the
    next point's lap excess over the line a + b*n fitted to total cycles per lap
    of the points so far is <= tau."""
    import numpy as np

    pts = sorted(zip(ns, y))
    base = [n for n, _ in pts[:start]]
    for n, v in pts[start:]:
        bn = np.array(base, float)
        T = bn * np.array([dict(pts)[m] for m in base])
        b, a = np.polyfit(bn, T, 1)
        if n * v - (a + b * n) > tau:
            break
        base.append(n)
    return base


def capacity_search(measure, coarse, step=1, kary=12, log=print, tag=""):
    """Capacity onset of a set-associative structure; measure(n) -> cycles per item.

    tau is half the per-item miss cost on the coarse grid; the bracket [last n with
    lap excess <= tau, first n above] is split into kary+1 parts until `step` wide."""
    from uarchre.kernels.frontend import lap_excess, onset

    pts = dict(zip(coarse, pmap(measure, coarse)))
    ns = sorted(pts)
    y = [pts[n] for n in ns]
    tau = 0.5 * (y[-1] - min(y))
    base = plateau(ns, y, tau)
    res = {"points": pts, "base": base, "tau": tau, "rounds": 1, "bracket": None, "coarse_bracket": None}
    if tau <= 0.05 * min(y):
        res["error"] = "no step: the largest n costs no more per item than the cheapest"
        return res
    if len(base) < 3:
        res["error"] = "no plateau below the transition (grid starts too high)"
        return res

    def bracket():
        ns = sorted(pts)
        return onset(ns, lap_excess(ns, [pts[n] for n in ns], base), tau)

    br = bracket()
    res["coarse_bracket"] = br
    while br and br[0] is not None and br[1] - br[0] > step:
        lo, hi = br
        cand = sorted({lo + int(round((hi - lo) * i / (kary + 1) / step)) * step for i in range(1, kary + 1)})
        cand = [c for c in cand if lo < c < hi and c not in pts]
        if not cand:
            break
        pts.update(zip(cand, pmap(measure, cand)))
        res["rounds"] += 1
        br = bracket()
        if log:
            log(f"{tag} round {res['rounds']}: bracket {br}")
    res["bracket"] = br
    if br is None:
        res["error"] = "no point exceeds tau"
    elif br[0] is None:
        res["error"] = "transition below the grid"
    ns = sorted(pts)
    res["excess"] = dict(zip(ns, [float(e) for e in lap_excess(ns, [pts[n] for n in ns], base)]))
    return res


def geom_grid(lo, hi, per_octave=4, step=1):
    """Geometric grid lo..hi, `per_octave` points per doubling, multiples of step."""
    import math

    k = int(round(math.log2(hi / lo) * per_octave))
    return sorted({int(round(lo * 2 ** (i / per_octave) / step)) * step for i in range(k + 1)})
