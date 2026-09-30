"""Two-phase shuffled sweep protocol (P0.2).

Phase 1: one outer rep over the coarse grid places a provisional knee. Phase 2: a step-1 grid around
it (pad = max(coarse step, 2 x CI width, 8)); later outer reps cover both grids. An outer rep is one
launch per point, in seeded random order with curves interleaved, so slow drift shows up as noise,
not as a false knee. `measure(key, n, outer_rep)` returns a RunResult, a float, or None to skip the
point; `jobs` > 1 runs points in threads (use 1 on real hardware).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import statistics
import time

import numpy as np

from ..backends import RunResult
from .knee import find_knee


@dataclass
class Record:
    key: str
    n: int
    outer_rep: int
    order_index: int
    t_ns: int
    result: RunResult


def _fine_grid(ns, knee, step):
    if knee.knee is None:
        return []
    width = knee.ci[1] - knee.ci[0]
    pad = int(max(step, 2 * width, 8))
    lo, hi = int(knee.ci[0]) - pad, int(knee.ci[1]) + 1 + pad
    have = set(ns)
    return [n for n in range(max(1, lo), hi + 1) if n not in have]


def run_protocol(
    measure, keys, ns, *, outer_reps=3, seed=0, min_jump=1.3, expected_steps=1, jobs=1, log=None
):
    """Returns (records, reps) with reps[key] = {n: [median per outer rep]}."""
    rng = np.random.default_rng(seed)
    records = []

    def run_batch(points, rep):
        points = [points[i] for i in rng.permutation(len(points))]
        base = len(records)

        def one(i):
            key, n = points[i]
            t = time.time_ns()
            r = measure(key, n, rep)
            if r is None:
                if log:
                    log(f"  rep {rep} {key:5s} N={n:4d}  skipped")
                return None
            r = r if isinstance(r, RunResult) else RunResult([float(r)])
            if log:
                log(f"  rep {rep} {key:5s} N={n:4d}  {statistics.median(r.cycles):8.1f}")
            return Record(key, n, rep, base + i, t, r)

        if jobs > 1:
            with ThreadPoolExecutor(jobs) as ex:
                records.extend(r for r in ex.map(one, range(len(points))) if r is not None)
        else:
            records.extend(r for r in (one(i) for i in range(len(points))) if r is not None)

    ns = sorted(ns)
    step = ns[1] - ns[0] if len(ns) > 1 else 1
    run_batch([(k, n) for k in keys for n in ns], 0)
    fine = {}
    for key in keys:
        pts = sorted((r.n, r.result.median) for r in records if r.key == key)
        prov = find_knee([p[0] for p in pts], [p[1] for p in pts], min_jump, expected_steps)
        fine[key] = _fine_grid(ns, prov, step)
        if log:
            log(f"{key}: provisional knee {prov.knee}, fine grid {fine[key][:1]}..{fine[key][-1:]}")
    run_batch([(k, n) for k in keys for n in fine[k]], 0)
    for rep in range(1, outer_reps):
        run_batch([(k, n) for k in keys for n in ns + fine[k]], rep)

    reps = {k: {} for k in keys}
    for r in sorted(records, key=lambda r: r.outer_rep):
        reps[r.key].setdefault(r.n, []).append(r.result.median)
    return records, reps
