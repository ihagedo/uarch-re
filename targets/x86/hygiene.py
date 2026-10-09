"""Measurement hygiene for native x86 runs: waits for quiet CPUs, monitored launches
(preemption, context switches, SMT-sibling load) and level-drift checks.
"""

import os
import subprocess
import time

import numpy as np


def siblings(cpu):
    """Other logical CPUs of the same physical core as `cpu`."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as fh:
            txt = fh.read().strip()
    except OSError:
        return []
    out = set()
    for part in txt.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out - {int(cpu)})


def cpu_times(cpus):
    """{cpu: (busy_jiffies, total_jiffies)} from /proc/stat."""
    want = {f"cpu{c}": c for c in cpus}
    res = {}
    with open("/proc/stat") as fh:
        for line in fh:
            f = line.split()
            if f and f[0] in want:
                v = [int(x) for x in f[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                res[want[f[0]]] = (sum(v) - idle, sum(v))
    return res


def busy_fraction(before, after):
    out = {}
    for c in before:
        db = after[c][0] - before[c][0]
        dt = after[c][1] - before[c][1]
        out[c] = db / dt if dt > 0 else 0.0
    return out


def wait_quiet(cpus, max_busy, window=0.15, timeout=20.0):
    """Block until every CPU in `cpus` was at most `max_busy` busy over a
    `window`-second sample, or `timeout` passes. Returns seconds waited."""
    t0 = time.perf_counter()
    while True:
        b0 = cpu_times(cpus)
        time.sleep(window)
        busy = busy_fraction(b0, cpu_times(cpus))
        if max(busy.values()) <= max_busy or time.perf_counter() - t0 > timeout:
            return time.perf_counter() - t0 - window


def run_monitored(cmd, watch=(), trace=False):
    """Run `cmd`; return (stdout, info). info: wall_s, cpu_s, share (cpu/wall),
    nivcsw, busy {cpu: fraction}, and with `trace` the CPUs it was seen on."""
    b0 = cpu_times(watch) if watch else {}
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seen = []
    if trace:
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            try:
                with open(f"/proc/{p.pid}/stat") as fh:
                    c = int(fh.read().rsplit(")", 1)[1].split()[36])
                if not seen or seen[-1] != c:
                    seen.append(c)
            except (OSError, IndexError, ValueError):
                pass
            time.sleep(0.004)
    else:
        _, status, ru = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0
    out, err = p.stdout.read(), p.stderr.read()
    p.returncode = os.waitstatus_to_exitcode(status)
    if p.returncode:
        raise RuntimeError(f"{cmd} failed ({p.returncode}): {err[-500:]}")
    b1 = cpu_times(watch) if watch else {}
    cpu = ru.ru_utime + ru.ru_stime
    info = {
        "wall_s": round(wall, 4),
        "cpu_s": round(cpu, 4),
        "share": round(cpu / wall, 3) if wall else 1.0,
        "nivcsw": ru.ru_nivcsw,
        "busy": {c: round(v, 3) for c, v in busy_fraction(b0, b1).items()},
    }
    if trace:
        info["cpus_seen"] = seen
    return out, info


# with hundreds of points per outer rep, (b) and (c) detect 1 % shifts (DRAM latency
# moving with host load) that leave every knee unchanged, so they also need `effect`
A8 = {"ratio": 0.02, "kw_p": 0.01, "rho": 0.3, "effect": 0.03}


def residuals(rows):
    """Leave-one-out residuals: value / median(other outer reps at the same key and N) - 1.

    rows: dicts with key, n, outer_rep, order_index, value. Returns (outer_rep, order_index,
    residual) arrays; points measured in one rep only are dropped."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["key"], r["n"]), []).append(i)
    keep, res = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [rows[j]["value"] for j in idx if j != i]
            keep.append(i)
            res.append(rows[i]["value"] / float(np.median(others)) - 1)
    rep = np.array([rows[i]["outer_rep"] for i in keep])
    order = np.array([rows[i]["order_index"] for i in keep])
    return rep, order, np.array(res)


def drift_rules(rows, ratio_pairs=None, th=A8):
    """Level-drift checks on one campaign's per-launch medians: (a) clock-ratio change within a launch,
    (b) Kruskal-Wallis across outer reps, (c) residual trend over run order. A window probe's knee
    ignores level drift, so anomaly.py's A8 compares per-rep knees instead.
    Returns {fires, a, b, c} with the evidence."""
    from scipy import stats

    out = {}
    if ratio_pairs:
        ch = np.array([abs(post / pre - 1) for pre, post in ratio_pairs if pre and post])
        out["a"] = {
            "share_over": float((ch > th["ratio"]).mean()) if len(ch) else 0.0,
            "median_change": float(np.median(ch)) if len(ch) else 0.0,
            "fires": bool(len(ch) and np.median(ch) > th["ratio"]),
        }
    rep, order, res = residuals(rows)
    groups = [res[rep == k] for k in sorted(set(rep))]
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        h, p = stats.kruskal(*groups)
        meds = [float(np.median(g)) for g in groups]
        effect = max(abs(m) for m in meds)
        out["b"] = {
            "H": float(h),
            "p": float(p),
            "rep_median_residual": meds,
            "effect": effect,
            "fires_p_only": bool(p < th["kw_p"]),
            "fires": bool(p < th["kw_p"] and effect > th["effect"]),
        }
    rho, p = stats.spearmanr(order, res)
    x = (order - order.min()) / max(1, order.max() - order.min())
    drift = float(stats.theilslopes(res, x)[0]) if len(res) > 2 else 0.0  # change over the whole run
    out["c"] = {
        "rho": float(rho),
        "p": float(p),
        "drift_over_run": drift,
        "fires_rho_only": bool(abs(rho) > th["rho"]),
        "fires": bool(p < th["kw_p"] and abs(drift) > th["effect"]),
    }
    out["fires"] = any(v.get("fires") for k, v in out.items() if k in ("b", "c"))
    return out


def inject(rows, kind, size=0.05, rep=1):
    """Copy of rows with synthetic drift: 'step' scales outer rep `rep` by 1 + size,
    'ramp' scales each launch by 1 + size * order / max order."""
    mx = max(r["order_index"] for r in rows) or 1
    out = []
    for r in rows:
        f = 1.0
        if kind == "step" and r["outer_rep"] == rep:
            f = 1 + size
        elif kind == "ramp":
            f = 1 + size * r["order_index"] / mx
        out.append({**r, "value": r["value"] * f})
    return out
