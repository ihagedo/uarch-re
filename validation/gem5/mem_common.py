"""Shared runner for the P2 memory campaigns (validation/gem5/mem_*.py).

A campaign is a list of jobs {key, argv, params, sets}. `run_jobs` runs them
concurrently, stores each finished job (per-rep output, outdir, requested gem5
stats per m5 dump) in one JSON file, and skips jobs already there.
"""

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.kernels import memory as M

SCRATCH = os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER', 'u')}/uarch-re")
LOGS = os.path.join(SCRATCH, "logs", "memory")
RESULTS = os.path.join(ROOT, "results", "gem5")

STATS = [
    "system.cpu.numCycles",
    "simInsts",
    "system.cpu.dcache.demandMisses::total",
    "system.cpu.dcache.demandAccesses::total",
    "system.l2.demandMisses::total",
    "system.l2.demandAccesses::total",
    "system.l2.demandMisses::cpu.data",
    "system.l2.demandAccesses::cpu.data",
    "system.cpu.dcache.demandMisses::cpu.data",
    "system.cpu.dcache.demandAccesses::cpu.data",
    "system.mem_ctrl.readReqs",
    "system.cpu.dcache.prefetcher.pfIssued",
    "system.l2.prefetcher.pfIssued",
]


def log(name, msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, f"{name}.log"), "a") as fh:
        fh.write(line + "\n")


FS_STATS = [
    "system.cpu.numCycles",
    "simInsts",
    "system.cpu.mmu.dtb.readHits",
    "system.cpu.mmu.dtb.readMisses",
    "system.cpu.mmu.dtb.walker.num_4kb_walks",
    "system.cpu.dcache.demandMisses::total",
    "system.cpu.mmu.itb.readMisses",
    "system.cpu.mmu.itb.misses",
    "system.cpu.mmu.itb.hits",
]


def _run_fs(job):
    """Bare-metal full-system TLB job: dump i of stats.txt is P = plist[i]."""
    elf = M.tlb_baremetal_binary(job["plist"], job["nloads"], job["reps"], job.get("itlb", False))
    out, outdir = M.run_gem5_fs(elf, job.get("sets", ()), tag="tlbfs")
    dumps = M.gem5_stats_dumps(outdir)
    if len(dumps) < len(job["plist"]):
        raise RuntimeError(f"{len(dumps)} dumps for {len(job['plist'])} P values:\n{out[-2000:]}")
    n = job["nloads"] * job["reps"]
    runs = {"tlb": {str(p): [d["system.cpu.numCycles"] / n] for p, d in zip(job["plist"], dumps)}}
    stats = [{k: d.get(k) for k in FS_STATS} for d in dumps[: len(job["plist"])]]
    return {
        **job,
        "runs": runs,
        "scalars": {},
        "outdir": outdir,
        "stats": stats,
        "exit": out.strip()[-200:],
        "config_check": check_config(outdir, job.get("sets", ()), {}),
    }


def _memdep_bin(job):
    return M.memdep_binary(job["ns"], job.get("alias", True))


def _run_one(job):
    if job.get("fs"):
        return _run_fs(job)
    if job.get("probe") == "memdep":
        from uarchre.backends.platform.gem5 import run_gem5_binary

        out, outdir = run_gem5_binary(
            _memdep_bin(job),
            [str(a) for a in job["argv"]],
            job.get("params") or {},
            list(job.get("sets", ())),
        )
        runs, sc = M.parse_runs(out)
        if not runs:
            raise RuntimeError(f"no results:\n{out[-2000:]}")
    else:
        runs, sc, outdir = M.run_gem5(job["argv"], job.get("params"), job.get("sets", ()))
    rec = {
        **job,
        "runs": {t: {str(k): v for k, v in d.items()} for t, d in runs.items()},
        "scalars": sc,
        "outdir": outdir,
    }
    try:
        dumps = M.gem5_stats_dumps(outdir)
        rec["stats"] = [{k: d.get(k) for k in STATS} for d in dumps]
    except OSError:
        rec["stats"] = []
    # confirm every --set override reached config.ini
    rec["config_check"] = check_config(outdir, job.get("sets", ()), job.get("params", {}))
    return rec


def check_config(outdir, sets, params):
    """For each `a.b.c=value` set, report the value config.ini holds for
    [system.a.b] c (constructor values: the section's `type`)."""
    import configparser

    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(os.path.join(outdir, "config.ini"))
    out = {}
    for s in sets:
        path, val = s.split("=", 1)
        *parents, leaf = path.split(".")
        sec = ".".join(["system"] + parents)
        if "(" in val:  # an object: its section's type
            sec2 = sec + "." + leaf
            out[s] = cp.get(sec2, "type", fallback=None)
        else:
            out[s] = cp.get(sec, leaf, fallback=None)
    for k in params or {}:
        out[k] = cp.get("system.cpu", k, fallback=None)
    return out


def run_jobs(name, jobs, out_json, workers=16):
    """Run jobs not yet in out_json; returns {key: record}."""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    done = {}
    if os.path.exists(out_json):
        with open(out_json) as fh:
            done = json.load(fh)
    todo = [j for j in jobs if j["key"] not in done]
    log(name, f"{len(jobs)} jobs, {len(todo)} to run -> {out_json}")
    # compile before the threads start (concurrent compiles of one source
    # would share a temporary file name within this process)
    for j in todo:
        if j.get("fs"):
            M.tlb_baremetal_binary(j["plist"], j["nloads"], j["reps"], j.get("itlb", False))
        elif j.get("probe") == "memdep":
            _memdep_bin(j)
        else:
            M.binary("riscv64")
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, j): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                rec = f.result()
            except Exception:
                log(name, f"FAILED {j['key']}: {traceback.format_exc()[-1500:]}")
                continue
            with lock:
                done[j["key"]] = rec
                tmp = out_json + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(done, fh, indent=1)
                os.replace(tmp, out_json)
            log(name, f"done {j['key']} ({time.time() - t0:.0f} s) {rec['config_check']}")
    return done


def load(out_json):
    with open(out_json) as fh:
        return json.load(fh)


def med(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
