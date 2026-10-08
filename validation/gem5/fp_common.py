"""Shared runner for the frontend-predictor campaigns (validation/gem5/fp_*.py).

run_points(src, name, ...) compiles one harness from uarchre/kernels/predictors.py,
runs it under gem5 O3 (isa="aarch64": the ARM build) and returns ({key: [cycles per
rep]}, outdir). Runs with `fdp_args` go through fp_config.py, which attaches a
FetchDirectedPrefetcher.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.backends.platform import gem5 as g5
from uarchre.kernels.predictors import parse_points

RESULTS = os.path.join(ROOT, "results", "gem5")
SCRATCH = os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER', 'u')}/uarch-re")
LOGS = os.path.join(SCRATCH, "logs", "frontend-pred")
FP_CONFIG = os.path.join(ROOT, "validation", "gem5", "fp_config.py")
CACHE = os.path.join(ROOT, "build", "fp_cache")
_compile_lock = threading.Lock()  # compile_source's tmp name is per process, not per thread


def _run_fdp(binpath, argv, params, sets, fdp_args, isa="riscv64"):
    """Like gem5.run_gem5_binary, through fp_config.py (FDP wiring)."""
    args = " ".join(map(str, argv))
    tail = [FP_CONFIG, *fdp_args, "--isa", g5.GEM5_ISA[isa], "--cmd", binpath, "--args", args]
    for k, v in (params or {}).items():
        tail += [f"--{k}", str(v)]
    for s in sets:
        tail += ["--set", s]
    tag = hashlib.md5(" ".join(tail).encode()).hexdigest()[:10]
    outdir = os.path.join(ROOT, "build", "gem5out", f"{os.path.basename(binpath)}_{tag}")
    os.makedirs(outdir, exist_ok=True)
    cmd = ["taskset", "-c", g5.CPUS, g5.GEM5[isa], "-q", "--outdir", outdir, *tail]
    with g5.slot():
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"gem5 failed ({r.returncode}) for {binpath}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    return r.stdout, outdir


def compile_src(src, name, isa="riscv64"):
    with _compile_lock:
        return g5.compile_c(src, name, isa)


def run_points(src, name, params=None, sets=(), argv=(), fdp_args=None, isa="riscv64"):
    """Compile + run one harness; stdout is cached by (binary, params, sets), so
    re-running a campaign only simulates what is missing."""
    b = compile_src(src, name, isa)
    key = hashlib.md5(json.dumps([b, list(argv), params or {}, list(sets), fdp_args]).encode()).hexdigest()
    cpath = os.path.join(CACHE, key + ".json")
    if os.path.exists(cpath):
        c = load_json(cpath)
        out, od = c["stdout"], c["outdir"]
    else:
        if fdp_args is None:
            out, od = g5.run_gem5_binary(b, list(argv), params=params or {}, sets=list(sets), isa=isa)
        else:
            out, od = _run_fdp(b, list(argv), params, list(sets), list(fdp_args), isa)
        save_json(cpath, {"stdout": out, "outdir": od, "binary": b, "sets": list(sets)})
    pts = parse_points(out)
    if not pts:
        raise RuntimeError(f"{name}: no points in output:\n{out[-2000:]}")
    return pts, od


def config_value(outdir, section, key):
    """One key of one section of config.ini (to confirm an override took)."""
    sec = None
    with open(os.path.join(outdir, "config.ini")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                sec = line[1:-1]
            elif sec == section and line.startswith(key + "="):
                return line.split("=", 1)[1]
    return None


def pmap(fn, items, jobs=16):
    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(fn, items))


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
    return path


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def log(tag, msg):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, f"{tag}.log"), "a") as fh:
        fh.write(msg.rstrip() + "\n")
    print(msg, flush=True)
