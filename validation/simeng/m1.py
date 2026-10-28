#!/usr/bin/env python3
"""A targets/m1 PoC under SimEng: m1.py CONFIG.yaml FAMILY POC [runner args...]

The gem5-smoke path (static binaries, PMU cycle counter) with SimEng in place of gem5; P1.5 with the
M1 grids and fewer iterations. Writes results/simeng/<config>/m1/<family>/<PoC>/.
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "targets", "m1")]
import native  # noqa: E402

SIM = os.environ.get(
    "SIMENG", f"/var/tmp/{os.environ.get('USER')}/uarch-re/simeng/build/src/tools/simeng/simeng"
)
ITERS = {"sched": 60, "dispatch": 200, "drain": 200, "stream": 200, "elim": 200}

cfg_path, family, poc, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
mod = __import__(f"m1_{family}")


def _exec(self, binpath, argv, timeout):
    cmd = [SIM, cfg_path, binpath, *map(str, argv)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout


native.Runner._exec = _exec
native.RESULTS = os.path.join(ROOT, "results", "simeng", os.path.basename(cfg_path).removesuffix(".yaml"))
if family == "sched":
    m = mod
    mod.SMOKE = {
        "sizes": m.SIZES,
        "pairs": m.PAIRS,
        "dispatch": m.DISPATCH,
        "dispatch_m": m.DISPATCH_M,
        "dispatch_grid": m.DISPATCH_GRID,
        "drain_k": m.DRAIN_K,
        "drain": m.DRAIN,
        "stream_k": m.STREAM_K,
        "elim": m.ELIM,
        "elim_k": m.ELIM_K,
        "iters": ITERS,
        "reps": 2,
        "outer_reps": 2,
        "chain": 1 << 20,
        "fine": True,
        "jobs": 8,
    }
mod.main([poc, "--gem5-smoke", "--tag", "m1", *rest])
