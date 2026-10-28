#!/usr/bin/env python3
"""targets/m1/m1_sweep.py under SimEng, with the PMU cycle counter (SimEng maps no CNTFRQ_EL0):
sweep.py CONFIG.yaml TAG [m1_sweep args...]. Writes results/simeng/_sweep/results/m1/<TAG>/.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "targets", "m1")]
import m1_sweep  # noqa: E402
from uarchre.backends.isa.aarch64 import AArch64  # noqa: E402

SIM = os.environ.get(
    "SIMENG", f"/var/tmp/{os.environ.get('USER')}/uarch-re/simeng/build/src/tools/simeng/simeng"
)
cfg, tag, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
m1_sweep.AArch64 = lambda counter: AArch64("pmccntr")
m1_sweep.ROOT = os.path.join(ROOT, "results", "simeng", "_sweep")  # outdir = ROOT/results/m1/TAG
sys.argv = [
    "m1_sweep.py", "--tag", tag, "--static", "--cc", "aarch64-linux-gnu-gcc", "--exec-prefix", f"{SIM} {cfg}",
    "--force", "--no-plot", "--retries", "0", "--min-ghz", "0", *rest,
]  # fmt: skip
m1_sweep.main()
