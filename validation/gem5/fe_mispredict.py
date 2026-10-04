#!/usr/bin/env python3
"""P3.7 branch mispredict penalty on gem5 O3.

random_branches(512): 64 conditional branches on bits of a word loaded ahead (load latency
stays out of the penalty). Random data mispredicts ~half, 0xAAAA... none: penalty =
(cycles per branch, random - pattern) / 0.5. One redirect-loop delay is swept at a time;
the penalty should follow it 1:1. Time buffers are 12 entries (5 caps delays at 4).

    . tools/env.sh && python3 validation/gem5/fe_mispredict.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_common import RESULTS, run_fe, pmap, config_value, write_json, med
from uarchre.kernels.frontend import random_branches

OUT = os.path.join(RESULTS, "fe-mispredict")
WORDS = 512
SEEDS = (0, 1, 2)
COM = ["cpu.forwardComSize=12", "cpu.backComSize=12"]
SWEEPS = {
    "fetchToDecodeDelay": [1, 2, 3, 4, 6, 8],
    "decodeToRenameDelay": [1, 2, 3, 4, 6, 8],
    "renameToIEWDelay": [2, 3, 4, 6, 8],
}
# delays on the redirect loop at their defaults (BaseO3CPU.py)
LOOP = {
    "fetchToDecodeDelay": 1,
    "decodeToRenameDelay": 1,
    "renameToIEWDelay": 2,
    "issueToExecuteDelay": 1,
    "iewToCommitDelay": 1,
    "commitToFetchDelay": 1,
}
STATS = [
    "system.cpu.branchPred.mispredicted_0::DirectCond",
    "system.cpu.branchPred.committed_0::DirectCond",
    "system.cpu.numCycles",
]


def main():
    k = random_branches(WORDS)
    jobs = []
    for delay, vals in SWEEPS.items():
        for v in vals:
            for mode, seed in [(2, 0)] + [(1, s) for s in SEEDS]:
                jobs.append((delay, v, mode, seed))

    def one(j):
        delay, v, mode, seed = j
        r = run_fe(k, laps=1, reps=3, seed=seed, mode=mode, sets=COM + [f"cpu.{delay}={v}"], stats=STATS)
        assert config_value(r["outdir"], "system.cpu", delay) == str(v)
        return {
            "delay": delay,
            "value": v,
            "mode": mode,
            "seed": seed,
            "cpb": med(r["cpu"]),
            "reps": r["cpu"],
            **{s.split(".")[-1]: r["stats"][s] for s in STATS},
        }

    def log(i, row):
        print(
            f"  {row['delay']}={row['value']} mode {row['mode']} seed {row['seed']}: "
            f"{row['cpb']:.3f} cycles/branch",
            flush=True,
        )

    rows = pmap(one, jobs, log=log)
    write_json(os.path.join(OUT, "raw.json"), rows)

    summary = []
    for delay, vals in SWEEPS.items():
        for v in vals:
            pat = [r for r in rows if r["delay"] == delay and r["value"] == v and r["mode"] == 2][0]
            rnd = [r for r in rows if r["delay"] == delay and r["value"] == v and r["mode"] == 1]
            pens = [(r["cpb"] - pat["cpb"]) / 0.5 for r in rnd]
            key_m, key_c = "mispredicted_0::DirectCond", "numCycles"
            stat_pen = [(r[key_c] - pat[key_c]) / (r[key_m] - pat[key_m]) for r in rnd]
            nbr = 3 * 64 * WORDS  # dynamic kernel branches over all 3 reps
            rate = [(r[key_m] - pat[key_m]) / nbr for r in rnd]
            loop = sum({**LOOP, delay: v}.values())
            summary.append(
                {
                    "delay": delay,
                    "value": v,
                    "default": LOOP[delay],
                    "penalty": med(pens),
                    "penalty_seeds": pens,
                    "ci": [min(pens), max(pens)],
                    "penalty_stats": med(stat_pen),
                    "mispredict_rate": med(rate),
                    "cpb_pattern": pat["cpb"],
                    "cpb_random": [r["cpb"] for r in rnd],
                    "loop_delay_sum": loop,
                }
            )
            print(
                f"{delay}={v}: penalty {med(pens):.2f} [{min(pens):.2f}, {max(pens):.2f}]  stats-based "
                f"{med(stat_pen):.2f}  mispredict rate {med(rate):.3f}  loop delays {loop}",
                flush=True,
            )
    write_json(os.path.join(OUT, "summary.json"), summary)


if __name__ == "__main__":
    main()
