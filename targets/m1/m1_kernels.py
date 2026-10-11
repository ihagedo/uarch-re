#!/usr/bin/env python3
"""Run the non-window kernels on Apple silicon, one family or all of them, and pack the results.

python3 targets/m1/m1_kernels.py list
python3 targets/m1/m1_kernels.py run all [--tag m1]          # or a family: mem fe fp sched
python3 targets/m1/m1_kernels.py run fp P3.6 P3.4 --tag m1    # some PoCs of one family
python3 targets/m1/m1_kernels.py pack [--tag m1]              # results/m1/<tag>.tgz
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native  # noqa: E402

FAMILIES = ("mem", "fe", "fp", "sched")


def family(name):
    return importlib.import_module(f"m1_{name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run = native.add_args(sub.add_parser("run"))
    run.add_argument("which", choices=("all",) + FAMILIES)
    run.add_argument("pocs", nargs="*")
    pk = sub.add_parser("pack")
    pk.add_argument("--tag", default="m1")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        total = 0
        for f in FAMILIES:
            m = family(f)
            for poc in m.POCS:
                total += m.EST_MIN.get(poc, 0)
                print(f"{f:6s} {poc:6s} ~{m.EST_MIN.get(poc, 0):3d} min")
        print(f"all    ~{total} min")
        return
    if a.cmd == "pack":
        print("wrote", native.pack(a.tag))
        return
    r = native.Runner(a)
    for f in FAMILIES if a.which == "all" else (a.which,):
        m = family(f)
        pocs = a.pocs or list(m.POCS)
        for poc in pocs:
            raw, summary = m.POCS[poc](r, a)
            print(f"{f} {poc}: wrote {r.save(m.FAMILY, poc, raw, summary)}")
            for row in summary:
                print(f"  {row['structure']}: {row['value']} {row.get('unit', '')} ci {row.get('ci')}"
                      f"  published {row.get('published')}")  # fmt: skip
    if not a.gem5_smoke:
        print("wrote", native.pack(a.tag))


if __name__ == "__main__":
    main()
