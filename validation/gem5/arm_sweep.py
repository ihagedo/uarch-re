#!/usr/bin/env python3
"""`./sweep.py gem5 --isa aarch64`, plus the ArmO3CPU ground truth.

Takes `./sweep.py gem5` arguments; the platform and `--isa aarch64` are implied. The gem5
backend routes aarch64 to arm_config_o3.py; arm_truth() is the same truth table
(tests/test_aarch64.py checks it).

  validation/gem5/arm_sweep.py --tag arm-smoke --gem5 numROBEntries=64 --fillers nop \
      --n 40:80:8 --chain-mb 4 --iters 40 --outer-reps 1 --jobs 8
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

ARM_CONFIG = os.path.join(HERE, "arm_config_o3.py")

# registers O3 maps at reset (every index of every class in gem5's src/arch/arm/regs/*.hh):
# int 42 (X0-X30 alias these), vec 44 (scalar FP writes a VecReg, so the FP pool is
# numPhysVecRegs), cc 6 (nz, c, v, ge, fp, zero; a flag-setting op writes NZ, C and V)
ARM_ARCH = {"int": 42, "vec": 44, "cc": 6}
CC_PER_FLAGS_OP = 3
# ArmO3CPU / BaseO3CPU defaults, plus config_o3.py's numPhysVecRegs = 256;
# numPhysCCRegs defaults to 5 * numPhysIntRegs on ArmO3CPU
ARM_DEFAULTS = {
    "numROBEntries": 192,
    "numPhysIntRegs": 256,
    "numPhysVecRegs": 256,
    "LQEntries": 32,
    "SQEntries": 32,
}


def arm_truth(params):
    p = {**ARM_DEFAULTS, **params}
    cc = p.get("numPhysCCRegs", 5 * p["numPhysIntRegs"])
    return {
        "rob": p["numROBEntries"],
        "int_prf_headroom": p["numPhysIntRegs"] - ARM_ARCH["int"],
        "fp_prf_headroom": p["numPhysVecRegs"] - ARM_ARCH["vec"],
        "lq": p["LQEntries"],
        "sq": p["SQEntries"],
        "flags_prf_headroom": cc - ARM_ARCH["cc"],
        "flags_ops": (cc - ARM_ARCH["cc"]) // CC_PER_FLAGS_OP,
        "int_prf": p["numPhysIntRegs"],
        "fp_prf": p["numPhysVecRegs"],
        "flags_prf": cc,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "gem5":
        argv = argv[1:]
    if "--isa" not in argv:
        argv += ["--isa", "aarch64"]
    import importlib.util

    spec = importlib.util.spec_from_file_location("uarchre_sweep_cli", os.path.join(ROOT, "sweep.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    os.chdir(ROOT)  # sweep.py writes results/<platform>/<tag> relative to cwd
    sys.argv = ["sweep.py", "gem5"] + argv
    cli.main()


if __name__ == "__main__":
    main()
