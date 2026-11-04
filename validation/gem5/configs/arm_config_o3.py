"""gem5 ARM SE config: config_o3.py plus an ArmPMU, so PMCCNTR_EL0 counts core cycles.

SE mode has no generic timer and no PMU (both read 0); gem5 does not trap EL0 PMU access,
so the kernel enables the cycle counter itself. Takes config_o3.py's arguments (`--isa arm`
implied) and --numPhysCCRegs:

  gem5.opt arm_config_o3.py --cmd ./kernel --args "..." --numROBEntries 64 --numPhysCCRegs 96
"""

import os
import sys

import m5.objects
from m5.objects import ArmO3CPU, ArmPMU
from m5.params import NULL

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "config_o3.py")

if not hasattr(m5.objects, "RiscvO3CPU"):
    m5.objects.RiscvO3CPU = None  # the ARM-only build lacks it; never instantiated under --isa arm

# O3 params config_o3.py does not expose: taken out of argv here and set on the CPU
EXTRA = {"numPhysCCRegs": None}
argv, i = [], 0
while i < len(sys.argv):
    a = sys.argv[i]
    key = a[2:].split("=", 1)[0] if a.startswith("--") else None
    if key in EXTRA:
        if "=" in a:
            EXTRA[key] = int(a.split("=", 1)[1])
        else:
            EXTRA[key] = int(sys.argv[i + 1])
            i += 1
    else:
        argv.append(a)
    i += 1
if "--isa" not in argv:
    argv[1:1] = ["--isa", "arm"]
sys.argv = argv


class PmuArmO3CPU(ArmO3CPU):
    """ArmO3CPU with a PMU on every ISA and the extra O3 params applied."""

    def __init__(self, **kw):
        super().__init__(**kw)
        for k, v in EXTRA.items():
            if v is not None:
                setattr(self, k, v)

    def createThreads(self):
        super().createThreads()
        for isa in self.isa:
            isa.pmu = ArmPMU(interrupt=NULL, platform=NULL)
            isa.pmu.addArchEvents(cpu=self)


m5.objects.ArmO3CPU = PmuArmO3CPU

with open(BASE) as fh:
    code = compile(fh.read(), BASE, "exec")
exec(code, {"__name__": "__m5_main__", "__file__": BASE})
