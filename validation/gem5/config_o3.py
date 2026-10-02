"""Minimal gem5 SE-mode O3 config (RISC-V or ARM) with overridable structures.

  gem5.opt config_o3.py [--isa riscv|arm] --cmd ./kernel [--args "bytes iters reps"] \
      --numROBEntries 128 --LQEntries 32 --set cpu.branchPred.ras.numEntries=16

`--<O3 param>` sets a CPU param; `--set path=value` sets any param below `system` (int,
float, bool, an m5.objects constructor or a string; MemorySize params need a unit even for
entry counts, e.g. `cpu.SSITSize=1024B`). --fdp adds an I-cache FetchDirectedPrefetcher.
"""

import argparse
import m5
import m5.objects
from m5.objects import (
    System,
    SrcClockDomain,
    VoltageDomain,
    AddrRange,
    Cache,
    L2XBar,
    SystemXBar,
    MemCtrl,
    DDR3_1600_8x8,
    Process,
    IQUnit,
    SEWorkload,
    Root,
)

O3_PARAMS = [
    "numROBEntries",
    "numPhysIntRegs",
    "numPhysFloatRegs",
    "numPhysVecRegs",
    "numPhysCCRegs",
    "LQEntries",
    "SQEntries",
    "fetchWidth",
    "decodeWidth",
    "renameWidth",
    "dispatchWidth",
    "issueWidth",
    "wbWidth",
    "commitWidth",
    "squashWidth",
]

ap = argparse.ArgumentParser()
ap.add_argument("--isa", choices=["riscv", "arm"], default="riscv")
ap.add_argument("--cmd", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
ap.add_argument("--l1d", default="32kB")
ap.add_argument("--l1i", default="32kB")
ap.add_argument("--l2", default="256kB")  # an 8 MB chase misses to DRAM
ap.add_argument("--fdp", action="store_true", help="FetchDirectedPrefetcher on the I-cache")
ap.add_argument("--fdp-snoop", type=int, default=1)
ap.add_argument("--clock", default="2GHz")
for p in O3_PARAMS:
    ap.add_argument("--" + p, type=int, default=None)
# v25 replaced numIQEntries with a vector of IQUnits; one unified queue here
ap.add_argument("--numIQEntries", type=int, default=256)
args = ap.parse_args()


class L1D(Cache):
    size = args.l1d
    assoc = 8
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 20


class L1I(Cache):
    size = args.l1i
    assoc = 8
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 8
    tgts_per_mshr = 20


class L2(Cache):
    size = args.l2
    assoc = 8
    tag_latency = 10
    data_latency = 10
    response_latency = 4
    mshrs = 32
    tgts_per_mshr = 12


system = System()
system.clk_domain = SrcClockDomain(clock=args.clock, voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("1GB")]

# by name: a single-ISA build has only its own CPU class (ARM runs go through
# arm_config_o3.py, which also attaches the PMU the cycle counter reads)
cpu = getattr(m5.objects, {"riscv": "RiscvO3CPU", "arm": "ArmO3CPU"}[args.isa])()
# wide pipeline so structure sizes are the only limit
for w in [
    "fetchWidth",
    "decodeWidth",
    "renameWidth",
    "dispatchWidth",
    "issueWidth",
    "wbWidth",
    "commitWidth",
    "squashWidth",
]:
    setattr(cpu, w, 8)
cpu.instQueues = [IQUnit(numEntries=args.numIQEntries)]
cpu.numPhysVecRegs = 256
for p in O3_PARAMS:
    v = getattr(args, p)
    if v is not None:
        setattr(cpu, p, v)
system.cpu = cpu

cpu.icache = L1I()
cpu.dcache = L1D()
cpu.icache.cpu_side = cpu.icache_port
cpu.dcache.cpu_side = cpu.dcache_port
system.l2bus = L2XBar()
cpu.icache.mem_side = system.l2bus.cpu_side_ports
cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.l2 = L2()
system.l2.cpu_side = system.l2bus.mem_side_ports
system.membus = SystemXBar()
system.l2.mem_side = system.membus.cpu_side_ports
cpu.createInterruptController()
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

process = Process()
process.cmd = [args.cmd] + args.args.split()
system.workload = SEWorkload.init_compatible(args.cmd)
cpu.workload = process
cpu.createThreads()


def parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if "(" in v:
        return eval(v, vars(m5.objects))
    return v


for item in args.set:
    path, value = item.split("=", 1)
    *parents, leaf = path.split(".")
    obj = system
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, parse_value(value))

# --set cannot make the registerCache/registerMMU calls; the decoupled front end
# (cpu.decoupledFrontEnd=True) also needs fetchTargetWidth >= fetchBufferSize
if args.fdp:
    pf = m5.objects.FetchDirectedPrefetcher(
        use_virtual_addresses=True, cpu=cpu, cache_snoop=bool(args.fdp_snoop)
    )
    pf.registerCache(cpu.icache)
    pf.registerMMU(cpu.mmu)
    cpu.icache.prefetcher = pf

root = Root(full_system=False, system=system)
m5.instantiate()
ev = m5.simulate()
print("gem5 exit:", ev.getCause())
