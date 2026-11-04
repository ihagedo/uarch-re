"""Bare-metal full-system RISC-V O3 config for the P2.2 TLB probe.

SE mode translates through the process page table functionally (no TLB lookup, no
walk), so TLB probes run here: a RiscvBareMetal binary (entry 0x80000000) that builds
Sv39 page tables and chases pointers in S-mode. Core and caches match config_o3.py;
both walker ports sit on the L2 bus. `--set` works as in config_o3.py.

  gem5.opt mem_fs_config.py --bin probe.elf [--set cpu.mmu.dtb.size=32 ...]
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
    IQUnit,
    Root,
    RiscvBareMetal,
    RiscvO3CPU,
)

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
ap.add_argument("--clock", default="2GHz")
args = ap.parse_args()


class L1D(Cache):
    size = "32kB"
    assoc = 8
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 20


class L1I(Cache):
    size = "32kB"
    assoc = 8
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 8
    tgts_per_mshr = 20


class L2(Cache):
    size = "256kB"
    assoc = 8
    tag_latency = 10
    data_latency = 10
    response_latency = 4
    mshrs = 32
    tgts_per_mshr = 12


system = System()
system.clk_domain = SrcClockDomain(clock=args.clock, voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(0x80000000, size="512MB")]

cpu = RiscvO3CPU()
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
cpu.instQueues = [IQUnit(numEntries=256)]
cpu.numPhysVecRegs = 256
system.cpu = cpu

cpu.icache = L1I()
cpu.dcache = L1D()
cpu.icache.cpu_side = cpu.icache_port
cpu.dcache.cpu_side = cpu.dcache_port
system.l2bus = L2XBar()
cpu.icache.mem_side = system.l2bus.cpu_side_ports
cpu.dcache.mem_side = system.l2bus.cpu_side_ports
cpu.mmu.connectWalkerPorts(system.l2bus.cpu_side_ports, system.l2bus.cpu_side_ports)
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

system.workload = RiscvBareMetal(bootloader=args.bin)
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

root = Root(full_system=True, system=system)
m5.instantiate()
ev = m5.simulate()
print("gem5 exit:", ev.getCause(), "code", ev.getCode())
