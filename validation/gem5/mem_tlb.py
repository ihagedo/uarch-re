#!/usr/bin/env python3
"""P2.2 on gem5 O3 (RISC-V or ARM), SE mode: does the TLB size change timing?

In SE mode gem5 translates through the process page table functionally (RISC-V TLB::translate(), ARM
MMU::translateSe()): no TLB lookup, no walk, no latency. So page-stride chains over P = 16..8192 pages
should take the same cycles for every cpu.mmu.dtb.size; on ARM also the code-page jump chain for every
cpu.mmu.itb.size, with the TLB counters recorded.

    python3 validation/gem5/mem_tlb.py [--isa aarch64]
"""

import os

from mem_common import run_jobs, RESULTS, isa_args, for_isa

OUT = os.path.join(RESULTS, "mem-tlb", "se_control.json")
PAGES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SIZES = [4, 64, 1024]


def jobs():
    return [
        {
            "key": f"dtb{n}",
            "dtb_size": n,
            "argv": ["page", 1000, 3, 0, 0, "w=20000", *PAGES],
            "params": {},
            "sets": [f"cpu.mmu.dtb.size={n}"],
        }
        for n in SIZES
    ]


ARM_TLB_STATS = [
    "system.cpu.mmu.dtb.accesses",
    "system.cpu.mmu.itb.accesses",
    "system.cpu.mmu.l2_shared.accesses",
    "system.cpu.mmu.walker.dataWalksS1",
    "system.cpu.mmu.walker.instructionWalksS1",
]
CODE_P = [4, 16, 64, 128, 256]


def arm_jobs():
    out = [{**j, "extra_stats": ARM_TLB_STATS} for j in jobs() if j["dtb_size"] in (4, 1024)]
    for n in (8, 64):
        out.append(
            {
                "key": f"itb{n}",
                "itb_size": n,
                "argv": ["code", 1000, 3, *CODE_P],
                "params": {},
                "sets": [f"cpu.mmu.itb.size={n}"],
                "extra_stats": ARM_TLB_STATS,
            }
        )
    return for_isa(out, "aarch64")


if __name__ == "__main__":
    import sys

    isa, _ = isa_args(sys.argv[1:])
    if isa == "aarch64":
        run_jobs("arm_mem_tlb", arm_jobs(), os.path.join(RESULTS, "arm-memory-tlb", "se_control.json"))
    else:
        run_jobs("mem_tlb", jobs(), OUT)
