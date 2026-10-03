#!/usr/bin/env python3
"""P2.2 on gem5 O3 (RISC-V), SE mode: does the data TLB size change timing?

In SE mode gem5's RISC-V TLB::translate() goes through the process page table
functionally: no TLB lookup, no walk, no latency. So page-stride chains over
P = 16..8192 pages should take the same cycles for every cpu.mmu.dtb.size.

    python3 validation/gem5/mem_tlb.py
"""

import os

from mem_common import run_jobs, RESULTS

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


if __name__ == "__main__":
    run_jobs("mem_tlb", jobs(), OUT)
