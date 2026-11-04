# validation/gem5

gem5 v25.1 O3 campaigns, one folder per kernel family. Each script's
docstring has its usage line.

| folder | checks |
| --- | --- |
| `configs/` | the gem5 configs: O3 for RISC-V and x86, ARM with a PMU, full system for TLBs, fetch-directed prefetch |
| `window/` | ROB, register headroom, LQ and SQ (RISC-V, x86, ARM), and the spec round trip |
| `sched/` | scheduler queues, dispatch buffers, retire width |
| `mem/` | caches, TLBs, MLP, prefetchers, store sets, store misses, fill buffers |
| `frontend/` | fetch width, L1i, BTB, mispredict penalty, L1i replacement |
| `pred/` | branch history, indirect targets, RAS, fetch queue, fetch-directed prefetch |
| `micro/` | instruction latency and throughput |

`build.sh` builds gem5: `ISA=RISCV` (default), `ARM` or `X86`.
