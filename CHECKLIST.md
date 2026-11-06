# M1 checklist

Run order on one M1 Mac, top to bottom. Each line is one run and what it
should show. Values are published ones, with who published them.

`run X Y` is `python3 targets/m1/m1_kernels.py run X Y`. `sweep F` is the
`--fillers F` line in [targets/m1/README.md](targets/m1/README.md), which
also covers setup.

## 1. Setup

- [ ] An M1, M1 Pro, Max or Ultra on AC power, Low Power Mode off, Wi-Fi off, other apps closed
- [ ] `python3 -m unittest` passes
- [ ] Sanity sweep (README step 1): clock 3.1-3.2 GHz, 0 rejected launches
- [ ] Sanity sweep again with `--tag m1-sanity2`: the knee falls inside the first run's interval
- [ ] Sanity sweep with `--tag m1-ecore --exec-prefix "taskpolicy -c background"`: launches near 2.06 GHz, all rejected

## 2. Known numbers

Values not in dispute. A miss here points at the setup, not the core.

- [ ] `run sched P1.8`: add 1 cycle, 6 a cycle; mul 3, 2 a cycle; udiv 9; fadd 3; fmul and fmadd 4, 4 a cycle; fdiv 10; fsqrt 13 (Johnson's tables)
- [ ] `run mem P2.1`: L1D 128 KiB, 64 B lines, 8 ways, 3-cycle pointer chase; L2 12 MiB, 128 B lines; DRAM 300-350 cycles (7-cpu, Frumusanu, Handley, Johnson)
- [ ] `run mem P2.2`: L1 dTLB 160 entries, L2 TLB 3072 (7-cpu, Frumusanu, Handley, Chen)
- [ ] `run fe P3.1`: 8 instructions a cycle (Johnson, Handley, Frumusanu)
- [ ] `run fe P3.2`: L1i 192 KiB, one taken jump a cycle (Ravichandran et al., Kim et al., Chen)
- [ ] `run fe P3.3`: L0 BTB of 1024 branches, no taken bubble (Handley, Chen, Suzuki)
- [ ] `run pred P3.4`: history of 100 taken branches (Tuby and Morrison; Chen, Qu and Zhang)
- [ ] `run fe P3.7`: mispredict penalty 13 cycles (7-cpu, Suzuki)
- [ ] `run sched P1.7`: retire 8 groups, 56 NOPs, a cycle (Johnson, Handley)
- [ ] `sweep nop`: knee near 2277, 7 NOPs per retire group (Johnson, Handley, Chen)
- [ ] `sweep branch`: knee near 330 retire groups (Johnson, Handley, Chen)
- [ ] `sweep int`, `sweep fp`, `sweep flags`: knee near 380, 432 and 128 (file size: Handley, Johnson) or 362, 382 and 123 (free for speculation: Chen)

## 3. Disputes

- [ ] `run fe P3.1-page`: fetch block of 16 (Handley, Chen) or 8 (Johnson). Flat to 17 instructions then a step means 16; steps at 10, 18 and 26 mean 8
- [ ] `run sched P1.6-movimm` and `sweep movimm`: immediate moves done at rename, 2 in 8 (Johnson) or all (Suzuki). 5-6 of every 8 taking an ALU means Johnson, 0 means Suzuki; the window knee falls near 380, 418 or 620
- [ ] `python3 targets/m1/m1_sched.py P1.5 --fp-filler both`:
  - [ ] multiply queues: knee near 26 (13 each, paired: Handley) or 56 (28: Chen)
  - [ ] FP queues: knee near 72 (18 each: Handley) or 144 (36: Johnson, Chen)
  - [ ] integer total: add knee near 134 (Johnson, Chen) or about half (Handley)
  - [ ] load/store: preloaded stores cut load room from the first op (one queue: Chen) or after a break (four of 12: Handley)
  - [ ] FP dispatch buffer: 12 or 14 (Handley), from the fabs and fcmp pair
- [ ] `sweep load` with the miss delay: LQ near 130 (Johnson) or 151 (Frumusanu); the FSQRT-delay run checks rule A5
- [ ] `sweep store` with the miss delay: about 100 for an SQ of 60 (Johnson, Handley), so rule A5 should flag it
- [ ] `run mem P2.5`: store-to-load round trip of 6 cycles (Handley) or 7.5 (Chen); equal rows give the value, different rows explain the gap
- [ ] `run mem P2.3-bw`: one-core DRAM reads at 64 GB/s (Handley) or 58 (Frumusanu), median of 5 or more launches
- [ ] `run mem P2.1-slc`: SLC replacement LRU (Xu et al., SLAC) or random (Xu et al., EXAM). A sharp edge means LRU, a gradual one random

Settled on paper, to confirm:

- [ ] `run fe P3.2-ways`: L1i 6 ways (Ravichandran et al., Kim et al., Chen), not 12 (Handley)
- [ ] `run mem P2.2-ways`: L1 dTLB 5 ways of 32 sets (Handley, Chen). About 36 would mean 4 ways and a victim buffer
- [ ] `run mem P2.2`, iTLB row: 192 entries (Chen) or 128 (4 ways of 32 sets: Ravichandran et al.)
- [ ] `run mem P2.1`, L2 row: 17-18 cycles (Handley, 7-cpu)

## 4. Holes

No published value, or one source.

- [ ] `run mem P2.3`: L1D misses in flight
- [ ] `run pred P3.9`: fetch-to-decode queue depth
- [ ] `run pred P3.8`: fetch-directed prefetch, absent (Handley)
- [ ] `run pred P3.5`: indirect predictor targets
- [ ] `run pred P3.6`: return stack depth, 50 (Chen)
- [ ] `run mem P2.4`: prefetchers, a stride prefetcher that catches dependent chains (Handley)
- [ ] `run mem P2.5`: store-load pairs the dependence predictor tracks
- [ ] `run fe P3.2-repl`: L1i replacement policy
- [ ] `run mem P2.1-store`: whether a store miss allocates in the L1D
- [ ] `run mem P2.3-fill`: whether two misses to one line share a fill buffer
- [ ] `run fe P3.10`: loop buffer, present with a size or absent

## 5. Send back

- [ ] `python3 targets/m1/m1_kernels.py pack` and `tar czf m1-results.tgz results/m1`, with the console output
