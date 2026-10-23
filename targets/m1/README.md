# targets/m1: Apple silicon timing driver

`m1_sweep.py` runs the window probe (`uarchre/kernels/window.py`) on a
performance core of an Apple silicon Mac, timing only, no root, and writes the
same result files as `./sweep.py`. `m1_kernels.py` runs every other kernel
(caches, TLBs, MLP, store sets, frontend, predictors, schedulers, retire).
Neither has run on a Mac yet, so expect small fixes on the first run.

Matching the published numbers needs M1-generation silicon (M1, M1 Pro, Max or
Ultra). Later chips have fewer retire rows and different queue sizes.

## How it works

- Kernels: `uarchre/backends/isa/aarch64.py` emits each point as C with inline
  assembly, built with `clang -O1`.
- Timer: `mach_absolute_time()` ticks at 24 MHz, so each launch also times a
  chain of 1-cycle adds before and after its reps to convert ticks to cycles.
- P-core: macOS has no hard affinity, so the kernel asks for
  `QOS_CLASS_USER_INTERACTIVE` and a launch only counts if its clock reads at
  least `--min-ghz` (default 2.8; Firestorm ~3.2 GHz, Icestorm ~2.06 GHz).
  Launches that drift more than 2 % are re-run; every attempt is kept.

## What you need

- An Apple silicon Mac on AC power, Low Power Mode off, other apps closed. The
  driver refuses to run on battery, in Low Power Mode, under Rosetta, or with a
  load average above 2 (`--force` runs anyway and records why).
- Xcode Command Line Tools for `clang`: `xcode-select --install`.
- A native arm64 Python 3.10+ with numpy, PyYAML and matplotlib, from
  python.org or Homebrew (`brew install python`); the Xcode `python3` is 3.9,
  too old. `python3 -c "import platform; print(platform.machine())"` must print
  `arm64`. Then `python3 -m pip install --user numpy pyyaml matplotlib`.
- ~1 GB of free memory (two 256 MiB chains per launch) and ~1 GB of disk for
  compiled kernels in `build/native/`.

## Run it

From the repo root:

```sh
# 1. Sanity check (~2 min). Look for "clock 3.1-3.2 GHz" and "0 rejected".
python3 targets/m1/m1_sweep.py --tag m1-sanity --fillers nop --depth 2 \
    --n 1800:2600:200 --outer-reps 1

# 2. One structure per run (each ~5-15 min). Handley's published knee in the comment.
python3 targets/m1/m1_sweep.py --tag m1-nop-d2    --fillers nop    --depth 2 --n 1600:2800:32  # ~2230-2300 = 7 x ~330 retire rows, not a ROB
python3 targets/m1/m1_sweep.py --tag m1-int-d2    --fillers int    --depth 2 --n 200:480:8     # ~380 int registers' headroom
python3 targets/m1/m1_sweep.py --tag m1-fp-d2     --fillers fp     --depth 2 --n 240:520:8     # ~415-432
python3 targets/m1/m1_sweep.py --tag m1-flags-d2  --fillers flags  --depth 2 --n 48:220:4      # ~124-128
python3 targets/m1/m1_sweep.py --tag m1-load-d2   --fillers load   --depth 2 --n 64:220:4      # ~125 with a miss delay (genuine LQ ~130)
python3 targets/m1/m1_sweep.py --tag m1-store-d2  --fillers store  --depth 2 --n 32:180:4      # ~100 with a miss delay (genuine SQ ~60: rule A5 must flag it)
python3 targets/m1/m1_sweep.py --tag m1-branch-d2 --fillers branch --depth 2 --n 160:420:8     # ~330 (one failable op per retire row)
python3 targets/m1/m1_sweep.py --tag m1-movimm-d2 --fillers movimm --depth 2 --n 256:832:16   # ~380 like adds; ~418 extra immediate registers; ~620 none (rename retire queue)

# 3. Delay-type check (rule A5): load and store fillers behind a dependent FSQRT
#    chain; 24 links x 13 cycles outlasts the fill time. --iters 10000 because
#    chain nodes are budgeted as reps x iters x depth even for ALU delays.
python3 targets/m1/m1_sweep.py --tag m1-load-fp24  --fillers load  --delay fp --depth 24 --iters 10000 --n 64:400:8
python3 targets/m1/m1_sweep.py --tag m1-store-fp24 --fillers store --delay fp --depth 24 --iters 10000 --n 32:400:8

# 4. Send back the results and the full console output (the driver never uses the network).
tar czf m1-results.tgz results/m1
```

`python3 targets/m1/m1_sweep.py -h` lists the options (`--outer-reps`,
`--inner-reps`, `--iters`, `--chain-mb`, `--min-ghz`, `--max-drift`,
`--retries`, `--seed`, ...).

## The other kernels

```sh
python3 targets/m1/m1_kernels.py list            # PoCs and rough minutes (~50 in all)
python3 targets/m1/m1_kernels.py run all --tag m1
python3 targets/m1/m1_kernels.py run fp P3.6     # one family, some PoCs
```

Each launch sits between two add-chain clock probes: ticks become core cycles,
and a launch that ran slow (an E-core, throttling, a migration) is retried.
Results go to `results/m1/<tag>/<family>/<poc>/{raw,summary,run}.json` and are
packed into `results/m1/<tag>.tgz`; every summary row puts the measured value
next to the published one. Without a Mac, `--gem5-smoke` runs the same
binaries under gem5 ARM as a functional test.
`--l1i-line` overrides the L1i line size (default: `spec/cores/firestorm.yaml`).

## Output

`results/m1/<tag>/`:

| file | contents |
| --- | --- |
| `sweep_raw.csv` | every inner rep of every attempt: ticks (24 MHz counter ticks x 1024), cycles, probe before and after, counter rate, GHz, drift, accepted |
| `sweep.csv` | per-N median in core cycles |
| `knees.json` | knees with bootstrap CIs, size estimates, findings, run conditions, machine info |
| `machine.json` | `hw.model`, CPU brand, per-perflevel core counts and cache sizes, macOS version, power source, Low Power Mode, Rosetta, load average |
| `spec.yaml` | provenance-tagged spec fragment |
| `sweep.png` | quick-look plot |

`machine.json` never contains the serial number, hardware UUID, hostname,
user name or IP address. Read it before you send it.

## Linux on Apple silicon (Asahi)

The same script runs under Asahi Linux. It reads `CNTVCT_EL0` instead (also
24 MHz) and pins with `taskset`. On an M1 the Firestorm cores are CPUs 4-7
(check that `/sys/devices/system/cpu/cpu4/cpu_capacity` is the largest):

```sh
python3 targets/m1/m1_sweep.py --tag asahi-nop-d2 --fillers nop --depth 2 --n 1600:2800:32 --pin-cpu 4
```

It needs `cc` (gcc or clang); `--cc clang` picks clang.

## Known limitations

- No PMU counters (no root `kperf`).
- On M1 a NOP knee measures 7 x retire rows, not a ROB; rule A4 flags it.
- 256 MiB chains also miss the L2 TLB, which lengthens the delay (harmless)
  and adds some variance.
