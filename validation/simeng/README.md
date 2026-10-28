# validation/simeng

The M1 kernels on [SimEng](https://github.com/UoB-HPC/SimEng), Bristol's AArch64 cycle model, with
its M1 config: a ground truth beside gem5 and BOOM where the config sets the size under test.
SimEng has no caches without SST, so every data access here takes 300 cycles.

| file | what |
| --- | --- |
| `probes.patch` | maps the PMU registers the kernels read; `isb` and `chkfeat` run as no-ops |
| `configs.py` | `configs/m1_firestorm.yaml` with current key names, and one variant per size under test |
| `m1.py` | a `targets/m1` PoC under SimEng (static binaries, cycle counter) |
| `sweep.py` | `targets/m1/m1_sweep.py` under SimEng |

## Build and run

```
git clone https://github.com/UoB-HPC/SimEng && cd SimEng
git checkout 93fe67d1 && git apply <uarch-re>/validation/simeng/probes.patch
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DSIMENG_ENABLE_TESTS=OFF -DSIMENG_OPTIMIZE=ON
cmake --build build -j 12
```

With GCC 14 a build without `-DSIMENG_OPTIMIZE=ON` stops on `-Werror=maybe-uninitialized`. `SIMENG`
and `SIMENG_SRC` point the scripts at the binary and the tree.

```
python3 validation/simeng/configs.py CFG SPECIAL && simeng CFG/gen.yaml
python3 validation/simeng/m1.py CFG/base.yaml sched P1.5 --parts sizes
python3 validation/simeng/m1.py CFG/lb24.yaml fe P3.10
python3 validation/simeng/sweep.py CFG/lq151.yaml lq151 --fillers load --depth 2 --n 32:192:8 \
    --iters 40 --inner-reps 2 --outer-reps 2 --chain-mb 1
```
