# validation/boom

The window probe run bare-metal on Chipyard BOOM v3 RTL under Verilator (level L2:
the configured RTL parameters are ground truth). The driver is
`uarchre/backends/platform/boom.py`; its docstring says how the bare-metal harness
differs from the hosted one.

| file | what |
| --- | --- |
| `setup.sh` | one-time Chipyard 1.14.0 setup on local disk (conda env, toolchain, CIRCT) |
| `build.sh` | one Verilator simulator per BOOM config |
| `campaign.py` | window-probe sweeps on one config (`--config --tag`) or the `CAMPAIGN` set (`--all [--only tags]`) |

## Build and run

```
validation/boom/setup.sh
validation/boom/build.sh SmallBoomV3Config MediumBoomV3Config LargeBoomV3Config MegaBoomV3Config
taskset -c 4-19 python3 validation/boom/campaign.py --all --only small-d2,medium-d2,large-d2,mega-d2 --jobs 4
python3 -m unittest tests.test_boom
```

`CHIPYARD` (default `/var/tmp/$USER/chipyard`) points the driver at the tree. Every
simulator launch runs on CPUs 4-19 (`UARCHRE_GEM5_CPUS`) and holds one of the
host-wide gem5 slots (`UARCHRE_GEM5_SLOTS`), so BOOM and gem5 campaigns share one
budget and CPUs 0-3 stay quiet for native timing. A launch is ~50k simulated cycles,
~30 s.

## Results

`results/boom/<tag>/`: `sweep_raw.csv` (every inner rep), `sweep.csv` (median per
point), `knees.json` (knees, estimates and the RTL ground truth), `spec.yaml`,
`sweep.png`, and `sims.csv` (simulated cycles and wall time per launch).
