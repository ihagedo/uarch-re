# validation

Cores with known ground truth. Every detection method gets a measured error
here before it is trusted on real silicon.

| dir | level | ground truth | covers |
| --- | --- | --- | --- |
| `toy/` | L0 | Python OoO model (`uarchre/backends/platform/toy.py`) with switches for retire groups, issue-time LSQ, two-level SQ and load prediction. It shares the analysis's assumptions, so it checks the pipeline and the anomaly rules, not accuracy | ROB, PRF, LQ/SQ and near-tie sweeps (`window_sweep.py`); P0.1, P0.3, P4.1-P4.4 |
| `gem5/` | L1 | gem5 v25.1 O3 for RISC-V, x86 (an SSE-only probe: gem5 decodes no AVX) and ARM (with a PMU), any parameter overridable; bare-metal full system for TLBs. One folder per kernel family (`gem5/README.md`) | P0.1, P0.6, P1.x, P2.x, P3.x, P6.1 |
| `simeng/` | L1 | SimEng with Bristol's M1 config, sizes set per variant | P1.3, P1.5, P3.10 |
| `boom/` | L2 | Chipyard BOOM v3 Small/Medium/Large/Mega, bare-metal under Verilator | P0.5, P1.1-P1.3 |

Each campaign script's docstring has its usage line.
PoC ids are listed in `docs/pocs.md`.
