# explainer

Teaching material driven by `spec/cores/*.yaml`. No number is copied by hand;
a `null` spec field shows up as a gap.

## factory/

The toy OoO model (`uarchre/backends/platform/toy.py`) ported to plain
ES-module JavaScript, with no build step and no dependencies. Its viewer
animates the window probe on a core spec: the ROB as an order ledger, rename
headroom as tote racks, the LQ and SQ as docks, misses as supplier trucks.

| file | role |
| --- | --- |
| `engine.js` | `Machine` (one cycle per `step()`) and `runToy()` |
| `kernels.js`, `resolve.js`, `knee.js` | ports of `uarchre/kernels`, `toy_trace.resolve_spec` and `analysis/knee.py` |
| `index.html`, `view.js`, `render.js` | the viewer: ledger, racks, docks, trucks, stall readout, N sweep |
| `conformance.js`, `test.html`, `cli.mjs` | conformance checks, in the browser and in Node |
| `build_golden.py` | writes `golden/` and `specs/*.json` (JSON mirrors of the specs) |

ES modules do not load from `file://`, so serve the folder:

```
python3 -m http.server -d explainer/factory 8000   # viewer at /, conformance at /test.html
node explainer/factory/cli.mjs sweep --spec r10000 --set miss_latency=200 --filler nop --n 2:60:1
```

Conformance: `build_golden.py` records event traces from the Python model
(`toy_trace.trace_toy`), and `conformance.js` requires every dispatch, issue,
complete, retire and stall event to match, along with cycles/iter, the
resolver output and the knee fits. After a change to `toy.py`, the kernels,
`knee.py` or a spec:

```
python3 explainer/factory/build_golden.py      # --check only reports stale files
node explainer/factory/cli.mjs conformance     # exit 1 on any difference
python3 -m unittest tests.test_factory         # both of the above; Node tests skip without Node >= 18
```

A semantic change goes into `toy.py`, then `toy_trace.py`, then `engine.js`.
The engine has a classic ROB only, so it refuses the Firestorm spec.

## labs/

`renaming-lab.md` is a student handout on register renaming, from the
R10000's active list to Firestorm's retire rows. `build_lab.py` renders it from
the two specs and three `./sweep.py toy` runs; `renaming-lab.json` keeps the
commands and every number they produce.

```
taskset -c 4-19 python3 explainer/labs/build_lab.py   # rebuild, then review the diff
python3 -m unittest tests.test_lab                     # re-runs the commands, checks every number
```
