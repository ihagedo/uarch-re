#!/usr/bin/env python3
"""Build the register-renaming lab handout from the specs and toy runs.

    taskset -c 4-19 python3 explainer/labs/build_lab.py     # ~30 s; rewrites renaming-lab.{md,json}

Runs the handout's ./sweep.py toy commands in a scratch directory and renders
renaming-lab.md from their knees.json and sweep.csv. renaming-lab.json keeps the
commands and every toy-derived number for tests/test_lab.py.
"""

import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.backends.platform.toy_trace import resolve_spec, toy_arg

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "renaming-lab.md")
JSON = os.path.join(HERE, "renaming-lab.json")


def load_spec(core):
    with open(os.path.join(ROOT, "spec", "cores", f"{core}.yaml")) as fh:
        return yaml.safe_load(fh)


def v(node, *path):
    for p in path:
        node = node[p]
    return node


# ------------------------------------------------------------------ spec ---


def spec_numbers():
    """Everything the handout quotes from the two spec files."""
    r, f = load_spec("r10000"), load_spec("firestorm")
    rb, fb = r["backend"], f["backend"]
    s = {
        "r_active_list": v(rb, "retire", "active_list", "value"),
        "r_int_prf": v(rb, "int_prf", "value"),
        "r_int_head": v(rb, "int_prf_headroom", "value"),
        "r_fp_prf": v(rb, "fp_prf", "value"),
        "r_fp_head": v(rb, "fp_prf_headroom", "value"),
        "r_width": v(rb, "rename_width", "value"),
        "r_grad": v(rb, "retire", "graduate_width", "value"),
        "r_aq": v(rb, "queues", "address_queue", "value"),
        "r_branch_stack": v(r, "frontend", "branch_stack", "value"),
        "f_rows": v(fb, "retire", "retire_groups", "value"),
        "f_rows_range": v(fb, "retire", "retire_groups", "range"),
        "f_group": v(fb, "retire", "group_max_uops", "value"),
        "f_failable": v(fb, "retire", "failable_per_group", "value"),
        "f_rrq": v(fb, "retire", "rename_retire_queue", "value"),
        "f_rrq_range": v(fb, "retire", "rename_retire_queue", "range"),
        "f_retire_rate": v(fb, "retire", "retire_rate", "value"),
        "f_nop": v(fb, "retire", "nop_window", "value"),
        "f_nop_range": v(fb, "retire", "nop_window", "range"),
        "f_int_prf": v(fb, "int_prf", "value"),
        "f_int_range": v(fb, "int_prf", "range"),
        "f_fp_prf": v(fb, "fp_prf", "value"),
        "f_flags": v(fb, "flags_prf", "value"),
        "f_flags_range": v(fb, "flags_prf", "range"),
        "f_lq": v(fb, "load_queue", "value"),
        "f_lq_probe": v(fb, "load_queue", "window_probe_expected"),
        "f_sq": v(fb, "store_queue", "value"),
        "f_sq_probe": v(fb, "store_queue", "window_probe_expected"),
        "f_width": v(fb, "rename_width", "value"),
        "f_dram": v(f, "memory", "dram_latency", "value"),
        "r_prov": v(rb, "retire", "active_list", "provenance"),
        "f_prov": v(fb, "retire", "retire_groups", "provenance"),
    }
    s["r_int_arch"] = s["r_int_prf"] - s["r_int_head"]
    s["r_fp_arch"] = s["r_fp_prf"] - s["r_fp_head"]
    # predicted R10K knees (toy offsets at depth 1: 2 for nop and int, 0 for fp)
    s["pred_nop"] = s["r_active_list"] - 2
    s["pred_int"] = min(s["r_int_head"] - 2, s["pred_nop"])
    s["pred_fp"] = min(s["r_fp_head"], s["pred_nop"])
    s["rows_x_group_lo"] = s["f_rows_range"][0] * s["f_group"]
    s["rows_x_group_hi"] = s["f_rows_range"][1] * s["f_group"]
    return s


# -------------------------------------------------------------- commands ---


def commands(s=None):
    """The ./sweep.py commands of the handout, generated from the specs."""
    r10k = resolve_spec(load_spec("r10000"), {"miss_latency": 200})["cfg"]
    al64 = resolve_spec(load_spec("r10000"), {"miss_latency": 200, "rob": 64})["cfg"]
    f = load_spec("firestorm")
    # retire rows as a classic ROB: resolve_spec refuses Firestorm's coalesced retire groups
    fs = (
        f"rob={v(f, 'backend', 'retire', 'retire_groups', 'value')},"
        f"width={v(f, 'backend', 'rename_width', 'value')},"
        f"miss_latency={v(f, 'memory', 'dram_latency', 'value')}"
    )
    return [
        {
            "id": "C1",
            "tag": "lab-r10k",
            "fillers": ["nop", "int", "fp"],
            "argv": [
                "toy",
                "--tag",
                "lab-r10k",
                "--toy",
                toy_arg(r10k),
                "--fillers",
                "nop,int,fp",
                "--n",
                "4:48:4",
            ],
        },
        {
            "id": "C2",
            "tag": "lab-r10k-al64",
            "fillers": ["nop", "int", "fp"],
            "argv": [
                "toy",
                "--tag",
                "lab-r10k-al64",
                "--toy",
                toy_arg(al64),
                "--fillers",
                "nop,int,fp",
                "--n",
                "8:96:4",
            ],
        },
        {
            "id": "C3",
            "tag": "lab-firestorm-classic",
            "fillers": ["nop"],
            "argv": [
                "toy",
                "--tag",
                "lab-firestorm-classic",
                "--toy",
                fs,
                "--fillers",
                "nop",
                "--n",
                "240:420:12",
            ],
        },
    ]


def command_line(c):
    return "./sweep.py " + " ".join(shlex.quote(a) for a in c["argv"])


def _pin():
    try:
        cpus = {c for c in os.sched_getaffinity(0) if c >= 4}  # CPUs 0-3 stay quiet for native timing
        if cpus:
            os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        pass


def run_commands(workdir, log=None):
    """Run every command concurrently as `python3 <repo>/sweep.py ...` with cwd=workdir.

    Returns {tag: {"knees": <knees.json>, "curve": {filler: {n: cycles_per_iter}}}}.
    """
    procs = []
    for c in commands():
        argv = [sys.executable, os.path.join(ROOT, "sweep.py"), *c["argv"]]
        procs.append(
            (
                c,
                subprocess.Popen(
                    argv,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=_pin,
                ),
            )
        )
    out = {}
    for c, p in procs:
        text, _ = p.communicate()
        if p.returncode:
            raise RuntimeError(f"{command_line(c)} failed:\n{text[-2000:]}")
        if log:
            log(f"{c['id']}: {command_line(c)}")
        d = os.path.join(workdir, "results", "toy", c["tag"])
        with open(os.path.join(d, "knees.json")) as fh:
            knees = json.load(fh)
        curve = {}
        with open(os.path.join(d, "sweep.csv")) as fh:
            for row in csv.DictReader(fh):
                curve.setdefault(row["filler"], {})[int(row["n"])] = float(row["cycles_per_iter"])
        out[c["tag"]] = {"knees": knees, "curve": curve}
    return out


STRUCT = {"nop": "rob", "int": "int_prf_headroom", "fp": "fp_prf_headroom"}


def lab_numbers(outputs):
    """Every toy-derived number the handout shows, keyed '<tag>.<filler>.<what>'."""
    nums = {}
    for c in commands():
        o = outputs[c["tag"]]
        for f in c["fillers"]:
            k = o["knees"]["knees"][f]["knee"]
            e = o["knees"]["estimates"][STRUCT[f]]
            key = f"{c['tag']}.{f}"
            nums[f"{key}.knee"] = int(k)
            nums[f"{key}.offset"] = e["offset"]["value"]
            nums[f"{key}.estimate"] = int(e["value"])
            nums[f"{key}.status"] = e["status"].split(":")[0]
            nums[f"{key}.flags"] = ",".join(e["anomalies"]) or "none"
            cv = o["curve"][f]
            nums[f"{key}.cpi_at_knee"] = f"{cv[int(k)]:.1f}"
            nums[f"{key}.cpi_after"] = f"{cv[int(k) + 1]:.1f}"
    return nums


# ---------------------------------------------------------------- render ---


def rng(r):
    return f"{r[0]}-{r[1]}"


def render(s, nums):
    n = lambda key: nums[key]
    C = {c["id"]: command_line(c) for c in commands()}
    r1, r2, r3 = "lab-r10k", "lab-r10k-al64", "lab-firestorm-classic"

    def row(tag, f, struct, note=""):
        k = f"{tag}.{f}"
        flags = n(f"{k}.flags")
        return (
            f"| {f} | {n(k + '.knee')} | {n(k + '.offset')} | {struct} = {n(k + '.estimate')} | "
            f"{n(k + '.status')}{'' if flags == 'none' else f' ({flags})'} | {n(k + '.cpi_at_knee')} → "
            f"{n(k + '.cpi_after')} |{note}"
        )

    fs_ratio = s["f_nop"] / n(f"{r3}.nop.estimate")
    g, (lo, hi) = s["f_group"], s["f_rows_range"]
    # Part 4 cells too long for the template
    f_retire = (
        f"{s['f_rows']} retire rows ({rng(s['f_rows_range'])}), up to {s['f_group']} uops each, "
        f"{s['f_failable']} failable op (load, store, branch) per row"
    )
    f_rrq = f"rename retire queue, ~{s['f_rrq']} slots ({rng(s['f_rrq_range'])})"
    f_int = f"{'unknown' if s['f_int_prf'] is None else s['f_int_prf']} (range {rng(s['f_int_range'])})"
    f_lsq = (
        f"~{s['f_lq']} / ~{s['f_sq']} (window probe with a miss delay: "
        f"~{s['f_lq_probe']} / ~{s['f_sq_probe']})"
    )
    th = (
        "| filler | knee N | offset | estimate | status | cycles/iter at knee → knee + 1 |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    return f"""# Lab: register renaming, from the R10000 to Firestorm

<!-- Generated by explainer/labs/build_lab.py; do not edit by hand. -->

Every number here comes from the two spec files or from a command you run. If
yours differs, the spec or the model changed.

You will learn:

- what renaming needs: a free physical register per result in flight, freed
  when the next writer of the same logical register retires;
- why the window probe measures rename headroom (physical minus logical
  registers), not the register file size;
- why two structures of nearly the same size give an ambiguous reading;
- how Firestorm splits the active list in two, and why its NOP window looks
  about {round(fs_ratio)} times larger than its retire structure.

Setup, from the repository root: `. tools/env.sh`. Each command below takes
under a minute, prints the knees and estimates, and writes `results/toy/<tag>/`
(`knees.json`, `sweep.csv`, `sweep.png`).

## Part 1. The R10000 rename machine

From `spec/cores/r10000.yaml`, every value `{s["r_prov"]}` (Yeager, "The MIPS
R10000 Superscalar Microprocessor"):

| structure | R10000 |
| --- | --- |
| logical integer registers (r1-r31, Hi, Lo) | {s["r_int_arch"]} |
| physical integer registers | {s["r_int_prf"]} |
| integer rename headroom = {s["r_int_prf"]} − {s["r_int_arch"]} | **{s["r_int_head"]}** |
| logical FP registers | {s["r_fp_arch"]} |
| physical FP registers | {s["r_fp_prf"]} |
| FP rename headroom = {s["r_fp_prf"]} − {s["r_fp_arch"]} | **{s["r_fp_head"]}** |
| active list (every instruction, NOPs included) | **{s["r_active_list"]}** |
| rename and graduate width (instructions per cycle) | {s["r_width"]} and {s["r_grad"]} |
| address queue (loads and stores) | {s["r_aq"]} |
| branch stack (map-table snapshots) | {s["r_branch_stack"]} |

An instruction that writes a register takes a free physical register at rename,
and the map table points the logical register at it. The old physical register
stays allocated, since older instructions may still read it and a mispredict
may need it back; it is freed when the overwriting instruction graduates. So the
{s["r_int_arch"]} registers holding committed state are always pinned, and at most {s["r_int_head"]} integer
results can be in flight.

**Exercise 1 (predict).** One iteration of the window probe is: miss load A,
N fillers, miss load B, N fillers, loop branch. The two misses overlap (one
miss latency per iteration) only while A, the fillers and B are all in flight.
Before running anything, predict the largest N for each filler:

- NOP fillers take only an active-list line: N + 2 ≤ {s["r_active_list"]}, so N = {s["pred_nop"]}.
- Integer fillers also take an integer register, as do both miss loads:
  N + 2 ≤ {s["r_int_head"]}, so N = {s["pred_int"]}.
- FP fillers take an FP register and the misses do not: N ≤ {s["r_fp_head"]}, but the
  active list stops them first, so N = {s["pred_fp"]}.

## Part 2. Measure it

```
{C["C1"]}
```

The `--toy` string is the R10000 spec mapped onto the toy model by
`resolve_spec` (`uarchre/backends/platform/toy_trace.py`): active list → `rob`,
logical integer registers → `arch_int`, L1 latency → `load_latency`, address
queue → both `lq` and `sq`. The spec has no DRAM latency, so misses take the
toy default of 200 cycles.

{th}
{row(r1, "nop", "rob")}
{row(r1, "int", "int_prf_headroom")}
{row(r1, "fp", "fp_prf_headroom")}

The knee is the last N on the low plateau; past it the misses serialize and an
iteration costs about two miss latencies instead of one. Estimate = knee +
offset, where the offset counts the miss loads that hold the same structure
(`uarchre/analysis/offsets.yaml`).

**Exercise 2.** The NOP reading gives the active list exactly. The integer
reading gives the headroom ({n(f"{r1}.int.estimate")}) but is flagged `tie.rob`: it is within 8 % of
the active-list estimate ({n(f"{r1}.nop.estimate")}), so from these curves alone the active list could
be what binds. In two sentences, say which structure really stops the integer
fillers and why the analysis is right not to claim it.

**Exercise 3.** The FP reading ({n(f"{r1}.fp.estimate")}) equals the NOP knee, not the FP headroom
({s["r_fp_head"]}). What is the most you can claim about the FP headroom from this run?

## Part 3. A longer active list

A what-if on the toy model, not a real R10000: double the active list and
change nothing else.

```
{C["C2"]}
```

{th}
{row(r2, "nop", "rob")}
{row(r2, "int", "int_prf_headroom")}
{row(r2, "fp", "fp_prf_headroom")}

**Exercise 4.** The integer and FP estimates are now {n(f"{r2}.int.estimate")} and
{n(f"{r2}.fp.estimate")}, the true headrooms, and neither is flagged. The register files
did not change; what did? Why can a window probe only ever measure the
smallest structure a filler consumes?

## Part 4. Scale up to Firestorm

From `spec/cores/firestorm.yaml`, values `{s["f_prov"]}` from published reverse
engineering, with a range where the sources disagree:

| | R10000 | Firestorm (Apple M1 P-core) |
| --- | --- | --- |
| rename width | {s["r_width"]} | {s["f_width"]} |
| in-order retire structure | active list, {s["r_active_list"]} lines | {f_retire} |
| record of mappings to free | inside the active list | {f_rrq} |
| integer physical registers | {s["r_int_prf"]} | {f_int} |
| FP physical registers | {s["r_fp_prf"]} | ~{s["f_fp_prf"]} |
| flags physical registers | none | ~{s["f_flags"]} ({rng(s["f_flags_range"])}) |
| NOP window | {s["r_active_list"]} (the active list) | ~{s["f_nop"]} ({rng(s["f_nop_range"])}) |
| load / store queue | {s["r_aq"]}-entry shared address queue | {f_lsq} |

The R10000's active list does two jobs: it orders retirement, and each line
remembers which old physical register to free. Firestorm splits them. Retire
rows order retirement, several uops to a row; the rename retire queue holds one
slot per renamed destination and remembers what to free.

**Exercise 5.** Treat Firestorm's {s["f_rows"]} retire rows as a classic ROB of {s["f_rows"]} entries
(rows → `rob`, rename width → `width`, DRAM latency → `miss_latency`, all from
the spec) and run the probe:

```
{C["C3"]}
```

{th}
{row(r3, "nop", "rob")}

The published NOP window is ~{s["f_nop"]}, {fs_ratio:.1f} times this estimate. A group size
of {g} explains it: {lo} rows × {g} = {s["rows_x_group_lo"]} and {hi} × {g} = {s["rows_x_group_hi"]}, against
a published range of {rng(s["f_nop_range"])}. Which end of the row range fits better? Why
would load or store fillers see roughly the row count instead (hint: failable
ops per row)?

**Exercise 6.** The spec's integer register count is a gap (`null`, range
{rng(s["f_int_range"])}). The window probe measures headroom, so even a perfect
integer knee leaves one number missing before you can recover the register
file size. Which number, and where could it come from?

## Check your numbers

`renaming-lab.json` lists the commands and every expected value;
`tests/test_lab.py` re-runs the commands and checks them.
"""


def main():
    def log(m):
        print(m, flush=True)

    base = os.path.join(ROOT, "results")
    with tempfile.TemporaryDirectory(prefix="lab-", dir=base if os.path.isdir(base) else None) as d:
        outputs = run_commands(d, log=log)
    s = spec_numbers()
    nums = lab_numbers(outputs)
    with open(MD, "w") as fh:
        fh.write(render(s, nums))
    with open(JSON, "w") as fh:
        json.dump(
            {
                "generated_by": "explainer/labs/build_lab.py",
                "handout": "explainer/labs/renaming-lab.md",
                "commands": [{"id": c["id"], "tag": c["tag"], "line": command_line(c)} for c in commands()],
                "numbers": nums,
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    md, js = os.path.relpath(MD, ROOT), os.path.relpath(JSON, ROOT)
    print(f"wrote {md} and {js}: {len(nums)} toy-derived numbers")


if __name__ == "__main__":
    main()
