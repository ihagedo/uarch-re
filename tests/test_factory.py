"""Factory engine (P6.3): trace export, resolver, golden suite, JS conformance.

  python3 -m unittest tests.test_factory

The Node tests skip without Node >= 18.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import fields

import yaml

from uarchre.backends.platform.toy import ToyConfig, run_toy
from uarchre.backends.platform.toy_trace import (
    CLASSIC_FIELDS,
    UnsupportedSpec,
    resolve_spec,
    toy_arg,
    trace_toy,
)
from uarchre.kernels.window import window_probe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FACTORY = os.path.join(ROOT, "explainer", "factory")
GOLDEN = os.path.join(FACTORY, "golden")


def find_node():
    for c in (os.environ.get("NODE"), shutil.which("node"), os.path.expanduser("~/opt/node/bin/node")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def load_spec(core):
    with open(os.path.join(ROOT, "spec", "cores", f"{core}.yaml")) as fh:
        return yaml.safe_load(fh)


def golden_headers():
    with open(os.path.join(GOLDEN, "index.json")) as fh:
        index = json.load(fh)
    for c in index["cases"]:
        with open(os.path.join(GOLDEN, c["file"])) as fh:
            yield json.loads(fh.readline())


def build_golden_module():
    """explainer/factory/build_golden.py as module `build_golden` (importable by Pool workers)."""
    if "build_golden" not in sys.modules:
        if FACTORY not in sys.path:
            sys.path.insert(0, FACTORY)
        spec = importlib.util.spec_from_file_location(
            "build_golden", os.path.join(FACTORY, "build_golden.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["build_golden"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["build_golden"]


R10K = resolve_spec(load_spec("r10000"), {"miss_latency": 200})["cfg"]


class TestTrace(unittest.TestCase):
    def test_traced_cycles_equal_run_toy_on_golden_cases(self):
        n = 0
        for h in golden_headers():
            k = h["kernel"]
            kernel = window_probe(
                k["filler"], k["n"], rotation=k["rotation"], depth=k["depth"], delay=k["delay"]
            )
            cfg = ToyConfig(**h["cfg"])
            ref = run_toy(kernel, cfg, iters=h["iters"], warm=h["warm"])
            tr = trace_toy(kernel, cfg, iters=h["iters"], warm=h["warm"])
            self.assertEqual(tr.cycles_per_iter, ref, h["case"])
            self.assertEqual(h["cycles_per_iter"], ref, f"{h['case']}: golden is stale")
            n += 1
        self.assertGreaterEqual(n, 9)

    def test_traced_cycles_equal_run_toy_at_sweep_length(self):
        cfg = ToyConfig(**R10K)
        for f in ("nop", "int", "fp", "load", "store", "mix"):
            for n in (14, 29, 30, 31):
                k = window_probe(f, n)
                self.assertEqual(trace_toy(k, cfg, events=False).cycles_per_iter, run_toy(k, cfg), (f, n))

    def test_jitter_draws_in_the_same_order(self):
        cfg = ToyConfig(**{**R10K, "miss_latency_jitter": 0.1, "seed": 3})
        k = window_probe("nop", 24, depth=2)
        self.assertEqual(
            trace_toy(k, cfg, iters=10, warm=2).cycles_per_iter, run_toy(k, cfg, iters=10, warm=2)
        )

    def test_non_classic_mechanisms_are_refused(self):
        extra = [f for f in fields(ToyConfig) if f.name not in CLASSIC_FIELDS]
        if not any(f.name == "retire_group" for f in extra):
            self.skipTest("toy.py has no retire_group field")
        with self.assertRaises(ValueError):
            trace_toy(window_probe("nop", 8), ToyConfig(retire_group=7), iters=2, warm=1)

    def test_every_op_has_an_ordered_life(self):
        cfg = ToyConfig(**R10K)
        tr = trace_toy(window_probe("int", 30), cfg, iters=3, warm=1)
        life = {}
        for e in tr.events:
            if e["ev"] != "stall":
                life.setdefault(e["seq"], {})[e["ev"]] = e["c"]
        retired = [s for s, d in life.items() if "retire" in d]
        self.assertEqual(retired, sorted(retired))  # in-order retire
        for s in retired:
            d = life[s]
            self.assertLess(d["dispatch"], d["issue"] + 1)
            self.assertLess(d["issue"], d["complete"])
            self.assertLessEqual(d["complete"], d["retire"])
        whys = {e["why"] for e in tr.events if e["ev"] == "stall"}
        self.assertIn("int_prf_empty", whys)  # N = 30 fillers + 2 misses > integer headroom 31


class TestResolver(unittest.TestCase):
    def test_r10k_maps_onto_the_toy(self):
        r = resolve_spec(load_spec("r10000"), {"miss_latency": 200})
        c = r["cfg"]
        self.assertEqual(
            (c["rob"], c["int_prf"] - c["arch_int"], c["fp_prf"] - c["arch_fp"], c["width"]), (32, 31, 32, 4)
        )
        self.assertEqual((c["lq"], c["sq"], c["load_latency"]), (16, 16, 2))
        self.assertEqual(r["prov"]["rob"], {"from": "backend.retire.active_list", "provenance": "documented"})
        self.assertEqual(r["prov"]["miss_latency"]["from"], "override")
        self.assertTrue(any("address queue" in a for a in r["assumptions"]))

    def test_missing_dram_latency_is_refused(self):
        with self.assertRaisesRegex(UnsupportedSpec, "dram_latency"):
            resolve_spec(load_spec("r10000"))

    def test_firestorm_needs_retire_groups(self):
        with self.assertRaisesRegex(UnsupportedSpec, "coalesced_retire_groups"):
            resolve_spec(load_spec("firestorm"), {"miss_latency": 300})

    def test_toy_arg_round_trips_through_sweep_py(self):
        s = toy_arg(R10K)
        self.assertEqual(s, "rob=32,int_prf=64,fp_prf=64,lq=16,sq=16,load_latency=2,arch_int=33")
        kv = dict(p.split("=") for p in s.split(","))
        self.assertEqual(ToyConfig(**{k: int(v) for k, v in kv.items()}), ToyConfig(**R10K))


class TestGolden(unittest.TestCase):
    def test_golden_suite_is_up_to_date(self):
        bg = build_golden_module()
        stale = bg.stale(bg.build())
        self.assertEqual(stale, [], "regenerate with: python3 explainer/factory/build_golden.py")


@unittest.skipUnless(find_node(), "needs Node >= 18 ($NODE, PATH, or ~/opt/node/bin/node)")
class TestNodeConformance(unittest.TestCase):
    def test_engine_matches_golden_suite(self):
        p = subprocess.run(
            [find_node(), os.path.join(FACTORY, "cli.mjs"), "conformance", "--json"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertIn(p.returncode, (0, 1), p.stderr)
        s = json.loads(p.stdout)
        self.assertEqual(s["problems"], [])
        self.assertTrue(s["pass"])
        self.assertEqual(s["events"]["identical"], s["events"]["total"])
        self.assertGreaterEqual(s["events"]["kernels"], 9)
        self.assertEqual(s["sweep"]["identical"], s["sweep"]["points"])
        self.assertEqual((s["milestone"]["nopKnee"], s["milestone"]["activeList"]), (30, 32))
        self.assertTrue(s["milestone"]["intAmbiguous"])

    def test_js_sweep_from_spec(self):
        p = subprocess.run(
            [
                find_node(),
                os.path.join(FACTORY, "cli.mjs"),
                "sweep",
                "--spec",
                "r10000",
                "--set",
                "miss_latency=200",
                "--filler",
                "nop",
                "--n",
                "26:34:1",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(p.stdout)
        cfg = ToyConfig(**R10K)
        self.assertEqual(r["ys"], [run_toy(window_probe("nop", n), cfg) for n in r["ns"]])
        self.assertEqual(r["knee"]["knee"], 30)

    def test_viewer_renders_the_near_tie(self):
        p = subprocess.run(
            [
                find_node(),
                os.path.join(FACTORY, "cli.mjs"),
                "render",
                "--spec",
                "r10000",
                "--set",
                "miss_latency=200",
                "--filler",
                "int",
                "--n",
                "30",
                "--cycles",
                "870",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(p.stdout)
        self.assertIn("Order ledger 31 / 32", r["floor"])
        self.assertIn("Int tote rack (rename headroom) 31 / 31", r["floor"])
        self.assertTrue(r["readout"].startswith("Line stopped: int tote rack empty"), r["readout"])
        self.assertTrue(r["sweep"].startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
