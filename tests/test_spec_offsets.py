"""Offsets lookup, attribution, spec writing and the schema rules.

python3 -m unittest tests.test_spec_offsets
"""

import copy
import json
import os
import unittest

import jsonschema

from uarchre.analysis import offsets
from uarchre.analysis.knee import Knee, attribute
from uarchre.spec import spec_from_estimates
from tools.validate_specs import migrate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(ROOT, "spec", "schema", "core.schema.json")) as fh:
    VALIDATOR = jsonschema.Draft202012Validator(json.load(fh))


def knee(k, ci=None, **kw):
    return Knee(float(k), ci or [float(k), float(k)], 2.0, 0.99, **kw)


class TestOffsets(unittest.TestCase):
    def test_exact_entry_wins(self):
        o = offsets.lookup("nop", {"emitter": "riscv64", "platform": "gem5", "depth": 2, "delay": "miss"})
        self.assertEqual(
            (o["value"], o["calibrated"], o["calibration"]), (3, True, "riscv64.gem5.nop.d2.miss")
        )

    def test_fallback_is_the_toy_offset(self):
        o = offsets.lookup("int", {"emitter": "riscv64", "platform": "gem5", "depth": 2, "delay": "miss"})
        self.assertEqual((o["value"], o["calibrated"]), (2, False))
        self.assertEqual(o["calibration"], "analytic:abstract.toy.int.d2.miss")

    def test_toy_offsets_are_not_two_times_depth(self):
        got = {d: offsets.offset_analytic("nop", d)["offset"] for d in (1, 2, 3)}
        self.assertEqual(got, {1: 2, 2: 3, 3: 3})

    def test_unknown_depth_raises(self):
        with self.assertRaises(KeyError):
            offsets.offset_analytic("nop", 7)


class TestAttribute(unittest.TestCase):
    GEM5_D2 = {"emitter": "riscv64", "platform": "gem5", "depth": 2, "delay": "miss"}

    def test_gem5_depth2_is_exact(self):
        e = attribute({"nop": knee(61)}, self.GEM5_D2)["rob"]
        self.assertEqual((e["value"], e["ci"], e["anomalies"]), (64.0, [64.0, 64.0], []))

    def test_uncalibrated_offset_is_an_info_finding(self):
        est = attribute({"nop": knee(125), "int": knee(60)}, self.GEM5_D2)
        self.assertEqual(est["int_prf_headroom"]["anomalies"], ["offset.uncalibrated"])
        spec = spec_from_estimates("t", "riscv64", est, "x.csv")
        self.assertEqual(spec["backend"]["int_prf_headroom"]["value"], 62.0)  # info keeps the value
        self.assertEqual(list(VALIDATOR.iter_errors(spec)), [])

    def test_offset_sd_widens_ci(self):
        table = copy.deepcopy(offsets.load())
        table.append(
            {
                "id": "x",
                "emitter": "e",
                "platform": "p",
                "filler": "nop",
                "depth": 1,
                "delay": "miss",
                "offset": 2,
                "offset_sd": 0.5,
            }
        )
        e = attribute({"nop": knee(60, [59, 61])}, {"emitter": "e", "platform": "p"}, table)["rob"]
        self.assertEqual(e["ci"], [60.0, 64.0])

    def test_near_tie_becomes_a_bound(self):
        est = attribute({"nop": knee(94), "int": knee(92)})
        self.assertIn("tie.rob", est["int_prf_headroom"]["anomalies"])
        f = spec_from_estimates("t", "riscv64", est, "x.csv")["backend"]["int_prf_headroom"]
        self.assertEqual((f["value"], f["bound"]), (None, {"lower": 94.0}))
        self.assertNotIn("ci_method", f)

    def test_shape_anomaly_becomes_a_range_bound(self):
        est = attribute({"nop": knee(60, [58, 63], anomaly="poor fit", flags=["shape.poor_fit"])})
        spec = spec_from_estimates("t", "riscv64", est, "x.csv")
        self.assertEqual(spec["backend"]["rob_entries"]["bound"], {"lower": 60.0, "upper": 65.0})
        self.assertEqual(list(VALIDATOR.iter_errors(spec)), [])


def spec_with(**f):
    return {
        "core": "t",
        "isa": "riscv64",
        "frontend": {},
        "decoupling": {},
        "memory": {},
        "backend": {"rob_entries": {"provenance": "measured", "evidence": "x", **f}},
    }


class TestSchema(unittest.TestCase):
    def errors(self, spec):
        return [e.message for e in VALIDATOR.iter_errors(spec)]

    def test_measured_value_needs_ci_method(self):
        self.assertTrue(self.errors(spec_with(value=64, ci=[64, 64])))
        self.assertEqual(self.errors(spec_with(value=64, ci=[64, 64], ci_method="bootstrap")), [])

    def test_bound_needs_null_value_and_anomalies(self):
        self.assertTrue(self.errors(spec_with(value=None, bound={"lower": 90})))
        self.assertTrue(
            self.errors(
                spec_with(
                    value=90, ci=[90, 90], ci_method="bracket", bound={"lower": 90}, anomalies=["tie.rob"]
                )
            )
        )
        self.assertEqual(self.errors(spec_with(value=None, bound={"lower": 90}, anomalies=["tie.rob"])), [])

    def test_offset_needs_calibration(self):
        self.assertTrue(self.errors(spec_with(value=None, offset={"value": 2})))

    def test_old_specs_migrate_with_a_warning(self):
        old = spec_with(value=64, ci=[64, 64])
        self.assertEqual(list(migrate(old)), ["backend/rob_entries"])
        self.assertEqual(self.errors(old), [])


if __name__ == "__main__":
    unittest.main()
