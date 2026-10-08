"""Renaming lab (P6.4): the handout's commands reproduce every number it shows.

  python3 -m unittest tests.test_lab      # ~20 s: runs the three ./sweep.py toy commands

They run in a scratch directory, so a student's results/toy/lab-* are left alone.
"""

import importlib.util
import json
import os
import re
import shlex
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABS = os.path.join(ROOT, "explainer", "labs")


def build_lab():
    if "build_lab" not in sys.modules:
        spec = importlib.util.spec_from_file_location("build_lab", os.path.join(LABS, "build_lab.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["build_lab"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["build_lab"]


class TestLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bl = build_lab()
        with open(os.path.join(LABS, "renaming-lab.md")) as fh:
            cls.md = fh.read()
        with open(os.path.join(LABS, "renaming-lab.json")) as fh:
            cls.lab = json.load(fh)
        base = os.path.join(ROOT, "results")
        with tempfile.TemporaryDirectory(prefix="test-lab-", dir=base if os.path.isdir(base) else None) as d:
            cls.outputs = cls.bl.run_commands(d)
        cls.nums = cls.bl.lab_numbers(cls.outputs)

    def test_commands_are_in_the_handout_verbatim(self):
        blocks = re.findall(r"```\n(\./sweep\.py [^\n]+)\n```", self.md)
        listed = [c["line"] for c in self.lab["commands"]]
        self.assertEqual(blocks, listed)
        self.assertEqual(listed, [self.bl.command_line(c) for c in self.bl.commands()])
        for line in listed:  # what the test ran is what the student types
            self.assertEqual(shlex.split(line)[0], "./sweep.py")

    def test_every_toy_number_reproduces(self):
        expected = self.lab["numbers"]
        self.assertEqual(set(self.nums), set(expected))
        wrong = {k: (self.nums[k], expected[k]) for k in expected if self.nums[k] != expected[k]}
        self.assertEqual(wrong, {}, "rebuild with explainer/labs/build_lab.py and review the diff")
        self.assertGreaterEqual(len(expected), 40)

    def test_the_teaching_points_hold(self):
        n = self.nums
        self.assertEqual((n["lab-r10k.nop.knee"], n["lab-r10k.nop.estimate"]), (30, 32))
        self.assertEqual((n["lab-r10k.int.estimate"], n["lab-r10k.int.status"]), (31, "ambiguous"))
        self.assertEqual((n["lab-r10k.fp.estimate"], n["lab-r10k.fp.status"]), (30, "ambiguous"))
        self.assertEqual((n["lab-r10k-al64.int.estimate"], n["lab-r10k-al64.int.status"]), (31, "measured"))
        self.assertEqual((n["lab-r10k-al64.fp.estimate"], n["lab-r10k-al64.fp.status"]), (32, "measured"))
        self.assertEqual(n["lab-firestorm-classic.nop.estimate"], 330)

    def test_handout_rerenders_byte_for_byte(self):
        self.assertEqual(
            self.bl.render(self.bl.spec_numbers(), self.nums),
            self.md,
            "renaming-lab.md is stale: rebuild with explainer/labs/build_lab.py",
        )


if __name__ == "__main__":
    unittest.main()
