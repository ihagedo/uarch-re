"""BOOM RTL driver (P0.5): bare-metal harness, ground truth, Spike smoke run.

python3 -m unittest tests.test_boom
UARCHRE_BOOM_SIM=1 python3 -m unittest tests.test_boom    # also one RTL run (~1-3 min)
"""

import os
import re
import subprocess
import unittest

from uarchre.backends.isa_base import parse_output
from uarchre.backends.platform import boom
from uarchre.kernels.window import window_probe

SPIKE = os.path.join(boom.RISCV, "bin", "spike")
MIXINS = os.path.join(
    boom.CHIPYARD, "generators", "boom", "src", "main", "scala", "v3", "common", "config-mixins.scala"
)
SCALA_KEYS = {
    "rob": "numRobEntries",
    "decode": "decodeWidth",
    "fetch": "fetchWidth",
    "int_prf": "numIntPhysRegisters",
    "fp_prf": "numFpPhysRegisters",
    "lq": "numLdqEntries",
    "sq": "numStqEntries",
    "fetch_buffer": "numFetchBufferEntries",
}


class TestHarness(unittest.TestCase):
    def test_layout_never_wraps(self):
        k = window_probe("nop", 16, depth=2)
        nodes, slot = boom.chain_layout(k, iters=100, reps=3)
        self.assertGreater(nodes, 3 * 100 * 2)
        self.assertGreaterEqual(nodes * slot * 64, 2 << 20)  # spans >= 4x the 512 KiB L2
        self.assertGreaterEqual(slot, 4)
        with self.assertRaises(ValueError):
            boom.chain_layout(k, iters=1 << 20, reps=3)  # does not fit the chain span

    def test_emit(self):
        k = window_probe("nop", 4)
        src = boom.emit_baremetal(k, iters=10, reps=2)
        self.assertIn("rep %d cycles_per_iter", src)
        self.assertIn("rdcycle", src)
        self.assertIn(".uarchre_chains", src)
        self.assertNotIn("malloc", src)
        self.assertIn(hex(boom.L2_FLUSH32), boom.emit_baremetal(k, iters=10, reps=2, chain="flush"))
        self.assertIn("#define L2_FLUSH32 0ULL", boom.emit_baremetal(k, iters=10, reps=2, chain="cached"))
        with self.assertRaises(ValueError):
            boom.emit_baremetal(k, chain="bogus")

    def test_preload_chain_is_one_sparse_cycle(self):
        """The preloaded chain visits every node once before returning, no two
        nodes share or neighbour a line, and a seed changes the layout."""
        k = window_probe("nop", 4)
        nodes, slot = boom.chain_layout(k, iters=10, reps=2)
        src = boom.emit_baremetal(k, iters=10, reps=2)
        for arr in ("chain_a", "chain_b"):
            links = dict(
                (int(a), int(b)) for a, b in re.findall(rf"\[(\d+)\] = \(uintptr_t\)&{arr}\[(\d+)\]", src)
            )
            self.assertEqual(len(links), nodes)
            start = int(re.search(rf"\*\w = \(void \*\)&{arr}\[(\d+)\]", src).group(1))
            seen, w = [], start
            for _ in range(nodes):
                seen.append(w)
                w = links[w]
            self.assertEqual(w, start)
            self.assertEqual(len(set(seen)), nodes)
            lines = sorted(x // 8 for x in seen)
            self.assertTrue(all(b - a >= 2 for a, b in zip(lines, lines[1:])))
        self.assertNotEqual(src, boom.emit_baremetal(k, iters=10, reps=2, seed=1))

    def test_parse(self):
        out = (
            "clock_ratio_pre 1.0012\nrep 0 cycles_per_iter 312.450\nrep 1 cycles_per_iter 310.010\n"
            "clock_ratio_post 1.0012\nbuild_cycles 1 nodes 2 slot_lines 3\ncycles_per_iter 310.010\n"
        )
        r = parse_output(out)
        self.assertEqual(r.cycles, [312.45, 310.01])
        self.assertEqual(r.clock_ratio_pre, 1.0012)

    @unittest.skipUnless(os.path.exists(MIXINS), "needs the Chipyard tree (validation/boom/setup.sh)")
    def test_truth_matches_scala(self):
        with open(MIXINS) as fh:
            src = fh.read()
        for cfg, p in boom.BOOM_V3.items():
            m = re.search(rf"class {p['mixin']}\(.*?\n\)\n", src, re.S)
            self.assertIsNotNone(m, p["mixin"])
            for key, scala in SCALA_KEYS.items():
                v = re.search(rf"\b{scala}\s*=\s*(\d+)", m.group(0))
                self.assertEqual(int(v.group(1)), p[key], f"{cfg} {scala}")

    def test_truth_matches_rtl(self):
        """numRobEntries is what the elaborated RTL has (rob_val bits in Rob.sv)."""
        built = [c for c in boom.BOOM_V3 if boom.rtl_rob_entries(c) is not None]
        if not built:
            self.skipTest("no BOOM config generated yet (validation/boom/build.sh)")
        for c in built:
            self.assertEqual(boom.rtl_rob_entries(c), boom.BOOM_V3[c]["rob"], c)

    @unittest.skipUnless(os.path.exists(boom.CC) and os.path.exists(SPIKE), "needs Chipyard's RISC-V tools")
    def test_spike_functional(self):
        """The bare-metal binary runs to completion on Spike (functional only;
        Spike has no L2 flush device, so only the preload mode runs there)."""
        k = window_probe("nop", 8)
        elf = boom.compile_baremetal(boom.emit_baremetal(k, iters=50, reps=2), k.name)
        out = subprocess.run(
            [SPIKE, "--isa=rv64gc_zicntr", elf], capture_output=True, text=True, timeout=60
        ).stdout
        r = parse_output(out)
        self.assertEqual(len(r.cycles), 2)
        self.assertAlmostEqual(r.clock_ratio_pre, 1.0, delta=0.1)

    @unittest.skipUnless(
        os.environ.get("UARCHRE_BOOM_SIM") and os.path.exists(boom.sim_path("SmallBoomV3Config")),
        "slow: set UARCHRE_BOOM_SIM=1 with SmallBoomV3Config built",
    )
    def test_rtl_run(self):
        r = boom.run_boom_reps(window_probe("nop", 8), "SmallBoomV3Config", iters=40, reps=2)
        self.assertEqual(len(r.cycles), 2)
        self.assertGreater(r.best, 30)  # a DRAM miss, not an L1 hit


if __name__ == "__main__":
    unittest.main()
