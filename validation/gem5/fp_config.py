"""config_o3.py plus a FetchDirectedPrefetcher on the I-cache (P3.8).

The prefetcher needs registerMMU/registerCache calls that `--set` cannot make, so
this runs config_o3.py unchanged and adds them just before m5.instantiate().
Without --fdp it is config_o3.py.

  gem5.opt fp_config.py [--fdp] [--fdp-snoop 0|1] --cmd ./kernel ... --set cpu.decoupledFrontEnd=True
"""

import os
import sys

import m5
import m5.objects

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config_o3.py")

fdp = "--fdp" in sys.argv
snoop = True
argv = [a for a in sys.argv if a != "--fdp"]
if "--fdp-snoop" in argv:
    i = argv.index("--fdp-snoop")
    snoop = argv[i + 1] != "0"
    del argv[i : i + 2]
sys.argv = [CONFIG] + argv[1:]

g = {"__name__": "__m5_main__", "__file__": CONFIG}
_instantiate = m5.instantiate


def _hook(*a, **k):
    if fdp:
        system = g["system"]
        cpu, icache = system.cpu, system.cpu.icache
        pf = m5.objects.FetchDirectedPrefetcher(use_virtual_addresses=True, cpu=cpu, cache_snoop=snoop)
        pf.registerCache(icache)
        pf.registerMMU(cpu.mmu)
        icache.prefetcher = pf
    return _instantiate(*a, **k)


m5.instantiate = _hook
with open(CONFIG) as fh:
    exec(compile(fh.read(), CONFIG, "exec"), g)
