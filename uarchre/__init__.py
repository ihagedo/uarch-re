"""uarch-re: black-box microarchitecture reverse engineering.

kernels   ISA-independent experiment definitions (abstract op streams)
backends  ISA emitters (x86-64, RISC-V, AArch64) and platform drivers (native, gem5, BOOM, toy)
analysis  knee detection, bootstrap CIs, calibrated offsets, anomaly rules
spec      provenance-tagged core YAML
"""

import sys

if sys.version_info < (3, 10):  # noqa: UP036 (stops a 3.9 interpreter with a message)
    raise SystemExit(
        "uarchre needs Python 3.10 or newer (macOS: python.org or Homebrew, not the Xcode python3)"
    )
