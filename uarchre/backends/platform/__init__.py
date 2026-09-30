"""Platform drivers: given a Kernel, return cycles per iteration.

native  compile with the host gcc and run
gem5    cross-compile a static binary and run it under gem5 SE mode
boom    bare-metal RISC-V on Chipyard BOOM under Verilator
toy     execute the abstract op stream on the Python OoO model
"""
