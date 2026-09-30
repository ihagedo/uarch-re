"""ISA emitters: lower an abstract Kernel to a self-contained C file with an inline-asm loop.

Each ISA module subclasses isa_base.Emitter: miss(), filler(), delay(), loop_head/loop_tail,
probe_add/probe_tail, `clobbers` and `cycle_counter_c` (defines `cycles()`).
"""

from ..isa_base import Emitter, emit_c  # noqa: F401
