"""AArch64 emitter (GCC / clang inline asm), P0.6.

M1 rules (Handley): the loop head writes every register the loop touches, and NZCV (unclaimed
registers run 4-wide adds, not 6-wide); no adr/adrp (eliminated at rename from A15 on); filler values
are arbitrary 64-bit patterns and FP sources normal doubles; loads and stores use 8 non-sequential
addresses; one chain node per 128 B line. x18, x29 and x30 are never touched.
Counters: `pmccntr` (default; gem5 SE with validation/gem5/arm_config_o3.py; traps at EL0 on Linux
and macOS), `cntvct` (Linux) and `mach` (macOS). The last two tick at 24 MHz; knees in N do not care.
"""

from ..isa_base import Emitter
from ...kernels import Op

INT_ROT = [f"x{i}" for i in range(9, 17)]  # int / load filler destinations; x16 is safe, no calls
FP_ROT = [f"d{i}" for i in range(16, 24)]  # fp filler destinations
# 8-byte slots of the 128 B scratch buffer, in a fixed non-sequential order
OFFSETS = [8 * k for k in (5, 1, 12, 9, 14, 3, 7, 10)]
# arbitrary 64-bit patterns for the filler sources and the int delay chains
SRC_A, SRC_B = 0x5BD1E9955BD1E995, 0x1B8735939E3779B9
DLY_A, DLY_B = 0xC2B2AE3D27D4EB4F, 0x9E3779B97F4A7C15

COUNTERS = {
    "pmccntr": r"""
/* PMCCNTR_EL0: core cycles under gem5 SE with validation/gem5/arm_config_o3.py */
__attribute__((constructor)) static void uarchre_pmu_on(void) {
    uint64_t c0, c1;
    __asm__ __volatile__("msr pmcr_el0, %0\n\tmsr pmcntenset_el0, %1\n\tisb"
                         :: "r"((uint64_t)0x45), "r"((uint64_t)1 << 31) : "memory");
    __asm__ __volatile__("isb\n\tmrs %0, pmccntr_el0" : "=r"(c0) :: "memory");
    for (volatile int i = 0; i < 64; i++) ;
    __asm__ __volatile__("isb\n\tmrs %0, pmccntr_el0" : "=r"(c1) :: "memory");
    if (c1 == c0) {
        printf("error: PMCCNTR_EL0 does not count; under gem5 use validation/gem5/arm_config_o3.py "
               "(it attaches an ArmPMU)\n");
        exit(2);
    }
}
static inline uint64_t cycles(void) {
    uint64_t c; __asm__ __volatile__("isb\n\tmrs %0, pmccntr_el0\n\tisb" : "=r"(c) :: "memory"); return c;
}
""",
    "cntvct": r"""
/* CNTVCT_EL0: the generic timer (24 MHz on Apple silicon), not core cycles.
   Ticks are scaled by TICK_SCALE so the harness's %.4f / %.3f output keeps
   ~5 significant digits; counter_hz is reported in the same scaled ticks. */
#define TICK_SCALE 1024
__attribute__((constructor)) static void uarchre_counter_hz(void) {
    uint64_t f; __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
    printf("counter_hz %llu\n", (unsigned long long)f * TICK_SCALE);
}
static inline uint64_t cycles(void) {
    uint64_t c; __asm__ __volatile__("isb\n\tmrs %0, cntvct_el0\n\tisb" : "=r"(c) :: "memory");
    return c * TICK_SCALE;
}
""",
    "mach": r"""
#include <mach/mach_time.h>
#include <pthread.h>
/* mach_absolute_time: the 24 MHz system counter on Apple silicon, not core cycles.
   The constructor asks for a performance core (QoS is a hint: macOS has no hard
   affinity on Apple silicon, so the driver checks the clock probe's GHz) and
   reports the counter rate from the timebase. Ticks are scaled by TICK_SCALE so
   the harness's %.4f / %.3f output keeps ~5 significant digits (0.0075 ticks
   per add unscaled); counter_hz is reported in the same scaled ticks. */
#define TICK_SCALE 1024
__attribute__((constructor)) static void uarchre_macos_setup(void) {
    mach_timebase_info_data_t tb;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    mach_timebase_info(&tb);
    printf("counter_hz %.1f\n", TICK_SCALE * 1e9 * (double)tb.denom / (double)tb.numer);
    printf("qos_class %d\n", (int)qos_class_self());
}
static inline uint64_t cycles(void) {
    uint64_t c;
    __asm__ __volatile__("isb" ::: "memory");
    c = mach_absolute_time() * TICK_SCALE;
    __asm__ __volatile__("isb" ::: "memory");
    return c;
}
""",
}


def _mov64(reg, value):
    """movz/movk sequence for a 64-bit constant (no literal pool, no adr)."""
    out = [f"movz {reg}, #{value & 0xFFFF}"]
    for sh in (16, 32, 48):
        out.append(f"movk {reg}, #{(value >> sh) & 0xFFFF}, lsl #{sh}")
    return out


class AArch64(Emitter):
    name = "aarch64"
    line = 128
    default_bytes = 256 << 20
    clobbers = (
        ["x6", "x7", "x8"] + INT_ROT + ["x19", "x20"] + [f"v{i}" for i in range(16, 26)] + ["v30", "v31"]
    )

    def __init__(self, counter="pmccntr"):
        if counter not in COUNTERS:
            raise ValueError(f"counter must be one of {sorted(COUNTERS)}")
        self.counter = counter
        self.cycle_counter_c = COUNTERS[counter]
        # gem5 pays simulation time per add; a 24 MHz timer needs a long probe
        self.default_probe_adds = 1 << 12 if counter == "pmccntr" else 1 << 22

    def loop_head(self):
        head = []
        head += _mov64("x6", SRC_A) + _mov64("x7", SRC_B)
        head += ["mov x8, #1"]  # udiv by 1: full 64-bit quotient, the slow case for an early-out divider
        head += _mov64("x19", DLY_A) + _mov64("x20", DLY_B)  # int delay chains
        head += [f"mov {r}, x6" for r in INT_ROT]  # claim the destinations
        head += ["scvtf d30, x6", "scvtf d31, x7"]  # normal, non-trivial FP sources
        head += [f"fmov {d}, d30" for d in FP_ROT]
        head += ["fmov d24, #2.0", "fmov d25, #2.0"]  # fsqrt delay chains
        head += [f"stp x6, x7, [%3, #{o}]" for o in range(0, 128, 16)]  # loaded values
        head += ["cmp x6, x7"]  # claim NZCV
        return head + [".p2align 6", "1:"]

    def loop_tail(self):
        return ["subs %2, %2, #1", "b.ne 1b"]

    def probe_add(self):
        return "add %0, %0, %1"  # x += n: reg-reg, never a constant value

    def probe_tail(self):
        return ["subs %1, %1, #1", "b.ne 1b"]

    def miss(self, chain):
        return f"ldr %{chain}, [%{chain}]"

    def filler(self, op, idx):
        if op == Op.NOP:
            return "nop"
        if op == Op.INT:
            return f"add {INT_ROT[idx % 8]}, x6, x7"
        if op == Op.FP:
            return f"fadd {FP_ROT[idx % 8]}, d30, d31"
        if op == Op.LOAD:
            return f"ldr {INT_ROT[idx % 8]}, [%3, #{OFFSETS[idx % 8]}]"
        if op == Op.STORE:
            return f"str x6, [%3, #{OFFSETS[idx % 8]}]"
        if op == Op.BRANCH:
            return "cbnz xzr, 3f\\n\\t3:"
        if op == Op.FLAGS:
            return "cmp x6, x7"
        raise ValueError(f"aarch64 has no {op} filler")

    def delay(self, op, chain):
        if op == Op.DELAY_FP:
            return f"fsqrt d{24 + chain}, d{24 + chain}"
        return f"udiv x{19 + chain}, x{19 + chain}, x8"
