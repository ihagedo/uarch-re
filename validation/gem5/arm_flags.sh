#!/bin/bash
# P1.2 flags registers on gem5 ARM O3: sweep numPhysCCRegs with the flags filler (cmp x6, x7)
# at miss depth 1. gem5 renames NZ, C and V as three CC registers (6 mapped at reset), so the
# knee is (numPhysCCRegs - 6) / 3 cmps; ROB 512 keeps the ROB out of the way.
#   JOBS=7 validation/gem5/arm_flags.sh
cd "$(dirname "$0")/../.." || exit 1
. tools/env.sh
LOG="$UARCHRE_SCRATCH/logs/arm"
mkdir -p "$LOG"

CCS="${CCS:-48 72 96 144 192 288 384}"
ROB=512

for c in $CCS; do echo "$c"; done |
  xargs -P "${JOBS:-7}" -n 1 sh -c '
    c=$1
    k=$(( (c - 6) / 3 ))                       # expected knee in cmps
    lo=$(( k / 2 > 2 ? k / 2 : 2 )); hi=$(( k + 20 )); step=$(( k < 40 ? 2 : 4 ))
    validation/gem5/arm_sweep.py --tag "arm-flags-cc$c" --gem5 "numROBEntries='"$ROB"',numPhysCCRegs=$c" \
      --fillers flags --n "$lo:$hi:$step" --depth 1 --chain-mb 2 --iters 40 --jobs "${SWEEP_JOBS:-4}" \
      > "'"$LOG"'/flags-cc$c.log" 2>&1
    echo "cc=$c exit=$?"
  ' _
