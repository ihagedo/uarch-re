#!/bin/bash
# P1.1 reorder window on gem5 O3 (RISC-V): nop fillers across ROB sizes at miss depth 1 and 2.
# The knees are the data for the gem5 offsets in uarchre/analysis/offsets.yaml (P0.1).
#   JOBS=12 validation/gem5/p11_rob.sh
cd "$(dirname "$0")/../.." || exit 1
. tools/env.sh
LOG="$UARCHRE_SCRATCH/logs/p11"
mkdir -p "$LOG"

ROBS="32 48 64 96 128 192 256"
DEPTHS="1 2"
CHAIN_MB=4      # 2 chains x 4 MB = 32x the 256 kB L2: nearly every chase misses
ITERS=40        # reps (2) x iters x depth nodes stay far below the 64k nodes per chain

for d in $DEPTHS; do for r in $ROBS; do echo "$r $d"; done; done |
  xargs -P "${JOBS:-12}" -n 2 sh -c '
    r=$1 d=$2
    lo=$(( r / 2 > 2 ? r / 2 : 2 )); hi=$(( r + 24 ))
    ./sweep.py gem5 --tag "p11-rob$r-d$d" --gem5 "numROBEntries=$r" --fillers nop \
      --n "$lo:$hi:4" --depth "$d" --chain-mb '"$CHAIN_MB"' --iters '"$ITERS"' \
      > "'"$LOG"'/rob$r-d$d.log" 2>&1
    echo "rob=$r depth=$d exit=$?"
  ' _
