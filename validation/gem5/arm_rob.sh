#!/bin/bash
# P1.1 reorder window on gem5 ARM O3, the data for the aarch64 offset calibration (P0.6):
# p11_rob.sh on ARM. The grid starts at max(ROB/2, ROB-48) to save simulation time; the
# knee fit still sees >= 10 plateau points.
#   JOBS=7 validation/gem5/arm_rob.sh            # 7 sweeps at once, each --jobs 4
cd "$(dirname "$0")/../.." || exit 1
. tools/env.sh
LOG="$UARCHRE_SCRATCH/logs/arm"
mkdir -p "$LOG"

ROBS="${ROBS:-32 48 64 96 128 192 256}"
DEPTHS="${DEPTHS:-1 2}"
CHAIN_MB=2      # 2 chains x 2 MiB at 128 B per node = 16x the 256 kB L2. Building the
                # chains dominates simulated time, so 2 MiB halves the cost of a point
ITERS=40        # 2 reps x 40 iters x depth nodes stay far below 16k nodes per chain

for d in $DEPTHS; do for r in $ROBS; do echo "$r $d"; done; done |
  xargs -P "${JOBS:-7}" -n 2 sh -c '
    r=$1 d=$2
    lo=$(( r / 2 > r - 48 ? r / 2 : r - 48 )); lo=$(( lo > 2 ? lo : 2 )); hi=$(( r + 24 ))
    validation/gem5/arm_sweep.py --tag "arm-rob$r-d$d" --gem5 "numROBEntries=$r" --fillers nop \
      --n "$lo:$hi:4" --depth "$d" --chain-mb '"$CHAIN_MB"' --iters '"$ITERS"' --jobs "${SWEEP_JOBS:-4}" \
      > "'"$LOG"'/rob$r-d$d.log" 2>&1
    echo "rob=$r depth=$d exit=$?"
  ' _
