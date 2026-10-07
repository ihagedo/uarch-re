#!/bin/bash
# Build Chipyard Verilator simulators for BOOM configs (after setup.sh), default SmallBoomV3Config.
#   validation/boom/build.sh MediumBoomV3Config LargeBoomV3Config
#   JOBS=2 validation/boom/build.sh MegaBoomV3Config
# Configs build one at a time: elaboration takes up to 8 GB and each compile job ~1-2 GB.
set -e
USER=${USER:-$(id -un)}
CHIPYARD=${CHIPYARD:-/var/tmp/$USER/chipyard}
LOG=${UARCHRE_SCRATCH:-/var/tmp/$USER/uarch-re}/logs/boom
JOBS=${JOBS:-4}
CPUS=${UARCHRE_GEM5_CPUS:-4-19}
mkdir -p "$LOG"
source "$CHIPYARD/env.sh"
for cfg in "${@:-SmallBoomV3Config}"; do
  echo "$(date -Is) building $cfg (-j$JOBS), log $LOG/build-$cfg.log"
  taskset -c "$CPUS" /usr/bin/time -v make -C "$CHIPYARD/sims/verilator" CONFIG="$cfg" -j"$JOBS" VERILATOR_THREADS=1 \
    > "$LOG/build-$cfg.log" 2>&1
  ls -la "$CHIPYARD/sims/verilator/simulator-chipyard.harness-$cfg"
  echo "$(date -Is) done $cfg"
done
