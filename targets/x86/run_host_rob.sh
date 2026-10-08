#!/bin/bash
# ROB-only native sweep around the expected knee; see run_host.sh.
cd "$(dirname "$0")/../.."
PAT="scons|cc1plus"
trap 'pkill -u "$USER" -CONT -f "$PAT"' EXIT
pkill -u "$USER" -STOP -f "$PAT"
./sweep.py native --fillers nop --n 320:640:16 --chain-mb 32 --tag xeon-host-rob --depth 2
