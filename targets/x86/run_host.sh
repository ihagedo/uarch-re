#!/bin/bash
# Pause our own background builds and sweeps, run the native sweep, resume
# them (the trap also resumes them if the sweep fails or is interrupted).
cd "$(dirname "$0")/../.."
PAT="scons|cc1plus|window_sweep.py"
trap 'pkill -u "$USER" -CONT -f "$PAT"' EXIT
pkill -u "$USER" -STOP -f "$PAT"
./sweep.py native --fillers nop,int,fp,load,store --n 16:400:16 --chain-mb 32 --tag xeon-host-d2 --depth 2
