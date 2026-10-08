#!/bin/bash
# Unpinned ROB probe on a hybrid part: the scheduler picks CPU 2 (P) or CPU 14 (E) and each
# launch records the CPUs it ran on (CPU 0 stays out; other latency runs use it). Then the
# P1.8 follow-up micro tests on the P-core. Run after campaign_pcore.sh and campaign_ecore.sh.
#   targets/x86/campaign_float.sh [PID to wait for first]
cd "$(dirname "$0")/../.." && . tools/env.sh
LOGS=/var/tmp/$USER/uarch-re/logs/x86; mkdir -p "$LOGS"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "== $(date +%T) x86-float-miss-d1"
python3 targets/x86/window_campaign.py --cpu 2,14 --core float --trace --tag x86-float-miss-d1 \
    --fillers nop --n 16:640:16 --min-share 0 --retries 20 --quiet-timeout 900 --no-wait-own \
    > "$LOGS/x86-float-miss-d1.log" 2>&1
echo "== $(date +%T) x86-micro-p-extra"
python3 targets/x86/micro_campaign.py --cpu 2 --tag x86-micro-p-extra --tests p18x > "$LOGS/x86-micro-p-extra.log" 2>&1
echo "== $(date +%T) done"
