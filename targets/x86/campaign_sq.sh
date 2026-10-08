#!/bin/bash
# SQ follow-up (P0.4 / P5.1): is the 2-5 entry SQ undercount caused by the
# store filler hitting one address? Same probe with stores spread over 6 words
# of one line and over 2 lines, P-core and E-core in parallel.
cd "$(dirname "$0")/../.." && . tools/env.sh
LOGS=/var/tmp/$USER/uarch-re/logs/x86; mkdir -p "$LOGS"
python3 targets/x86/window_campaign.py --cpu 2 --core p --retries 20 --quiet-timeout 900 --tag x86-p-sq-addr \
    --fillers store,v:store6w,v:store6l --n 16:240:16 --outer-reps 3 > "$LOGS/x86-p-sq-addr.log" 2>&1 &
python3 targets/x86/window_campaign.py --cpu 14 --core e --iters 2000 --inner-reps 9 --min-share 0 \
    --quiet-timeout 0.5 --tag x86-e-sq-addr --fillers store,v:store6w,v:store6l --n 8:120:8 --outer-reps 3 \
    > "$LOGS/x86-e-sq-addr.log" 2>&1 &
wait
echo "== $(date +%T) done"
