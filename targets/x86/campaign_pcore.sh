#!/bin/bash
# P0.4 / P5.1 / P1.6 / P1.8 / P4.5 runs on the P-core (CPU 2), one at a time. A busy SMT
# sibling (CPU 3) halves the ROB (static partitioning), so each launch waits for an idle
# sibling and is retried if it got busy. With a PID, waits for it and skips the first three runs.
#   targets/x86/campaign_pcore.sh [PID to wait for first]
cd "$(dirname "$0")/../.." && . tools/env.sh
LOGS=/var/tmp/$USER/uarch-re/logs/x86; mkdir -p "$LOGS"
W="python3 targets/x86/window_campaign.py --cpu 2 --core p --retries 20 --quiet-timeout 900"
run() { local tag=$1; shift; echo "== $(date +%T) $tag"; "$@" > "$LOGS/$tag.log" 2>&1 || echo "FAILED $tag"; }
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 10; done

[ -n "$1" ] || run x86-footprint-p python3 targets/x86/footprint.py --cpus 2 --tag x86-footprint-p
[ -n "$1" ] || run x86-micro-p     python3 targets/x86/micro_campaign.py --cpu 2 --tag x86-micro-p
[ -n "$1" ] || run x86-p-miss-d1   $W --tag x86-p-miss-d1 --fillers nop,int,fp,load,store --n 16:640:16 --depth 1
run x86-p-miss-d2   $W --tag x86-p-miss-d2 --fillers nop,load,store --n 16:640:16 --depth 2
run x86-p45-units   $W --tag x86-p45-units --fillers v:nop8,v:nop9,v:br_nop7,v:br_split6 --n 4:100:4 --outer-reps 3
run x86-p45-instr   $W --tag x86-p45-instr --fillers v:mov,v:xor,v:addimm,v:branch --n 16:640:16 --outer-reps 3
# busy sibling on purpose: a spinner on CPU 3 while the ROB probe runs on CPU 2
taskset -c 3 python3 -c "while True: pass" & SPIN=$!
run x86-p-smt-busy  $W --tag x86-p-smt-busy --fillers nop --n 16:640:16 --outer-reps 3 --max-sibling 1.0
kill $SPIN
run x86-p-fp-d32    $W --tag x86-p-fp-d32 --fillers nop,load,store --n 16:640:16 --depth 32 --delay fp --outer-reps 3
run x86-p-miss-d1-rerun $W --tag x86-p-miss-d1-rerun --fillers nop,store --n 16:640:16 --depth 1 \
    --seed 100 --outer-reps 3
echo "== $(date +%T) done"
