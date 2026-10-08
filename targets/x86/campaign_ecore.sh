#!/bin/bash
# The probes on a Gracemont E-core (CPU 14, no SMT sibling), alongside campaign_pcore.sh.
# CPU 14 is inside the gem5 CPU set (4-19), so inner reps are short (2000 iterations, ~1 ms)
# and many (9): most fit in one scheduler slice and the per-launch median ignores the ones a
# context switch split. Hence --min-share 0: preemption is recorded, not retried.
cd "$(dirname "$0")/../.." && . tools/env.sh
LOGS=/var/tmp/$USER/uarch-re/logs/x86; mkdir -p "$LOGS"
W="python3 targets/x86/window_campaign.py --cpu 14 --core e --iters 2000 --inner-reps 9 --min-share 0 --quiet-timeout 0.5"
run() { local tag=$1; shift; echo "== $(date +%T) $tag"; "$@" > "$LOGS/$tag.log" 2>&1 || echo "FAILED $tag"; }

run x86-e-miss-d1   $W --tag x86-e-miss-d1 --fillers nop,int,fp,load,store --n 8:320:8 --depth 1
run x86-e-miss-d2   $W --tag x86-e-miss-d2 --fillers nop,load,store --n 8:320:8 --depth 2
run x86-e45-units   $W --tag x86-e45-units --fillers v:nop8,v:nop9,v:br_nop7,v:br_split6 --n 2:50:2 --outer-reps 3
run x86-e-fp-d32    $W --tag x86-e-fp-d32 --fillers nop,load,store --n 8:320:8 --depth 32 --delay fp --outer-reps 3
run x86-footprint-e python3 targets/x86/footprint.py --cpus 14 --tag x86-footprint-e --iters 2000 --reps 9
run x86-micro-e     python3 targets/x86/micro_campaign.py --cpu 14 --tag x86-micro-e
run x86-e45-instr   $W --tag x86-e45-instr --fillers v:mov,v:xor,v:addimm,v:branch --n 8:320:8 --outer-reps 3
echo "== $(date +%T) done"
