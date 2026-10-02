#!/bin/bash
# Build gem5 v25.1 <ISA> opt with minimal features to keep build time down.
#   ISA=RISCV (default) | ARM | X86      JOBS=8 (default; ~2 GB peak per compile job)
# gem5-src is a symlink to local disk (/var/tmp/$USER/gem5-src): the build tree is
# ~10 GB per ISA, too big and too slow for NFS home.
cd "$(dirname "$0")/gem5-src"
exec nice -n 10 scons "build/${ISA:-RISCV}/gem5.opt" -j"${JOBS:-8}" --without-tcmalloc \
  --no-compress-debug --ignore-style 2>&1
