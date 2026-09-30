# Source from the repo root:  . tools/env.sh  (cross compilers on PATH, $GEM5, scratch checks)
# Cross compilers: Debian gcc-14 cross packages unpacked into ~/opt/<arch>-cross
# (riscv64, aarch64); static linking only. gem5: validation/gem5/build.sh.
# build/, results/ and refs/ are symlinks into $UARCHRE_SCRATCH on local disk.

export PATH="$HOME/opt/riscv64-cross/bin:$HOME/opt/aarch64-cross/bin:$PATH"
export UARCHRE_SCRATCH="${UARCHRE_SCRATCH:-/var/tmp/$USER/uarch-re}"
UARCHRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
for d in build results refs; do
  [ -L "$UARCHRE_ROOT/$d" ] || echo "env.sh: $d is not a symlink into $UARCHRE_SCRATCH" >&2
done
export GEM5="${GEM5:-$UARCHRE_ROOT/validation/gem5/gem5-src/build/RISCV/gem5.opt}"
command -v riscv64-linux-gnu-gcc >/dev/null || echo "env.sh: riscv64-linux-gnu-gcc not found" >&2
command -v aarch64-linux-gnu-gcc >/dev/null || echo "env.sh: aarch64-linux-gnu-gcc not found" >&2
[ -x "$GEM5" ] || echo "env.sh: gem5 not built yet: $GEM5" >&2
