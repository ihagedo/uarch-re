"""Replacement policies of one cache set, and periodic access sequences that tell them apart.

misses(policy, ways, seq) is the steady-state misses per period of seq (line ids) on a set of `ways`,
after a reset chain that thrashes the set; sequences(ways) picks a few seqs that separate every pair of
POLICIES by misses per period.
"""

import itertools
import random

POLICIES = ("lru", "fifo", "tree", "tree-heap", "bit", "random")


class _Set:
    def __init__(self, ways, rng=None):
        self.ways, self.lines, self.rng = ways, [None] * ways, rng

    def access(self, x):
        if x in self.lines:
            self.touch(self.lines.index(x))
            return 0
        w = self.lines.index(None) if None in self.lines else self.victim()
        self.lines[w] = x
        self.fill(w)
        return 1

    def touch(self, w):
        pass

    def fill(self, w):
        self.touch(w)


class _LRU(_Set):
    def __init__(self, ways, rng=None):
        super().__init__(ways)
        self.t, self.age = 0, [0] * ways

    def touch(self, w):
        self.t += 1
        self.age[w] = self.t

    def victim(self):
        return min(range(self.ways), key=self.age.__getitem__)


class _FIFO(_LRU):
    def touch(self, w):
        pass

    def fill(self, w):
        _LRU.touch(self, w)


class _Tree(_Set):
    """Tree-PLRU: a bit per node points at the half to evict; ways not a power of two split unevenly."""

    def __init__(self, ways, rng=None):
        super().__init__(ways)
        self.bits = {}

    def touch(self, w):
        lo, hi = 0, self.ways
        while hi - lo > 1:
            mid = (lo + hi + 1) // 2
            self.bits[lo, hi] = 0 if w >= mid else 1
            lo, hi = (mid, hi) if w >= mid else (lo, mid)

    def victim(self):
        lo, hi = 0, self.ways
        while hi - lo > 1:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if self.bits.get((lo, hi), 0) else (lo, mid)
        return lo


class _Heap(_Set):
    """Tree-PLRU in heap layout (gem5's TreePLRU): nodes 0..ways-2, node i's children 2i+1 and 2i+2, way
    w at node ways-1+w; the same tree as _Tree when ways is a power of two."""

    def __init__(self, ways, rng=None):
        super().__init__(ways)
        self.bits = [False] * (ways - 1)

    def touch(self, w):
        i = self.ways - 1 + w
        while i:
            right, i = i % 2 == 0, (i - 1) // 2
            self.bits[i] = not right

    def victim(self):
        i = 0
        while i < self.ways - 1:
            i = 2 * i + (2 if self.bits[i] else 1)
        return i - (self.ways - 1)


class _Bit(_Set):
    """Bit-PLRU: an MRU bit per way, all but the newest cleared when every bit is set."""

    def __init__(self, ways, rng=None):
        super().__init__(ways)
        self.mru = [0] * ways

    def touch(self, w):
        self.mru[w] = 1
        if all(self.mru):
            self.mru = [int(i == w) for i in range(self.ways)]

    def victim(self):
        return self.mru.index(0)


class _Random(_Set):
    def victim(self):
        return self.rng.randrange(self.ways)


_SETS = {"lru": _LRU, "fifo": _FIFO, "tree": _Tree, "tree-heap": _Heap, "bit": _Bit, "random": _Random}


def policies(ways):
    """POLICIES that differ at `ways`: the two tree layouts are one tree at a power of two."""
    return tuple(p for p in POLICIES if p != "tree-heap" or ways & (ways - 1))


def reset(ways, longest=16):
    """Line ids of the reset chain run before every seq: more lines than ways, none of a seq's."""
    return list(range(ways + 2, ways + 2 + min(ways + 4, longest)))


def misses(policy, ways, seq, warm=30, meas=60, trials=20, start=None, stray=0):
    """Steady-state misses per period after three laps of reset(ways); random averages `trials` seeded
    runs. `start` seeds a set filled first in a random order with lines of seq and others (None: empty),
    then `stray` random accesses to seq's and the reset's lines after the reset (wrong-path fetches when
    the probe switches chains)."""
    runs = range(trials if policy == "random" else 1)
    total = 0
    for t in runs:
        s = _SETS[policy](ways, random.Random(t))
        rng = random.Random(start)
        if start is not None:
            for _ in range(4 * ways):
                s.access(rng.randrange(-ways, max(seq) + 1))
        pool = sorted(set(seq) | set(reset(ways)))
        for x in reset(ways) * 3 + [rng.choice(pool) for _ in range(stray)] + seq * warm:
            s.access(x)
        total += sum(s.access(x) for _ in range(meas) for x in seq)
    return total / (len(runs) * meas)


def settles(ways, seq):
    """True if no deterministic policy's steady state on seq depends on the set's earlier contents or on
    a few stray accesses after the reset."""
    return all(
        len({misses(p, ways, seq, start=t, stray=k) for t in range(6) for k in (0, 2, 6)}) == 1
        for p in policies(ways)
        if p != "random"
    )


def sequences(ways, n=1000, k=4, seed=0, longest=16):
    """(seqs, settled): k of n random periodic seqs over ways + 1 or + 2 lines, at most `longest` visits,
    chosen greedily so that every policy pair differs by as many misses per period as one of them shows.
    Seqs that settle come first; settled is False when too few did and others filled in."""
    rng = random.Random(seed)
    cands = []
    for _ in range(n):
        d = min(rng.choice([ways + 1, ways + 2]), longest)
        seq = list(range(d)) + [rng.randrange(d) for _ in range(rng.randint(0, longest - d))]
        rng.shuffle(seq)
        cands.append(seq)
    good = [s for s in cands if settles(ways, s)]
    chosen = _spread(ways, good, k)
    settled = len(chosen) == k
    if not settled:
        chosen += _spread(ways, [s for s in cands if s not in good], k - len(chosen))
    return chosen, settled


def _spread(ways, cands, k):
    """Greedy: the seq that most raises the smallest pairwise separation, k times."""
    table = [{p: misses(p, ways, s) for p in policies(ways)} for s in cands]
    pairs = list(itertools.combinations(policies(ways), 2))
    sep, chosen = dict.fromkeys(pairs, 0.0), []
    for _ in range(min(k, len(cands))):
        i = max(
            range(len(cands)),
            key=lambda i: min(max(sep[a, b], abs(table[i][a] - table[i][b])) for a, b in pairs),
        )
        chosen.append(cands[i])
        sep = {(a, b): max(sep[a, b], abs(table[i][a] - table[i][b])) for a, b in pairs}
    return chosen
