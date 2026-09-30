"""Two-pass sweep: coarse pass to locate the step, fine pass at step 1 around it."""

from .knee import find_knee


def sweep_refine(run, ns, fine_step=1, pad=None, min_jump=1.3):
    """run(n) -> cycles/iter. Returns (ns, ys, Knee) with the fine pass merged in."""
    ns = list(ns)
    ys = [run(n) for n in ns]
    k = find_knee(ns, ys, min_jump)
    if k.knee is None or len(ns) < 2:
        return ns, ys, k
    step = ns[1] - ns[0]
    if step <= fine_step:
        return ns, ys, k
    pad = pad if pad is not None else step
    lo = int(k.ci[0] - pad)
    hi = int(k.ci[1] + 1 + pad)
    have = set(ns)
    extra = [n for n in range(max(1, lo), hi + 1, fine_step) if n not in have]
    ns2 = ns + extra
    ys2 = ys + [run(n) for n in extra]
    order = sorted(range(len(ns2)), key=lambda i: ns2[i])
    ns2 = [ns2[i] for i in order]
    ys2 = [ys2[i] for i in order]
    return ns2, ys2, find_knee(ns2, ys2, min_jump)
