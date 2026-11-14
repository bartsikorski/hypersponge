"""The first, straightforward level_table: every level with big-integer arithmetic.  Exact but
its cost grows with the square of the depth.  Kept as the reference for check_levels.py."""
from __future__ import annotations

import numpy as np

from sponge_ref import BIG, MAXLEV, Pos, Table, _f


def level_table_simple(pos: Pos, n: int, m: int, max_level: int | None = None, far: float = BIG,
                fine_min: float | None = None, fine_depth: int = 26, cover: bool = False) -> Table:
    """Walls of every level that can matter, relative to the camera, in local units.

    Level k <= E (coarse): exact integer arithmetic on the digits.
    Level k >  E (fine)  : float64 arithmetic on the fraction f.
    A coarse level is kept only if R_k comes within `far` of the camera.
    Fine levels stop at width < fine_min (default: a fraction of the camera's own
    distance estimate) or at max_level.
    cover (2D only): rows stop after the first level whose removed region contains every
    point within `far` of the camera, because that row removes the whole view.
    """
    E = pos.E
    f = pos.f
    last = E + fine_depth if max_level is None else min(max_level, E + fine_depth)
    rows = []
    for k in range(1, last + 1):
        lo = np.full(4, BIG)
        hi = np.full(4, BIG)
        if k <= E:
            p = 3 ** (E - k)                       # width of the middle third, an int
            w = min(float(p), BIG) if p < 10 ** 30 else BIG
            for i in range(n):
                P = pos.O[i] // (3 * p)            # index of the level-(k-1) cell (floor)
                lo_int = (3 * P + 1) * p - pos.O[i]
                lo[i] = _f(lo_int) - f[i]
                hi[i] = _f(lo_int + p) - f[i]
        else:
            s = 3.0 ** (k - 1 - E)
            w = 1.0 / (3.0 * s)
            for i in range(n):
                P = np.floor(f[i] * s)
                lo[i] = (3.0 * P + 1.0) * w - f[i]
                hi[i] = (3.0 * P + 2.0) * w - f[i]
        rows.append((k, w, lo, hi))

    # distance estimate at the camera over all levels (q = 0)
    box_lo = np.full(4, -BIG)
    box_hi = np.full(4, BIG)
    for i in range(n):
        box_lo[i] = _f(-pos.O[i]) - f[i]
        box_hi[i] = _f(3 ** E - pos.O[i]) - f[i]
    bx = np.maximum(box_lo, -box_hi)[:n]
    de_cam = float(np.sqrt(np.sum(np.maximum(bx, 0.0) ** 2)) + min(float(bx.max()), 0.0))
    info = []
    exits = []
    for (k, w, lo, hi) in rows:
        a = np.sort(np.maximum(lo, -hi)[:n])
        de_cam = max(de_cam, -a[m - 1])
        info.append(float(np.sqrt(np.sum(np.maximum(a[:m], 0.0) ** 2))))   # distance camera -> R_k
        exits.append(float(-a[m - 1]))                                     # distance camera -> out of R_k
    if fine_min is None:
        fine_min = 0.0
    keep = []
    covered = 0
    for (k, w, lo, hi), dist, ex in zip(rows, info, exits):
        if w < fine_min:
            break
        if dist < far:
            keep.append((k, w, lo, hi))
        if cover and ex >= far:
            covered = k
            break
    dropped = max(0, len(keep) - MAXLEV)
    keep = keep[dropped:]                           # drop the coarsest if the arrays overflow
    off = np.zeros(4)
    off[n:] = 1.0
    if not keep:
        keep = [(0, BIG, np.full(4, BIG), np.full(4, BIG))]
    return Table(np.array([r[2] for r in keep]), np.array([r[3] for r in keep]),
                 np.array([r[1] for r in keep]), np.array([r[0] for r in keep]),
                 box_lo, box_hi, off, m, de_cam, dropped, covered)
