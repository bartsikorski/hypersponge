"""The camera-relative level table and the unit rules (design sections 3.3, 3.4, 13.3).

The shader never sees absolute coordinates.  For every level that can matter the table holds,
per axis, the two walls of the middle third that surrounds the camera, measured from the camera
in local units.  Walls of coarse levels are computed with exact integers, so a wall next to the
camera is exact however deep the camera is.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, log

import numpy as np

from .position import Pos

BIG = 1.0e30          # "infinitely far" in local units (float32 safe)
MAXLEV = 64           # size of the shader's row arrays
T_EXACT = 36          # levels within 3^36 local units of the unit are always computed
FINE_DEPTH = 26       # fine levels below the unit that are ever looked at


def _f(v: int) -> float:
    """Python int to float, clamped to +-BIG."""
    if v > 10 ** 30:
        return BIG
    if v < -10 ** 30:
        return -BIG
    return float(v)


@dataclass
class Table:
    lo: np.ndarray            # (rows, 4) float64 camera-relative lower wall of the middle third
    hi: np.ndarray            # (rows, 4) upper wall
    w: np.ndarray             # (rows,)   slab width in local units, widest first
    level: np.ndarray         # (rows,)   level number k of each row
    box_lo: np.ndarray        # (4,)      camera-relative faces of the outer cube
    box_hi: np.ndarray        # (4,)
    off: np.ndarray           # (4,)      1.0 for a switched-off axis
    m: int = 2
    de_cam: float = 0.0       # distance estimate at the camera over all levels (local units)
    dropped: int = 0          # rows dropped by the 64-row cap
    covered: int = 0          # 2D: level of the hole that contains the whole view (0 = none)

    def arrays(self, dtype=np.float32):
        """lo, hi, w, box_lo, box_hi clipped to +-BIG and cast."""
        c = lambda a: np.clip(a, -BIG, BIG).astype(dtype)   # noqa: E731
        return c(self.lo), c(self.hi), c(self.w), c(self.box_lo), c(self.box_hi)

    def key(self):
        """Hashable summary: equal keys mean equal shader uniforms."""
        return (self.lo.tobytes(), self.hi.tobytes(), self.w.tobytes(), self.level.tobytes(),
                self.box_lo.tobytes(), self.box_hi.tobytes(), self.off.tobytes(), self.m)


def _digits(v: int, count: int) -> np.ndarray:
    """Base-3 digits of 0 <= v < 3^count, most significant first."""
    if count <= 0:
        return np.zeros(0, dtype=np.int8)
    chunk = 20
    base = 3 ** chunk
    parts = []
    for _ in range(0, count, chunk):
        v, c = divmod(v, base)
        parts.append(c)
    c = np.array(parts[::-1], dtype=np.int64)
    pw = 3 ** np.arange(chunk - 1, -1, -1, dtype=np.int64)
    return ((c[:, None] // pw[None, :]) % 3).reshape(-1)[-count:].astype(np.int8)


def level_table(pos: Pos, n: int, m: int, max_level: int | None = None, far: float = BIG,
                fine_min: float | None = None, fine_depth: int = FINE_DEPTH, cover: bool = False) -> Table:
    """Walls of every level that can matter, relative to the camera, in local units.

    Level k <= E (coarse): exact integer arithmetic on the digits.
    Level k >  E (fine)  : float64 arithmetic on the fraction f.
    A coarse level is kept only if R_k comes within `far` of the camera (far < 3^35).
    Rows narrower than fine_min are skipped; levels stop at max_level.
    cover (2D only): rows stop after the first level whose removed region contains every
    point within `far` of the camera, because that row removes the whole view.

    Levels more than T_EXACT above the unit are examined only as digits (numpy), and get integer
    arithmetic only if the camera is inside their removed region or beside one of their walls
    (the digits after that level are all 0 or all 2).  The cost is about the same at every depth.
    """
    E = pos.E
    f = pos.f
    last = E + fine_depth if max_level is None else min(max_level, E + fine_depth)
    full = 3 ** E
    rest = [pos.O[i] % full for i in range(n)]            # walls repeat with the unit cube's grid
    T = min(E, T_EXACT)
    lowm = 3 ** T
    low = [r % lowm for r in rest]

    # levels 1 .. E-T: candidates from the digits alone
    ncoarse = min(E - T, last)
    cand = []
    if ncoarse > 0:
        dig = np.stack([_digits(rest[i] // lowm, E - T) for i in range(n)])
        near = np.zeros((n, E - T), dtype=bool)           # a wall of this level may be close:
        for i in range(n):                                # every digit after it is 0, or every one is 2
            nz = np.flatnonzero(dig[i] != 0)
            n2 = np.flatnonzero(dig[i] != 2)
            if nz.size:
                near[i, nz[-1]] = True                    # just above a wall of this level
            if n2.size:
                near[i, n2[-1]] = True                    # just below one
        count = ((dig == 1) | near).sum(axis=0)
        cand = [int(j) + 1 for j in np.flatnonzero(count[:ncoarse] >= m)]

    def coarse_row(k, src, i_lo, i_hi):
        p = 3 ** (E - k)                                  # slab width in local units, an integer
        w = float(p) if p < 10 ** 30 else BIG
        for i in range(n):
            lo_int = p - src[i] % (3 * p)
            i_lo[i] = _f(lo_int) - f[i]
            i_hi[i] = _f(lo_int + p) - f[i]
        return w

    def levels():
        for k in cand:
            lo = np.full(4, BIG)
            hi = np.full(4, BIG)
            yield k, coarse_row(k, rest, lo, hi), lo, hi
        for k in range(max(1, E - T + 1), min(E, last) + 1):
            lo = np.full(4, BIG)
            hi = np.full(4, BIG)
            yield k, coarse_row(k, low, lo, hi), lo, hi
        for k in range(E + 1, last + 1):
            lo = np.full(4, BIG)
            hi = np.full(4, BIG)
            s = 3.0 ** (k - 1 - E)
            w = 1.0 / (3.0 * s)
            for i in range(n):
                P = np.floor(f[i] * s)
                lo[i] = (3.0 * P + 1.0) * w - f[i]
                hi[i] = (3.0 * P + 2.0) * w - f[i]
            yield k, w, lo, hi

    # outer cube and the distance estimate at the camera (q = 0)
    box_lo = np.full(4, -BIG)
    box_hi = np.full(4, BIG)
    for i in range(n):
        box_lo[i] = _f(-pos.O[i]) - f[i]
        box_hi[i] = _f(full - pos.O[i]) - f[i]
    bx = np.maximum(box_lo, -box_hi)[:n]
    de_cam = float(np.sqrt(np.sum(np.maximum(bx, 0.0) ** 2)) + min(float(bx.max()), 0.0))
    if fine_min is None:
        fine_min = 0.0
    keep = []
    covered = 0
    for (k, w, lo, hi) in levels():
        a = np.sort(np.maximum(lo, -hi)[:n])
        de_cam = max(de_cam, -a[m - 1])                    # distance camera -> out of R_k
        if w < fine_min:
            continue
        if float(np.sqrt(np.sum(np.maximum(a[:m], 0.0) ** 2))) < far:      # distance camera -> R_k
            keep.append((k, w, lo, hi))
        if cover and -a[m - 1] >= far:
            covered = k
            break
    dropped = max(0, len(keep) - MAXLEV)
    keep = keep[dropped:]                                  # drop the coarsest if the arrays overflow
    off = np.zeros(4)
    off[n:] = 1.0
    if not keep:                                           # a dummy row that removes nothing
        keep = [(0, BIG, np.full(4, BIG), np.full(4, BIG))]
    return Table(np.array([r[2] for r in keep]), np.array([r[3] for r in keep]),
                 np.array([r[1] for r in keep]), np.array([r[0] for r in keep]),
                 box_lo, box_hi, off, m, de_cam, dropped, covered)


def normalize(pos: Pos, n: int, m: int, max_level: int | None = None) -> float:
    """3D and 4D unit rule: choose E so that the camera's own distance estimate is in [1, 3)
    local units (E never below 0).  Returns that estimate.  Raises ValueError in solid."""
    for _ in range(100000):
        de = level_table(pos, n, m, max_level, far=0.0).de_cam
        if de <= 0.0:
            raise ValueError("camera inside solid")
        if de >= 1e29:
            step = -40
        else:
            step = -int(floor(log(de) / log(3.0) + 1e-12))
        step = max(step, -pos.E)
        if step == 0 or (1.0 <= de < 3.0):
            return de
        pos.rescale(step)
    raise RuntimeError("normalize did not settle")


def normalize_2d(pos: Pos, height: float) -> float:
    """2D unit rule: keep the view height in [1, 3) local units (E never below 0).  Returns the
    height in the new unit."""
    while height < 1.0:
        pos.rescale(+1)
        height *= 3.0
    while height >= 3.0 and pos.E > 0:
        pos.rescale(-1)
        height /= 3.0
    return height
