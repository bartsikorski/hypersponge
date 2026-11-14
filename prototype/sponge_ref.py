"""Hypersponge reference maths (prototype, CPU only, numpy).

Verified building blocks for the design document:

* exact membership / distance-estimator ground truth with Fractions,
* Pos: a position stored as base-3 digits (Python ints) + a float64 fraction,
* level_table(): camera-relative walls of every relevant level, exact, then float32,
* de_table(): the float32 distance estimator the shader will run (vectorised),
* march(): a ray marcher with pixel-size level of detail (for sample images and for
  the camera's "free distance" queries).

Conventions
-----------
The set lives in the unit cube [0,1]^n.  digit_k(x) = floor(x*3^k) mod 3, k = 1, 2, ...
R_k = points with at least m coordinates whose digit_k is 1.  S = cube minus all R_k.
A position is x = (O + f) * 3^-E  per axis: O integer (any size, any sign), f in [0,1).
"Local units" are units of 3^-E.  All shader quantities are local and camera-relative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import comb

import numpy as np

BIG = 1.0e30          # "infinitely far" in local units (float32 safe)
MAXLEV = 64           # size of the shader's level arrays


# ----------------------------------------------------------------------------------------
# exact ground truth (Fractions)
# ----------------------------------------------------------------------------------------
def kept_cells(n: int, m: int) -> int:
    return sum(comb(n, j) * 2 ** (n - j) for j in range(m))


def removal_level(x, m: int, max_level: int) -> int:
    """First level k at which the point is removed (0 = never up to max_level).
    x: sequence of Fractions inside [0,1)."""
    for k in range(1, max_level + 1):
        mids = sum(1 for xi in x if int(xi * 3 ** k) % 3 == 1)
        if mids >= m:
            return k
    return 0


def de_exact(x, m: int, max_level: int) -> Fraction:
    """max over levels of the 'exit distance' lower bound, world units, exact.
    Negative or zero: the point is in S_L (solid).  Box term only for points in the cube."""
    n = len(x)
    d = max(max(-xi, xi - 1) for xi in x)            # inside the cube: negative
    for k in range(1, max_level + 1):
        s = 3 ** k
        a = sorted(abs((xi * s) % 3 - Fraction(3, 2)) - Fraction(1, 2) for xi in x)
        d = max(d, -a[m - 1] / s)
    return d


# ----------------------------------------------------------------------------------------
# position as digits
# ----------------------------------------------------------------------------------------
@dataclass
class Pos:
    """x_i = (O_i + f_i) * 3^-E.  O: list of Python ints.  f: float64 array in [0,1)."""
    O: list
    f: np.ndarray
    E: int = 0

    @staticmethod
    def from_floats(x, E: int = 0) -> "Pos":
        x = np.asarray(x, dtype=np.float64) * 3.0 ** E
        fl = np.floor(x)
        return Pos([int(v) for v in fl], x - fl, E)

    def copy(self) -> "Pos":
        return Pos(list(self.O), self.f.copy(), self.E)

    def move(self, d) -> None:
        """Add a local-unit vector."""
        g = self.f + np.asarray(d, dtype=np.float64)
        fl = np.floor(g)
        self.f = g - fl
        self.f[self.f >= 1.0] = np.nextafter(1.0, 0.0)
        self.O = [o + int(c) for o, c in zip(self.O, fl)]

    def rescale(self, dE: int) -> None:
        """Change the unit by 3^-dE without moving (dE > 0: finer unit)."""
        while dE > 0:
            g = self.f * 3.0
            fl = np.floor(g)
            self.f = g - fl
            self.O = [3 * o + int(c) for o, c in zip(self.O, fl)]
            self.E += 1
            dE -= 1
        while dE < 0:
            r = [o % 3 for o in self.O]
            self.O = [o // 3 for o in self.O]
            self.f = (np.array(r, dtype=np.float64) + self.f) / 3.0
            self.f[self.f >= 1.0] = np.nextafter(1.0, 0.0)
            self.E -= 1
            dE += 1

    def exact(self):
        """World coordinates as Fractions (f is exact: a float is a dyadic rational)."""
        return [(Fraction(o) + Fraction(float(fi))) / Fraction(3) ** self.E for o, fi in zip(self.O, self.f)]


def _f(v: int) -> float:
    """Python int -> float, clamped."""
    if v > 10 ** 30:
        return BIG
    if v < -10 ** 30:
        return -BIG
    return float(v)


# ----------------------------------------------------------------------------------------
# camera-relative level table
# ----------------------------------------------------------------------------------------
@dataclass
class Table:
    lo: np.ndarray            # (L, 4) float64 camera-relative lower wall of the middle third
    hi: np.ndarray            # (L, 4)
    w: np.ndarray             # (L,)   width of the middle third (local units), descending
    level: np.ndarray         # (L,)   level number k
    box_lo: np.ndarray        # (4,)
    box_hi: np.ndarray        # (4,)
    off: np.ndarray           # (4,)   1.0 for a switched-off axis
    m: int = 2
    de_cam: float = 0.0       # distance estimate at the camera, all levels (local units)
    dropped: int = 0
    covered: int = 0          # 2D: level of the hole that contains the whole view (0 = none)

    def arrays(self, dtype=np.float32):
        c = lambda a: np.clip(a, -BIG, BIG).astype(dtype)
        return c(self.lo), c(self.hi), c(self.w), c(self.box_lo), c(self.box_hi)


T_EXACT = 36          # levels within 3^36 local units of the unit are always computed


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
                fine_min: float | None = None, fine_depth: int = 26, cover: bool = False) -> Table:
    """Walls of every level that can matter, relative to the camera, in local units.

    Level k <= E (coarse): exact integer arithmetic on the digits.
    Level k >  E (fine)  : float64 arithmetic on the fraction f.
    A coarse level is kept only if R_k comes within `far` of the camera (far < 3^35).
    Fine levels stop at width < fine_min or at max_level.
    cover (2D only): rows stop after the first level whose removed region contains every
    point within `far` of the camera, because that row removes the whole view.

    Cost: levels more than T_EXACT above the unit are examined only as digits (numpy), and
    get integer arithmetic only if the camera is inside their removed region or beside one
    of their walls (the digits after that level are all 0 or all 2).  The time is about the
    same at every depth.
    """
    E = pos.E
    f = pos.f
    last = E + fine_depth if max_level is None else min(max_level, E + fine_depth)
    full = 3 ** E
    rest = [pos.O[i] % full for i in range(n)]            # place inside the unit cube's cell grid
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
        p = 3 ** (E - k)
        w = float(p) if p < 10 ** 30 else BIG
        for i in range(n):
            lo_int = p - src[i] % (3 * p)
            i_lo[i] = _f(lo_int) - f[i]
            i_hi[i] = _f(lo_int + p) - f[i]
        return w

    def levels():
        for k in cand:
            lo = np.full(4, BIG); hi = np.full(4, BIG)
            yield k, coarse_row(k, rest, lo, hi), lo, hi
        for k in range(max(1, E - T + 1), min(E, last) + 1):
            lo = np.full(4, BIG); hi = np.full(4, BIG)
            yield k, coarse_row(k, low, lo, hi), lo, hi
        for k in range(E + 1, last + 1):
            lo = np.full(4, BIG); hi = np.full(4, BIG)
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
    keep = keep[dropped:]                           # drop the coarsest if the arrays overflow
    off = np.zeros(4)
    off[n:] = 1.0
    if not keep:
        keep = [(0, BIG, np.full(4, BIG), np.full(4, BIG))]
    return Table(np.array([r[2] for r in keep]), np.array([r[3] for r in keep]),
                 np.array([r[1] for r in keep]), np.array([r[0] for r in keep]),
                 box_lo, box_hi, off, m, de_cam, dropped, covered)


# ----------------------------------------------------------------------------------------
# the shader's distance estimator, float32, vectorised over points
# ----------------------------------------------------------------------------------------
def de_table(q, tb: Table, lodcut=None, dtype=np.float32, want_info: bool = False):
    """q: (N,4) camera-relative points (local units).  lodcut: (N,) levels with w < lodcut are
    skipped.  Returns d (N,) and, if want_info, the level index and the axis (0..3, +4 for
    the outer box) that set the maximum."""
    q = np.asarray(q, dtype=dtype)
    lo, hi, w, blo, bhi = tb.arrays(dtype)
    N = q.shape[0]
    big = dtype(BIG)
    off = tb.off.astype(bool)
    # outer cube
    b = np.maximum(blo - q, q - bhi)
    b[:, off] = -big
    bmax = b.max(axis=1)
    d = np.sqrt(np.sum(np.maximum(b, 0) ** 2, axis=1)) + np.minimum(bmax, 0)
    lev = np.full(N, -1)
    axis = b.argmax(axis=1) + 4
    for i in range(len(w)):
        wi = w[i]
        u = q - lo[i]
        a = np.maximum(lo[i] - q, q - hi[i])
        farm = np.abs(u - dtype(0.5) * wi) > dtype(1.5) * wi
        if farm.any():
            v = np.mod(u + wi, dtype(3.0) * wi) - wi
            a = np.where(farm, np.maximum(-v, v - wi), a)
        a[:, off] = big
        order = np.argsort(a, axis=1)
        am = np.take_along_axis(a, order[:, tb.m - 1:tb.m], axis=1)[:, 0]
        term = -am
        if lodcut is not None:
            term = np.where(wi >= lodcut, term, -big)
        better = term > d
        d = np.where(better, term, d)
        if want_info:
            lev = np.where(better, i, lev)
            axis = np.where(better, order[:, tb.m - 1], axis)
    if want_info:
        return d, lev, axis
    return d


def march(tb: Table, dirs, pix_angle: float, far: float, max_steps: int = 200, lod: float = 1.0,
          hit: float = 0.5, t0: float = 0.0):
    """Sphere tracing from the camera (origin) along unit 4-vectors dirs (N,4).
    Level of detail: at distance t, levels narrower than lod * t * pix_angle are skipped.
    Returns t (N,), hit mask, step counts."""
    dirs = np.asarray(dirs, dtype=np.float32)
    N = dirs.shape[0]
    t = np.full(N, t0, dtype=np.float32)
    done = np.zeros(N, dtype=bool)
    hitm = np.zeros(N, dtype=bool)
    steps = np.zeros(N, dtype=np.int32)
    for _ in range(max_steps):
        idx = np.nonzero(~done)[0]
        if idx.size == 0:
            break
        ti = t[idx]
        foot = ti * np.float32(pix_angle)
        d = de_table(dirs[idx] * ti[:, None], tb, lodcut=np.float32(lod) * foot)
        h = d < np.float32(hit) * foot
        hitm[idx[h]] = True
        done[idx[h]] = True
        ti = ti + np.maximum(d, 0)
        out = (ti > far) & ~h
        done[idx[out]] = True
        t[idx[~h]] = ti[~h]
        steps[idx] += 1
    return t, hitm, steps


def normalize(pos: Pos, n: int, m: int, max_level: int | None = None) -> float:
    """Choose the unit (E) so that the camera's own distance estimate is in [1, 3) local
    units (E never below 0).  Returns that estimate.  Raises if the camera is in solid."""
    from math import floor, log
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


# ----------------------------------------------------------------------------------------
# scalar versions (pure Python floats): one point / one ray, for camera logic
# ----------------------------------------------------------------------------------------
def table_rows(tb: Table):
    """Plain-Python copy of a table for the scalar functions."""
    n = int(4 - tb.off.sum())
    return (n, tb.m, [(float(w), [float(v) for v in lo[:n]], [float(v) for v in hi[:n]])
                      for w, lo, hi in zip(tb.w, tb.lo, tb.hi)],
            [float(v) for v in tb.box_lo[:n]], [float(v) for v in tb.box_hi[:n]])


def de_scalar(q, rows, lodcut: float = 0.0) -> float:
    n, m, levels, blo, bhi = rows
    s = 0.0
    bm = -BIG
    for i in range(n):
        b = blo[i] - q[i]
        b2 = q[i] - bhi[i]
        if b2 > b:
            b = b2
        if b > bm:
            bm = b
        if b > 0.0:
            s += b * b
    d = s ** 0.5 + (bm if bm < 0.0 else 0.0)
    a = [0.0] * n
    for w, lo, hi in levels:
        if w < lodcut:
            break
        for i in range(n):
            x = q[i]
            u = x - lo[i]
            if -w <= u <= 2.0 * w:
                v1 = lo[i] - x
                v2 = x - hi[i]
            else:
                v = (u + w) % (3.0 * w) - w
                v1 = -v
                v2 = v - w
            a[i] = v1 if v1 > v2 else v2
        am = sorted(a)[m - 1]
        if -am > d:
            d = -am
    return d


def free_distance(rows, direction, pix_angle: float, far: float, max_steps: int = 400,
                  hit: float = 0.5, lod: float = 1.0):
    """Distance from the camera to the first surface along a unit direction (active axes
    only), with the same level-of-detail rule as the picture.  Returns (t, hit?)."""
    n = rows[0]
    t = 0.0
    for _ in range(max_steps):
        foot = t * pix_angle
        d = de_scalar([direction[i] * t for i in range(n)], rows, lod * foot)
        if d < hit * foot:
            return t, True
        t += d
        if t > far:
            return far, False
    return t, True


def removal_row(q, tb: Table, lodcut: float, dtype=np.float32):
    """First row that removes each point: -1 kept, -2 outside the cube (the 2D shader's test)."""
    q = np.asarray(q, dtype=dtype)
    lo, hi, w, blo, bhi = tb.arrays(dtype)
    big = dtype(BIG)
    off = tb.off.astype(bool)
    b = np.maximum(blo - q, q - bhi)
    b[:, off] = -big
    out = np.where((np.sqrt(np.sum(np.maximum(b, 0) ** 2, axis=1)) + np.minimum(b.max(axis=1), 0)) > 0, -2, -1)
    for i in range(len(w)):
        wi = w[i]
        if wi < lodcut:
            break
        u = q - lo[i]
        a = np.maximum(lo[i] - q, q - hi[i])
        farm = np.abs(u - dtype(0.5) * wi) > dtype(1.5) * wi
        if farm.any():
            v = np.mod(u + wi, dtype(3.0) * wi) - wi
            a = np.where(farm, np.maximum(-v, v - wi), a)
        a[:, off] = big
        am = np.sort(a, axis=1)[:, tb.m - 1]
        out = np.where((out == -1) & (am < 0), i, out)
    return out


def normalize_2d(pos: Pos, height: float) -> float:
    """2D unit rule: keep the view height in [1, 3) local units (E never below 0)."""
    while height < 1.0:
        pos.rescale(+1)
        height *= 3.0
    while height >= 3.0 and pos.E > 0:
        pos.rescale(-1)
        height /= 3.0
    return height
