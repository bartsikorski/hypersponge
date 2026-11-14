"""Exact ground truth and the numpy twins of the shader's estimator (design sections 3.1, 13.4).

de_table, removal_row and march are float32 copies of common.glsl and march.frag, operation for
operation.  They are the references in the shader tests and must change together with the GLSL.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .levels import BIG, Table


# ------------------------------------------------------------------------------------------
# exact ground truth (Fractions)
# ------------------------------------------------------------------------------------------
def removal_level(x, m: int, max_level: int) -> int:
    """First level that removes the point, 0 = never up to max_level.  x: Fractions in [0, 1)."""
    for k in range(1, max_level + 1):
        mids = sum(1 for xi in x if int(xi * 3 ** k) % 3 == 1)
        if mids >= m:
            return k
    return 0


def de_exact(x, m: int, max_level: int) -> Fraction:
    """The estimator in world units, exact.  At most 0 on the solid S_L, positive outside."""
    d = max(max(-xi, xi - 1) for xi in x)            # inside the cube: negative
    for k in range(1, max_level + 1):
        s = 3 ** k
        a = sorted(abs((xi * s) % 3 - Fraction(3, 2)) - Fraction(1, 2) for xi in x)
        d = max(d, -a[m - 1] / s)
    return d


# ------------------------------------------------------------------------------------------
# the shader's estimator, float32, vectorised over points
# ------------------------------------------------------------------------------------------
def _wall_dist(q, lo_i, hi_i, wi, off, dtype):
    u = q - lo_i
    a = np.maximum(lo_i - q, q - hi_i)
    farm = np.abs(u - dtype(0.5) * wi) > dtype(1.5) * wi
    if farm.any():
        v = np.mod(u + wi, dtype(3.0) * wi) - wi
        a = np.where(farm, np.maximum(-v, v - wi), a)
    a[:, off] = dtype(BIG)
    return a


def de_table(q, tb: Table, lodcut=None, dtype=np.float32, want_info: bool = False):
    """The shader's de (and deInfo) for camera-relative points q (N, 4) in local units.  Rows
    narrower than lodcut (scalar or (N,)) are skipped.  With want_info also returns the row
    (-1 = outer cube) and the axis (0..3) that set the maximum."""
    q = np.asarray(q, dtype=dtype)
    lo, hi, w, blo, bhi = tb.arrays(dtype)
    N = q.shape[0]
    big = dtype(BIG)
    off = tb.off.astype(bool)
    b = np.maximum(blo - q, q - bhi)
    b[:, off] = -big
    bmax = b.max(axis=1)
    d = np.sqrt(np.sum(np.maximum(b, 0) ** 2, axis=1)) + np.minimum(bmax, 0)
    row = np.full(N, -1)
    axis = b.argmax(axis=1)
    for i in range(len(w)):
        a = _wall_dist(q, lo[i], hi[i], w[i], off, dtype)
        am = np.sort(a, axis=1)[:, tb.m - 1]
        term = -am
        if lodcut is not None:
            term = np.where(w[i] >= lodcut, term, -big)
        better = term > d
        d = np.where(better, term, d)
        if want_info:
            row = np.where(better, i, row)
            # the first axis holding the m-th smallest value, as the shader picks it
            axis = np.where(better, np.argmax(a == am[:, None], axis=1), axis)
    if want_info:
        return d, row, axis
    return d


def removal_row(q, tb: Table, lodcut: float, dtype=np.float32):
    """The shader's removalRow: first row that removes each point, -1 kept, -2 outside the cube."""
    q = np.asarray(q, dtype=dtype)
    lo, hi, w, blo, bhi = tb.arrays(dtype)
    big = dtype(BIG)
    off = tb.off.astype(bool)
    b = np.maximum(blo - q, q - bhi)
    b[:, off] = -big
    out = np.where((np.sqrt(np.sum(np.maximum(b, 0) ** 2, axis=1)) + np.minimum(b.max(axis=1), 0)) > 0, -2, -1)
    for i in range(len(w)):
        if w[i] < lodcut:
            break
        a = _wall_dist(q, lo[i], hi[i], w[i], off, dtype)
        am = np.sort(a, axis=1)[:, tb.m - 1]
        out = np.where((out == -1) & (am < 0), i, out)
    return out


def march(tb: Table, dirs, pix_angle: float, far: float, max_steps: int = 200, lod: float = 1.0,
          hit: float = 0.5, t0: float = 0.0):
    """Sphere tracing from the camera along unit 4-vectors dirs (N, 4).  At distance t, rows
    narrower than lod * t * pix_angle are skipped.  Returns t (N,), the hit mask and step counts."""
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
    hitm |= ~done                                    # running out of steps counts as a hit
    return t, hitm, steps


# ------------------------------------------------------------------------------------------
# scalar versions (pure Python floats): one point or one ray, for the camera
# ------------------------------------------------------------------------------------------
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
    """Distance from the camera to the first surface along a unit direction, with the picture's
    level of detail.  Returns (t, hit); a miss returns (far, False), running out of steps a hit."""
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
