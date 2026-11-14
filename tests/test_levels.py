"""The level table (design 13.2 to 13.4), from prototype/check_table.py (T3, T5) and
prototype/check_levels.py (the fast table against level_table_simple.py)."""
import random
from fractions import Fraction as F

import numpy as np
import pytest

from hypersponge.core.levels import BIG, level_table, normalize, normalize_2d
from hypersponge.core.position import Pos
from hypersponge.core.sdf import de_exact, de_table
from level_table_simple import level_table_simple


def rand_pos(n, E, mode, rng, rnd):
    O = []
    for i in range(n):
        if mode == "random":
            digs = [rnd.randrange(3) for _ in range(E)]
        elif mode == "ones":
            digs = [1] * E
        elif mode == "cantor":
            digs = [rnd.choice((0, 2)) for _ in range(E)]
        else:                                           # "wall"
            digs = [1] + [0] * (E - 1) if i < 2 else [rnd.randrange(3) for _ in range(E)]
        v = 0
        for d in digs:
            v = 3 * v + d
        O.append(v)
    f = rng.uniform(0, 1, n)
    if mode == "wall":
        f[:2] = rng.uniform(1e-9, 1e-3, 2)
    return Pos(O, f, E)


def exact_local(pos, q, n, m, L):
    X = pos.exact()
    s = F(3) ** pos.E
    return np.array([float(de_exact([X[i] + F(float(row[i])) / s for i in range(n)], m, L) * s) for row in q])


@pytest.mark.parametrize("n,m", [(2, 2), (3, 2), (4, 2), (4, 3), (3, 1), (4, 4)])
def test_float32_table_matches_exact_rationals(n, m):
    rng, rnd = np.random.default_rng(2 + n * m), random.Random(n * 10 + m)
    for E in (0, 1, 5, 20, 60, 200):
        for mode in ("random", "ones", "cantor", "wall"):
            if E == 0 and mode != "random":
                continue
            pos = rand_pos(n, E, mode, rng, rnd)
            K = 8
            tb = level_table(pos, n, m, max_level=E + K)
            if tb.dropped:                              # over 64 rows: a camera inside holes of 30+ levels
                continue
            N = 30
            scale = 10.0 ** rng.uniform(-3, min(4, 0.45 * E + 0.01) if E > 0 else -0.5, N)
            q = np.zeros((N, 4))
            q[:, :n] = rng.normal(0, 1, (N, n)) * scale[:, None]
            q = q.astype(np.float32)
            d32 = de_table(q, tb)
            dex = np.clip(exact_local(pos, q, n, m, E + K), -BIG, BIG)
            X = pos.exact()
            s = F(3) ** E
            ins = np.array([all(0 < X[i] + F(float(r[i])) / s < 1 for i in range(n)) for r in q])
            qn = np.linalg.norm(q, axis=1)[ins]
            err = np.abs(d32 - dex)[ins] / (qn + np.abs(dex[ins]) + 1e-30)
            assert (err < 2e-5).all(), (E, mode, err.max())
            away = np.abs(dex[ins]) > 1e-4 * qn
            assert (np.sign(d32[ins]) == np.sign(dex[ins]))[away].all(), (E, mode)


def norm_pos(n, m, E, mode, rng, rnd):
    while True:
        pos = rand_pos(n, E, mode, rng, rnd)
        try:
            normalize(pos, n, m)
            return pos
        except ValueError:
            pass


@pytest.mark.parametrize("n,m", [(3, 2), (4, 2), (4, 3)])
def test_far_limited_table_equals_full_table_within_far(n, m):
    rng, rnd = np.random.default_rng(5 + n + m), random.Random(n + 7 * m)
    for E in (10, 40, 62, 300):
        for mode in ("random", "cantor", "ones", "wall"):
            pos = norm_pos(n, m, E, mode, rng, rnd)
            far = 1000.0
            full = level_table(pos, n, m, fine_min=1e-3)
            pr = level_table(pos, n, m, far=far, fine_min=1e-3)
            if full.dropped:
                continue
            q = np.zeros((500, 4), dtype=np.float32)
            v = rng.normal(0, 1, (500, n))
            v /= np.linalg.norm(v, axis=1)[:, None]
            q[:, :n] = v * rng.uniform(0, far, (500, 1))
            a = de_table(q, full)
            b = de_table(q, pr)
            pos_ = a > 0
            assert np.array_equal(a[pos_], b[pos_]) and ((a > 0) == (b > 0)).all(), (E, mode)


def random_pos(n, E, kind, rng, rnd):
    O = []
    for i in range(4):
        if i >= n or E == 0:
            O.append(0)
            continue
        d = [rnd.choice("012") for _ in range(E)]
        if kind == "wall":                              # a long run of 0s or 2s: beside a coarse wall
            j = rnd.randrange(E)
            c = rnd.choice("02")
            stop = E if rnd.random() < 0.7 else rnd.randrange(j, E + 1)
            d[j:stop] = c * (stop - j)
        elif kind == "hole":                            # digit 1 on every axis at one coarse level
            d[rnd.randrange(E)] = "1"
        elif kind == "carpet":                          # no digit 1 at all
            d = [rnd.choice("02") for _ in range(E)]
        O.append(int("".join(d), 3))
    if kind == "outside" and E > 0:
        O[rnd.randrange(n)] += rnd.choice((-1, 1)) * 3 ** E * rnd.randrange(1, 3)
    f = rng.random(4)
    f[n:] = 0.0
    if kind == "wall":
        f[:n] = np.where(rng.random(n) < 0.5, f[:n] * 1e-6, 1 - f[:n] * 1e-6)
    return Pos(O, f, E)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_fast_table_equals_straightforward_table(n):
    rng, rnd = np.random.default_rng(21 + n), random.Random(21 + n)
    for m in range(1, n + 1):
        for E in (0, 1, 5, 35, 36, 37, 40, 80, 150, 300):
            for kind in ("random", "wall", "hole", "carpet", "outside", "normal"):
                for _ in range(2):
                    pos = random_pos(n, E, "wall" if kind == "normal" else kind, rng, rnd)
                    if kind == "normal":
                        try:
                            normalize(pos, n, m)
                        except ValueError:
                            continue
                    far = rnd.choice((0.0, 3.0, 1.0e3, 1.0e7))
                    ml = rnd.choice((None, None, max(1, E // 2), E + 3))
                    fm = rnd.choice((None, 1e-3))
                    cover = n == 2 and rnd.random() < 0.5
                    a = level_table(pos, n, m, ml, far, fm, cover=cover)
                    b = level_table_simple(pos, n, m, ml, far, fm, cover=cover)
                    case = (n, m, E, kind, far, ml)
                    assert a.covered == b.covered and a.dropped == b.dropped, case
                    if not (cover and a.covered):
                        assert a.de_cam == b.de_cam or (a.de_cam < -1e16 and b.de_cam < -1e16), case
                    for x, y in ((a.level, b.level), (a.w, b.w), (a.lo, b.lo), (a.hi, b.hi),
                                 (a.box_lo, b.box_lo), (a.box_hi, b.box_hi)):
                        assert x.shape == y.shape and np.array_equal(x, y), case


def test_deep_table_is_fast_and_matches():
    import time
    p = Pos.from_floats([0.25, 0.25, 0.25, 0.25])
    p.rescale(1000)
    p.move([0.3, 0.1, 0.2, 0.4])
    t0 = time.perf_counter()
    tb = level_table(p, 3, 2, far=40.0, fine_min=1e-3)
    t = time.perf_counter() - t0
    tb2 = level_table_simple(p, 3, 2, far=40.0, fine_min=1e-3)
    assert np.array_equal(tb.lo, tb2.lo) and np.array_equal(tb.level, tb2.level)
    assert t < 0.5                                       # about 1 ms in the prototype


def test_normalize_2d_keeps_height_in_one_to_three():
    for h in (1e-9, 0.01, 0.5, 1.0, 2.9, 3.0):
        p = Pos.from_floats([0.3, 0.6])
        x0 = p.exact()
        hh = normalize_2d(p, h)
        assert 1.0 <= hh < 3.0 or (p.E == 0 and hh == h)
        assert abs(hh * 3.0 ** -p.E - h) < 1e-12 * max(h, 1e-9) * 10
        assert all(abs(float(a - b)) < 1e-15 for a, b in zip(p.exact()[:2], x0[:2]))
