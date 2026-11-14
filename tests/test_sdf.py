"""Distance estimator against brute force (design 3.2), from prototype/check_estimator.py (T2)."""
import itertools

import numpy as np
import pytest

from hypersponge.core.levels import level_table
from hypersponge.core.position import Pos
from hypersponge.core.sdf import de_table, removal_level, removal_row


def cells(n, m, L):
    cs = [np.zeros(n)]
    size = 1.0
    for _ in range(L):
        size /= 3
        new = []
        for c in cs:
            for d in itertools.product(range(3), repeat=n):
                if sum(x == 1 for x in d) < m:
                    new.append(c + np.array(d) * size)
        cs = new
    return np.array(cs), size


def true_dist(p, cs, size):
    """Distance from points p (N, n) to the union of boxes [c, c + size]."""
    out = np.empty(len(p))
    for i in range(0, len(p), 256):
        pp = p[i:i + 256, None, :]
        dd = np.maximum(np.maximum(cs[None] - pp, pp - (cs[None] + size)), 0)
        out[i:i + 256] = np.sqrt((dd ** 2).sum(-1)).min(1)
    return out


@pytest.mark.parametrize("n,m,L", [(2, 2, 3), (2, 1, 3), (3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 1, 2),
                                   (4, 2, 2), (4, 3, 2), (4, 4, 2), (4, 1, 2)])
def test_estimator_sign_lower_bound_lipschitz(n, m, L):
    rng = np.random.default_rng(1 + 10 * n + m)
    cs, size = cells(n, m, L)
    N = 2000
    p = rng.uniform(-0.3, 1.3, (N, n))
    td = true_dist(p, cs, size)
    tb = level_table(Pos.from_floats([0.0] * n, 0), n, m, max_level=L)
    q = np.zeros((N, 4))
    q[:, :n] = p
    d = de_table(q, tb, dtype=np.float64)
    inside = td == 0
    assert np.all(d[inside] <= 1e-12), "sign inside"
    assert np.all(d[~inside] > -1e-12), "sign outside"
    assert np.all(d[~inside] <= td[~inside] + 1e-9), "lower bound"
    p2 = p + rng.normal(0, 0.01, (N, n))
    q2 = np.zeros((N, 4))
    q2[:, :n] = p2
    d2 = de_table(q2, tb, dtype=np.float64)
    lip = np.abs(d2 - d) / np.linalg.norm(p2 - p, axis=1)
    assert lip.max() <= 1.0 + 1e-6


def test_removal_row_agrees_with_removal_level():
    """2D membership by table rows against the exact digit rule."""
    from fractions import Fraction as F
    rng = np.random.default_rng(4)
    for m in (1, 2):
        tb = level_table(Pos.from_floats([0.0, 0.0, 0.0, 0.0]), 2, m, max_level=5)
        p = rng.uniform(0.0, 1.0, (3000, 2))
        q = np.zeros((3000, 4))
        q[:, :2] = p
        row = removal_row(q, tb, 0.0, dtype=np.float64)
        for j in range(0, 3000, 3):
            k = removal_level([F(float(v)) for v in p[j]], m, 5)
            got = 0 if row[j] == -1 else int(tb.level[row[j]])
            assert got == k, (m, p[j], got, k)
    out = removal_row(np.array([[-0.1, 0.5, 0, 0], [0.5, 1.2, 0, 0]]), tb, 0.0, dtype=np.float64)
    assert list(out) == [-2, -2]
