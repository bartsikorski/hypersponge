"""Pos arithmetic and rebase invariance (design 13.1), from prototype/check_table.py (T4)."""
import random
from fractions import Fraction as F

import numpy as np
import pytest

from hypersponge.core.levels import level_table, normalize
from hypersponge.core.position import Pos
from hypersponge.core.sdf import de_table


def rand_pos(n, E, rng, rnd):
    O = []
    for _ in range(n):
        v = 0
        for _ in range(E):
            v = 3 * v + rnd.randrange(3)
        O.append(v)
    return Pos(O, rng.uniform(0, 1, n), E)


def gap(a, b, E):
    return max(abs(float((x - y) * F(3) ** E)) for x, y in zip(a, b))


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("E", [3, 30, 150])
def test_rescale_and_move_are_exact(n, E):
    rng, rnd = np.random.default_rng(2 + n + E), random.Random(n * E)
    pos = rand_pos(n, E, rng, rnd)
    X0 = pos.exact()
    p2 = pos.copy()
    p2.rescale(+2)
    assert gap(p2.exact(), X0, E) < 1e-15
    p3 = pos.copy()
    p3.rescale(-1)
    assert gap(p3.exact(), X0, E) < 1e-15
    mv = rng.normal(0, 5, n)
    p4 = pos.copy()
    p4.move(mv)
    p4.move(-mv)
    assert gap(p4.exact(), X0, E) < 1e-13


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("E", [3, 30, 150])
def test_estimate_does_not_depend_on_the_unit(n, E):
    rng, rnd = np.random.default_rng(7 + n + E), random.Random(3 * n + E)
    while True:
        pos = rand_pos(n, E, rng, rnd)
        try:
            normalize(pos, n, 2)
            break
        except ValueError:
            pass
    p2 = pos.copy()
    p2.rescale(+2)
    tb = level_table(pos, n, 2, far=1e3, fine_min=1e-3)
    tb2 = level_table(p2, n, 2, far=9e3, fine_min=9e-3)
    assert list(tb.level) == list(tb2.level)
    q = np.zeros((300, 4), dtype=np.float32)
    q[:, :n] = rng.normal(0, 1, (300, n)) * (10.0 ** rng.uniform(-2, 2, 300))[:, None]
    d1 = de_table(q, tb)
    d2 = de_table(q * np.float32(9), tb2) / 9
    assert np.allclose(d1, d2, rtol=2e-4, atol=2e-6 * np.linalg.norm(q, axis=1).max())


def test_from_floats_and_to_floats():
    p = Pos.from_floats([0.25, 0.75])
    assert len(p.O) == 4 and p.E == 0
    p.rescale(40)
    assert np.allclose(p.to_floats()[:2], [0.25, 0.75], atol=1e-15)
    p.rescale(700)                                     # deep enough for the Fraction path
    assert np.allclose(p.to_floats()[:2], [0.25, 0.75], atol=1e-15)
    assert p.exact()[0] == F(0.25)
