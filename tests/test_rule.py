"""The rule family (design section 2), from prototype/check_estimator.py (T1)."""
import itertools

import pytest

from hypersponge.core.rule import NAMES, Rule, kept_cells


@pytest.mark.parametrize("n", [2, 3, 4])
def test_kept_cells_match_brute_force(n):
    for m in range(1, n + 1):
        c = sum(1 for d in itertools.product(range(3), repeat=n) if sum(x == 1 for x in d) < m)
        assert c == kept_cells(n, m) == Rule(n, m).kept_cells()


def test_names_and_labels():
    assert set(NAMES) == {(n, m) for n in (2, 3, 4) for m in range(1, n + 1)}
    assert Rule(2, 2).name() == "Sierpinski carpet"
    assert Rule(3, 2).label() == "Menger sponge: 20 of 27 kept, dimension 2.727"
    assert Rule(2, 2).label() == "Sierpinski carpet: 8 of 9 kept, dimension 1.893"
    assert Rule(2, 1).kept_cells() == 4 and abs(Rule(2, 1).dimension() - 1.2619) < 1e-4


@pytest.mark.parametrize("args", [(1, 1), (5, 2), (2, 0), (2, 3), (3, 4), (2, 2, 0)])
def test_bad_rules_are_refused(args):
    with pytest.raises(ValueError):
        Rule(*args)


def test_rule_is_a_value():
    assert Rule(2, 2) == Rule(2, 2, None) and Rule(2, 2, 3) != Rule(2, 2)
    assert len({Rule(2, 2), Rule(2, 2), Rule(2, 1)}) == 2
