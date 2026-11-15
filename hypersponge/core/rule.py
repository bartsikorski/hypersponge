"""The fractal family: the unit cube [0,1]^n with every point removed whose coordinates have
digit 1 (base 3) at the same level on at least m axes (design section 2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

NAMES = {
    (2, 1): "Cantor dust",
    (2, 2): "Sierpinski carpet",
    (3, 1): "Cantor dust",
    (3, 2): "Menger sponge",
    (3, 3): "Centre removed",
    (4, 1): "Cantor dust",
    (4, 2): "4D Menger sponge",
    (4, 3): "Denser 4D sponge",
    (4, 4): "Centre removed",
}

MAX_LEVEL_LIMIT = 12          # the panel's max level slider goes from 1 to this; None = unlimited


@dataclass(frozen=True)
class Rule:
    n: int                          # 2, 3 or 4
    m: int = 2                      # 1 .. n
    max_level: int | None = None    # None = unlimited

    def __post_init__(self):
        if self.n not in (2, 3, 4):
            raise ValueError(f"dimension must be 2, 3 or 4, not {self.n}")
        if not 1 <= self.m <= self.n:
            raise ValueError(f"rule m must be between 1 and {self.n}, not {self.m}")
        if self.max_level is not None and self.max_level < 1:
            raise ValueError("max level must be at least 1")

    def kept_cells(self) -> int:
        """Cells kept out of 3^n at each level: sum over j < m of C(n, j) * 2^(n - j)."""
        return kept_cells(self.n, self.m)

    def dimension(self) -> float:
        return math.log(self.kept_cells()) / math.log(3.0)

    def name(self) -> str:
        return NAMES[(self.n, self.m)]

    def label(self) -> str:
        """Panel text, for example 'Menger sponge: 20 of 27 kept, dimension 2.727'."""
        return f"{self.name()}: {self.kept_cells()} of {3 ** self.n} kept, dimension {self.dimension():.3f}"


def kept_cells(n: int, m: int) -> int:
    return sum(math.comb(n, j) * 2 ** (n - j) for j in range(m))
