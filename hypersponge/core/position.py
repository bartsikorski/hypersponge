"""A position stored as base-3 digits: x = (O + f) * 3^-E per axis (design sections 3.2, 13.2).

O is a Python integer of any size and sign (the index of the level-E cell), f a float64 in
[0, 1) (the place inside that cell) and E >= 0 the unit exponent.  One "local unit" is 3^-E
world units.  World coordinates are never stored as floats anywhere else: camera code works in
local units through move().
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

_ALMOST_ONE = float(np.nextafter(1.0, 0.0))


@dataclass
class Pos:
    O: list            # 4 Python ints
    f: np.ndarray      # 4 float64 in [0, 1)
    E: int = 0

    @staticmethod
    def from_floats(x, E: int = 0) -> "Pos":
        """World coordinates (up to 4; missing ones are 0) to a position at unit E.  For a deep
        unit, prefer from_floats(x, 0) followed by rescale(E): that keeps every digit of the
        floats exactly."""
        v = np.zeros(4)
        x = np.asarray(x, dtype=np.float64).ravel()
        v[:len(x)] = x
        v = v * 3.0 ** E
        fl = np.floor(v)
        return Pos([int(c) for c in fl], v - fl, int(E))

    def copy(self) -> "Pos":
        return Pos(list(self.O), self.f.copy(), self.E)

    def move(self, d) -> None:
        """Add a vector in local units and carry whole cells into O."""
        g = self.f + np.asarray(d, dtype=np.float64)
        fl = np.floor(g)
        self.f = g - fl
        self.f[self.f >= 1.0] = _ALMOST_ONE
        self.O = [o + int(c) for o, c in zip(self.O, fl)]

    def rescale(self, dE: int) -> None:
        """Change the unit by 3^-dE without moving (dE > 0: a finer unit)."""
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
            self.f[self.f >= 1.0] = _ALMOST_ONE
            self.E -= 1
            dE += 1

    def exact(self) -> list:
        """World coordinates as Fractions (f is exact: a float is a dyadic rational)."""
        s = Fraction(3) ** self.E
        return [(Fraction(o) + Fraction(float(fi))) / s for o, fi in zip(self.O, self.f)]

    def to_floats(self) -> np.ndarray:
        """World coordinates as float64, for the inset map and the status bar."""
        if self.E > 600:
            return np.array([float(Fraction(o, 3 ** self.E)) for o in self.O])
        s = 3 ** self.E
        return np.array([o / s for o in self.O]) + self.f * 3.0 ** -self.E
