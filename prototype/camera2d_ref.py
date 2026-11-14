"""Hypersponge 2D camera rules (prototype, CPU only): a centre position and a view height."""
from __future__ import annotations

import math

import numpy as np

from sponge_ref import Pos, level_table, normalize_2d

MAX_HEIGHT = 4.0                                   # widest view, in world units (cube edge = 1)


def safe_fraction(u: float) -> float:
    """A point near u in [0, 1) whose base-3 digits contain no 1.  At the first digit 1 the
    rest becomes 0202... (lower half of that middle third) or 2020... (upper half), which
    moves the point by at most a quarter of that cell."""
    base, scale = 0.0, 1.0
    for _ in range(34):
        u3 = u * 3.0
        d = min(int(u3), 2)
        if d == 1:
            return base + scale * (0.25 if u3 - 1.0 < 0.5 else 0.75)
        base += scale * d / 3.0
        scale /= 3.0
        u = u3 - d
    return base


class Camera2D:
    def __init__(self, m: int = 2, max_level=None, width: int = 1920, height: int = 1080):
        self.m, self.max_level, self.W, self.H = m, max_level, width, height
        self.reset()

    def reset(self) -> None:
        self.pos = Pos.from_floats([0.5, 0.5, 0.0, 0.0], 0)
        self.height = 1.25                         # view height in local units
        self.refresh()

    @property
    def pixel(self) -> float:                      # size of one pixel in local units
        return self.height / self.H

    def world_height(self) -> float:
        return self.height * 3.0 ** -self.pos.E if self.pos.E < 600 else 0.0

    def refresh(self) -> None:
        """Clamp the centre to the unit square, choose the unit, build the table."""
        full = 3 ** self.pos.E
        for i in range(2):
            if self.pos.O[i] < 0:
                self.pos.O[i] = 0; self.pos.f[i] = 0.0
            elif self.pos.O[i] >= full:
                self.pos.O[i] = full - 1; self.pos.f[i] = np.nextafter(1.0, 0.0)
        self.height = normalize_2d(self.pos, self.height)
        diag = float(np.hypot(self.W, self.H)) * self.pixel
        self.table = level_table(self.pos, 2, self.m, self.max_level, far=0.55 * diag, fine_min=self.pixel / 3.0,
                                 cover=True)

    def zoom(self, notches: float, cursor=(0.0, 0.0)) -> None:
        """Wheel: 1.25x per notch; the point under the cursor (pixels from the window centre,
        y up) stays under the cursor."""
        k = 0.8 ** notches
        if self.pos.E == 0:
            k = min(k, MAX_HEIGHT / self.height)   # zoom-out limit
        q = np.array([cursor[0], cursor[1], 0.0, 0.0]) * self.pixel
        if notches > 0:
            self.pos.move(self.anchor_shift(q))
        self.pos.move(q * (1.0 - k))
        self.height *= k
        self.refresh()

    def anchor_shift(self, q) -> np.ndarray:
        """Zoom anchor rule: a shift of under a quarter of a pixel after which the point under
        the cursor has no digit 1 below the finest visible level, so zooming in on material
        that is visible on screen never ends inside a hole that was too small to see."""
        j = math.floor(math.log(3.0 / self.pixel, 3.0) + 1e-9)
        s = 3.0 ** -j                              # finest visible slab width: pixel/3 <= s < pixel
        d = np.zeros(4)
        for i in range(2):
            x = (self.pos.f[i] + q[i]) / s
            cell = math.floor(x)
            d[i] = (cell + safe_fraction(x - cell)) * s - (self.pos.f[i] + q[i])
        return d

    def pan(self, dx: float, dy_up: float) -> None:
        """Drag: the picture follows the mouse."""
        self.pos.move(np.array([-dx, -dy_up, 0.0, 0.0]) * self.pixel)
        self.refresh()

    def fly(self, vx: float, vy: float, dt: float, fast: bool = False) -> None:
        """Keys: one view height per second, 4x with Shift."""
        s = self.height * dt * (4.0 if fast else 1.0)
        self.pos.move(np.array([vx, vy, 0.0, 0.0]) * s)
        self.refresh()
