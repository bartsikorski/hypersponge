"""Cameras (design section 15).  Pure Python and numpy: no OpenGL, so every rule is unit-tested.

Camera2D: a centre position and a view height.  Cursor positions and drags are given in
framebuffer pixels measured from the window centre, with y up.
"""
from __future__ import annotations

import math

import numpy as np

from .levels import Pos, level_table, normalize_2d
from .rule import Rule

MAX_HEIGHT = 4.0          # widest 2D view, in world units (the square's edge is 1)
START_HEIGHT = 1.25       # 2D view height at start and after Reset, world units
CUT = 1.0 / 3.0           # 2D detail cut in pixels: rows narrower than this are not drawn
DUST_CUT = 2.0            # ... for rule m = 1, whose grains would otherwise be a faint haze


def safe_fraction(u: float) -> float:
    """A point near u in [0, 1) whose base-3 digits contain no 1.  At the first digit 1 the rest
    becomes 0202... (lower half of that middle third) or 2020... (upper half), which moves the
    point by at most a quarter of the cell."""
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
    """The 2D view: pos is the view centre, height the view height in local units."""

    mode = "2d"

    def __init__(self, rule: Rule | None = None, width: int = 1280, height: int = 720):
        self.rule = rule or Rule(2, 2)
        self.W, self.H = int(width), int(height)
        self.reset()

    # ---- state -------------------------------------------------------------------------------
    def reset(self) -> None:
        self.set_view((0.5, 0.5), START_HEIGHT)

    def set_view(self, center, height: float) -> None:
        """Centre in world coordinates and view height in world units.  The centre's floats are
        kept exactly, at any depth."""
        self.pos = Pos.from_floats([center[0], center[1], 0.0, 0.0], 0)
        self.height = float(height)
        self.refresh()

    def set_rule(self, rule: Rule) -> None:
        if rule.n != 2:
            raise ValueError("Camera2D needs a 2D rule")
        self.rule = rule
        self.refresh()

    def resize(self, width: int, height: int) -> None:
        if (int(width), int(height)) != (self.W, self.H):
            self.W, self.H = int(width), int(height)
            self.refresh()

    @property
    def pixel(self) -> float:
        """Size of one framebuffer pixel in local units."""
        return self.height / self.H

    @property
    def cut(self) -> float:
        """Detail cut in pixels: rows narrower than cut * pixel are not drawn."""
        return DUST_CUT if self.rule.m == 1 else CUT

    @property
    def depth(self) -> int:
        return self.pos.E

    def world_height(self) -> float:
        return self.height * 3.0 ** -self.pos.E

    def center_world(self) -> np.ndarray:
        return self.pos.to_floats()[:2]

    def refresh(self) -> None:
        """Clamp the centre to the unit square, choose the unit, build the table."""
        full = 3 ** self.pos.E
        for i in range(2):
            if self.pos.O[i] < 0:
                self.pos.O[i] = 0
                self.pos.f[i] = 0.0
            elif self.pos.O[i] >= full:
                self.pos.O[i] = full - 1
                self.pos.f[i] = float(np.nextafter(1.0, 0.0))
        self.height = normalize_2d(self.pos, self.height)
        self.build_table()

    def build_table(self) -> None:
        """The level table for the current unit, without choosing the unit again."""
        diag = float(np.hypot(self.W, self.H)) * self.pixel
        self.table = level_table(self.pos, 2, self.rule.m, self.rule.max_level, far=0.55 * diag,
                                 fine_min=min(self.cut, CUT) * self.pixel, cover=True)

    def key(self):
        """Everything that changes the picture."""
        return ("2d", tuple(self.pos.O), self.pos.f.tobytes(), self.pos.E, self.height, self.W, self.H,
                self.rule)

    # ---- motion ------------------------------------------------------------------------------
    def zoom(self, notches: float, cursor=(0.0, 0.0)) -> None:
        """Wheel: 1.25 times per notch; the point under the cursor stays under the cursor."""
        if notches == 0:
            return
        k = 0.8 ** notches
        if self.pos.E == 0:
            k = min(k, MAX_HEIGHT / self.height)          # zoom-out limit
        q = np.array([cursor[0], cursor[1], 0.0, 0.0]) * self.pixel
        if notches > 0:
            self.pos.move(self.anchor_shift(q))
        self.pos.move(q * (1.0 - k))
        self.height *= k
        self.refresh()

    def anchor_shift(self, q) -> np.ndarray:
        """Zoom anchor rule (design 15.1): a shift of at most a quarter of the finest drawn cell
        after which the point at q (local units from the centre) has no digit 1 below the finest
        drawn level.  Zooming in on material that is visible therefore never ends in a hole that
        was too small to see."""
        j = math.floor(math.log(1.0 / (self.cut * self.pixel), 3.0) + 1e-9)
        s = 3.0 ** -j                                     # finest drawn slab width, local units
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
        """Keys: one view height per second, 4 times faster with Shift."""
        s = self.height * dt * (4.0 if fast else 1.0)
        self.pos.move(np.array([vx, vy, 0.0, 0.0]) * s)
        self.refresh()

    def cursor_local(self, cursor) -> np.ndarray:
        """Camera-relative point under a cursor offset, local units (4-vector)."""
        return np.array([cursor[0], cursor[1], 0.0, 0.0]) * self.pixel
