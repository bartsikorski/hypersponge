"""Cameras (design section 15).  Pure Python and numpy: no OpenGL, so every rule is unit-tested.

Camera2D: a centre position and a view height.  CameraND: an eye and a frame, for 3D.  Cursor
positions and drags are given in framebuffer pixels measured from the window centre, with y up.
"""
from __future__ import annotations

import math

import numpy as np

from .frame import LEASH, START, givens, look_frame, orthonormalize
from .levels import Pos, level_table, normalize, normalize_2d
from .rule import Rule
from .sdf import free_distance, table_rows

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


# ------------------------------------------------------------------------------------------
# 3D (4D in stage 3): an eye position and a frame (design sections 15.2 and 15.3)
# ------------------------------------------------------------------------------------------
FOV = 50.0                # field of view in degrees; the panel slider runs from 30 to 90
VIEW_RANGE = 40.0         # view distance in units of sigma; the panel slider runs from 10 to 300
SIGMA_EPS = 1e-3          # update_sigma leaves log_sigma alone below this change per call
DRAG_K = math.radians(0.3)            # radians per pixel dragged


class CameraND:
    """The 3D camera (n = 3).  pos is the eye, F the frame (right, up, forward, ana).

    Every translation goes through translate(), which never moves more than half the free
    distance in its direction, so the eye can approach a surface for ever and never enter solid.
    Rotations pivot on the cube's centre in overview and on the eye when immersed.  Ported from
    prototype/camera_ref.py, which passes check_camera_rules.py.
    """

    mode = "3d"

    def __init__(self, rule: Rule | None = None, width: int = 1280, height: int = 720,
                 fov: float = FOV, view_range: float = VIEW_RANGE):
        self.rule = rule or Rule(3, 2)
        if self.rule.n != 3:
            raise ValueError("CameraND supports n = 3 in this version")
        self.n = self.rule.n
        self.W, self.H = int(width), int(height)
        self.fov = float(fov)
        self.view_range = float(view_range)
        self.c = np.array([0.5, 0.5, 0.5, 0.5 if self.n == 4 else 0.0])
        self.R = math.sqrt(self.n) / 2.0
        self._good = None
        self.reset()

    # ---- state -------------------------------------------------------------------------------
    @property
    def pix(self) -> float:
        """Angle of one window pixel: 2 tan(fov / 2) / window height."""
        return 2.0 * math.tan(math.radians(self.fov) / 2.0) / self.H

    @property
    def depth(self) -> int:
        return self.pos.E

    @property
    def sigma(self) -> float:
        """Scene scale in local units."""
        return 3.0 ** (self.log_sigma + self.pos.E)

    def eye_world(self) -> np.ndarray:
        return self.pos.to_floats()

    def rel(self) -> np.ndarray:
        """Eye offset from the centre in camera-frame coordinates, world units."""
        return self.F.T @ (self.eye_world() - self.c)

    def rho(self) -> float:
        return float(np.linalg.norm(self.rel()[:3]))

    def h(self) -> float:
        return float(self.rel()[3])

    def overview(self) -> bool:
        return self.rho() >= 1.05 * self.R

    def key(self):
        """Everything about the camera that changes the picture."""
        return (self.mode, tuple(self.pos.O), self.pos.f.tobytes(), self.pos.E, self.F.tobytes(), self.fov,
                self.far, self.W, self.H, self.rule)

    def reset(self) -> None:
        """Overview start: eye at c + START in slice coordinates, looking at c (design 15.3)."""
        B = np.eye(4)
        self.F = look_frame(B, START)
        self.pos = Pos.from_floats(self.c + B[:, :3] @ START, 0)
        self.log_sigma = 0.0
        self._good = None
        self.refresh()
        self.update_sigma()

    def set_rule(self, rule: Rule) -> None:
        """A new m or max level can put the eye inside solid; then the camera resets (15.5)."""
        if rule.n != self.n:
            raise ValueError(f"this camera needs a {self.n}D rule")
        self.rule = rule
        try:
            self._refresh()
        except ValueError:
            self.reset()

    def resize(self, width: int, height: int) -> None:
        if (int(width), int(height)) != (self.W, self.H):
            self.W, self.H = int(width), int(height)
            self.refresh()

    def set_fov(self, fov: float) -> None:
        if float(fov) != self.fov:
            self.fov = float(fov)
            self.refresh()

    def set_view_range(self, view_range: float) -> None:
        if float(view_range) != self.view_range:
            self.view_range = float(view_range)
            self.refresh()

    def _refresh(self) -> None:
        r = self.rule
        self.de = normalize(self.pos, self.n, r.m, r.max_level)
        s = min(max(self.sigma, self.de), 3.0 ** 8 * self.de)
        self.log_sigma = math.log(s, 3.0) - self.pos.E
        self.far = self.view_range * s
        self.table = level_table(self.pos, self.n, r.m, r.max_level, far=self.far,
                                 fine_min=0.3 * self.de * self.pix)
        self.rows = table_rows(self.table)
        self._good = (self.pos.copy(), self.F.copy(), self.log_sigma)

    def refresh(self) -> None:
        """Choose the unit and build the table after any change of position.  If the eye is in
        solid, which the motion rule prevents, go back to the last good state."""
        try:
            self._refresh()
        except ValueError:
            if self._good is None:
                raise
            pos, F, ls = self._good
            self.pos, self.F, self.log_sigma = pos.copy(), F.copy(), ls
            self._refresh()

    def free(self, u):
        """Distance to the first surface along the unit 4-vector u (local units) and whether it
        hit, with the picture's level of detail."""
        return free_distance(self.rows, [float(v) for v in u], self.pix, self.far)

    def update_sigma(self, dt: float = 1e9) -> bool:
        """Move sigma towards the free distance along forward, time constant 0.25 s.  Returns
        True when it changed.  A change below SIGMA_EPS is not applied, so that a still view
        stays still and can accumulate (design 16.2)."""
        t, hit = self.free(self.F[:, 2])
        if hit:
            target = t
        else:
            to_c = float(np.linalg.norm(self.eye_world() - self.c)) * 3.0 ** self.pos.E
            target = min(2.0 * self.sigma, max(to_c, self.de))
        target = min(max(target, self.de), 3.0 ** 8 * self.de)
        k = 1.0 - math.exp(-dt / 0.25)
        delta = (math.log(target, 3.0) - self.pos.E - self.log_sigma) * k
        if abs(delta) < SIGMA_EPS:
            return False
        self.log_sigma += delta
        self.refresh()
        return True

    # ---- motion ------------------------------------------------------------------------------
    def translate(self, u, s: float) -> float:
        """The motion rule: a step along the unit vector u of at most half the free distance,
        and never further out than LEASH world units.  Returns the step taken (local units)."""
        u = np.asarray(u, dtype=np.float64)
        t, _ = self.free(u)
        step = min(float(s), 0.5 * t)
        d0 = self.eye_world() - self.c
        d1 = d0 + u * step * 3.0 ** -self.pos.E
        if np.linalg.norm(d1) > LEASH and np.linalg.norm(d1) > np.linalg.norm(d0):
            return 0.0
        if step > 0.0:
            self.pos.move(u * step)
            self.refresh()
        return step

    def rotate(self, i: int, j: int, angle: float) -> None:
        """Rotation in the camera's own (i, j) plane: about the centre in overview, where the eye
        keeps its camera-frame coordinates, and about the eye when immersed."""
        Fn = self.F @ givens(i, j, angle)
        if self.overview():
            y = self.rel()
            self.pos.move((Fn - self.F) @ y * 3.0 ** self.pos.E)
            self.F = orthonormalize(Fn)
            self.refresh()
        else:
            self.F = orthonormalize(Fn)          # the table does not depend on F

    def roll(self, angle: float) -> None:
        self.rotate(0, 1, angle)

    def drag_rotate(self, dx: float, dy_up: float, k: float = DRAG_K) -> None:
        """Left-drag: the scene follows the mouse (orbit in overview, look around immersed)."""
        s = -1.0 if self.overview() else 1.0
        if dx:
            self.rotate(0, 2, s * k * dx)
        if dy_up:
            self.rotate(1, 2, s * k * dy_up)

    def cursor_dir(self, cursor) -> np.ndarray:
        """Camera-space ray of a cursor offset (window pixels from the centre, y up), not unit."""
        return np.array([cursor[0] * self.pix, cursor[1] * self.pix, 1.0])

    def ray(self, cursor) -> np.ndarray:
        """Object-space unit ray under a cursor offset."""
        d = self.cursor_dir(cursor)
        return self.F[:, :3] @ (d / np.linalg.norm(d))

    def wheel(self, notches: float, dir_cam=(0.0, 0.0, 1.0)) -> None:
        """Move 20% of the way to the surface under the cursor per notch; back undoes it.  With
        nothing under the cursor, aim at the point of the ray nearest the centre."""
        if notches == 0:
            return
        D = self.F[:, :3] @ (np.asarray(dir_cam, float) / np.linalg.norm(dir_cam))
        t, hit = self.free(D)
        to_c = (self.c - self.eye_world()) * 3.0 ** self.pos.E
        if notches > 0:
            if not hit:
                t = max(0.0, float(to_c @ D))
            self.translate(D, t * (1.0 - 0.8 ** notches))
        else:
            if not hit:
                t = float(np.linalg.norm(to_c))
            self.translate(-D, t * (1.25 ** (-notches) - 1.0))

    def zoom(self, notches: float, cursor=(0.0, 0.0)) -> None:
        """The wheel with a cursor offset in window pixels, as Camera2D.zoom takes it."""
        self.wheel(notches, self.cursor_dir(cursor))

    def grab(self, cursor) -> float:
        """Right-drag scale: the free distance under the cursor, or sigma on a miss."""
        t, hit = self.free(self.ray(cursor))
        return t if hit else self.sigma

    def pan(self, dx: float, dy_up: float, grab: float | None = None) -> None:
        """Right-drag: slide so that the point under the cursor follows the mouse."""
        g = self.sigma if grab is None else grab
        v = -(dx * self.F[:, 0] + dy_up * self.F[:, 1]) * g * self.pix
        s = float(np.linalg.norm(v))
        if s > 0:
            self.translate(v / s, s)

    def fly(self, v_cam, dt: float, fast: bool = False) -> None:
        """Keys: sigma per second along the camera-space direction v_cam; Shift is 4 times faster."""
        v = self.F[:, :3] @ np.asarray(v_cam, float)
        s = float(np.linalg.norm(v))
        if s > 0:
            self.translate(v / s, self.sigma * dt * (4.0 if fast else 1.0))
