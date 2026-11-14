"""Hypersponge camera rules (prototype, CPU only): frames, regimes, pivots, the motion rule,
wheel, drags, slice turns and slides, presets, and the inset map's slice outline.

Camera space is (right, up, forward); the frame F holds those three object-space vectors and
ana (the slice normal) as its columns.  In 3D mode the fourth axis is switched off and ana = w.
"""
from __future__ import annotations

import math

import numpy as np

from sponge_ref import Pos, level_table, normalize, table_rows, free_distance

LEASH = 30.0                                   # the eye never moves further than this from the centre
START = np.array([1.45, 1.05, -1.75])          # overview eye, in slice coordinates from the target
PRESETS = {                                    # name: (ana, slice offset h)
    "menger": ((0, 0, 0, 1), -0.25),           # w = 1/4: exactly the Menger sponge
    "dust": ((0, 0, 0, 1), 0.0),               # w = 1/2: exactly Cantor dust
    "diagonal": ((1, 1, 1, 1), 0.0),
    "face diagonal": ((0, 0, 1, 1), 0.0),
    "cell diagonal": ((0, 1, 1, 1), 0.0),
    "generic": ((1, 2, 3, 4), 0.0),
}


def givens(i: int, j: int, angle: float) -> np.ndarray:
    """F @ givens(i, j, a): column i turns towards column j by angle a."""
    G = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s
    return G


def orthonormalize(F: np.ndarray) -> np.ndarray:
    """Closest orthonormal matrix (polar decomposition)."""
    U, _, Vt = np.linalg.svd(F)
    return U @ Vt


def slice_basis(ana) -> np.ndarray:
    """4x4 frame (b1, b2, b3, ana): Gram-Schmidt of the object axes, skipping the one most
    parallel to ana, with determinant +1.  ana = w gives the identity."""
    a = np.asarray(ana, dtype=np.float64)
    a = a / np.linalg.norm(a)
    drop = int(np.argmax(np.abs(a)))
    cols = []
    for i in range(4):
        if i == drop:
            continue
        v = np.eye(4)[i] - a[i] * a
        for b in cols:
            v = v - (v @ b) * b
        cols.append(v / np.linalg.norm(v))
    M = np.column_stack(cols + [a])
    if np.linalg.det(M) < 0:
        M[:, 2] = -M[:, 2]
    return M


def look_frame(B: np.ndarray, eye_s, target_s=(0.0, 0.0, 0.0), up_s=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Frame looking from eye_s to target_s (slice coordinates of basis B)."""
    fwd = np.asarray(target_s, float) - np.asarray(eye_s, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up_s, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    F = B.copy()
    F[:, :3] = B[:, :3] @ np.column_stack([right, up, fwd])
    return F


class CameraND:
    def __init__(self, n: int, m: int = 2, max_level=None, fov: float = 50.0, height: int = 1080,
                 view_range: float = 40.0):
        assert n in (3, 4)
        self.n, self.m, self.max_level = n, m, max_level
        self.pix = 2.0 * math.tan(math.radians(fov) / 2.0) / height
        self.view_range = view_range
        self.c = np.array([0.5, 0.5, 0.5, 0.5 if n == 4 else 0.0])
        self.R = math.sqrt(n) / 2.0
        self.preset("diagonal" if n == 4 else "menger")

    # ---- state -------------------------------------------------------------------------
    def eye_world(self) -> np.ndarray:
        p = self.pos
        return np.array([o / 3 ** p.E for o in p.O]) + p.f / 3.0 ** p.E if p.E < 600 else \
            np.array([float(o / 3 ** p.E) for o in p.O])

    def rel(self) -> np.ndarray:
        """Eye offset from the centre in camera-frame coordinates (world units)."""
        return self.F.T @ (self.eye_world() - self.c)

    def rho(self) -> float:
        return float(np.linalg.norm(self.rel()[:3]))

    def h(self) -> float:
        return float(self.rel()[3])

    def overview(self) -> bool:
        return self.rho() >= 1.05 * self.R

    @property
    def sigma(self) -> float:                      # scene scale in local units
        return 3.0 ** (self.log_sigma + self.pos.E)

    def w_ref(self):
        """W-depth style: (neutral w offset, scale) in local units.  In overview the colours span
        the whole cube; immersed they are relative to the eye and scaled by sigma."""
        b = min(max((self.rho() - 0.8 * self.R) / (0.25 * self.R), 0.0), 1.0)
        if b == 0.0:
            return (0.0, self.sigma)
        unit = 3.0 ** self.pos.E
        return (b * (self.c[3] - self.eye_world()[3]) * unit, (1.0 - b) * self.sigma + b * unit)

    def refresh(self) -> None:
        """Unit choice and level table after any change of position."""
        self.de = normalize(self.pos, self.n, self.m, self.max_level)
        s = min(max(self.sigma, self.de), 3.0 ** 8 * self.de)
        self.log_sigma = math.log(s, 3.0) - self.pos.E
        self.far = self.view_range * s
        self.table = level_table(self.pos, self.n, self.m, self.max_level, far=self.far,
                                 fine_min=0.3 * self.de * self.pix)
        self.rows = table_rows(self.table)

    def free(self, u):
        return free_distance(self.rows, [float(v) for v in u], self.pix, self.far)

    def update_sigma(self, dt: float = 1e9) -> None:
        t, hit = self.free(self.F[:, 2])
        target = t if hit else min(2.0 * self.sigma, max(np.linalg.norm(self.eye_world() - self.c) * 3.0 ** self.pos.E, self.de))
        target = min(max(target, self.de), 3.0 ** 8 * self.de)
        k = 1.0 - math.exp(-dt / 0.25)
        self.log_sigma += (math.log(target, 3.0) - self.pos.E - self.log_sigma) * k
        self.refresh()

    # ---- motion ------------------------------------------------------------------------
    def translate(self, u, s: float) -> float:
        """The motion rule: never more than half the free distance.  Returns the step taken."""
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
        """Rotation in the camera's own (i, j) plane.  Overview: about the centre (the eye keeps
        its camera-frame coordinates).  Immersed: about the eye."""
        Fn = self.F @ givens(i, j, angle)
        if self.overview():
            y = self.rel()
            self.pos.move((Fn - self.F) @ y * 3.0 ** self.pos.E)
            self.F = orthonormalize(Fn)
            self.refresh()
            self.keep_slice_on_cube()
        else:
            self.F = orthonormalize(Fn)

    def keep_slice_on_cube(self) -> None:
        """Overview, 4D: after ana changed, pull the slice back so that it still meets the cube."""
        if self.n == 4 and abs(self.h()) > self.offset_range():
            self.set_offset(math.copysign(self.offset_range(), self.h()))

    def rotate_object(self, i: int, j: int, angle: float) -> None:
        """Plane jog: rotation in the object's own (i, j) plane, same pivot rule as rotate()."""
        G = givens(i, j, angle)
        if self.overview():
            d = self.eye_world() - self.c
            self.pos.move((G @ d - d) * 3.0 ** self.pos.E)
            self.F = orthonormalize(G @ self.F)
            self.refresh()
            self.keep_slice_on_cube()
        else:
            self.F = orthonormalize(G @ self.F)

    def roll(self, angle: float) -> None:
        self.rotate(0, 1, angle)

    def drag_rotate(self, dx: float, dy_up: float, k: float = math.radians(0.3)) -> None:
        s = -1.0 if self.overview() else 1.0           # the scene follows the mouse in both regimes
        self.rotate(0, 2, s * k * dx)
        self.rotate(1, 2, s * k * dy_up)

    def drag_turn(self, dx: float, dy_up: float, k: float = math.radians(0.3)) -> None:
        self.rotate(0, 3, k * dx)
        self.rotate(1, 3, k * dy_up)

    def wheel(self, notches: float, dir_cam=(0.0, 0.0, 1.0)) -> None:
        D = self.F[:, :3] @ (np.asarray(dir_cam, float) / np.linalg.norm(dir_cam))
        t, hit = self.free(D)
        to_c = (self.c - self.eye_world()) * 3.0 ** self.pos.E
        if notches > 0:
            if not hit:                            # nothing under the cursor: go to the point of
                t = max(0.0, float(to_c @ D))      # the ray nearest the centre, never past it
            self.translate(D, t * (1.0 - 0.8 ** notches))
        else:
            if not hit:
                t = float(np.linalg.norm(to_c))
            self.translate(-D, t * (1.25 ** (-notches) - 1.0))

    def pan(self, dx: float, dy_up: float, grab: float) -> None:
        v = -(dx * self.F[:, 0] + dy_up * self.F[:, 1]) * grab * self.pix
        s = float(np.linalg.norm(v))
        if s > 0:
            self.translate(v / s, s)

    def fly(self, v_cam, dt: float, fast: bool = False) -> None:
        v = self.F[:, :3] @ np.asarray(v_cam, float)
        s = float(np.linalg.norm(v))
        if s > 0:
            self.translate(v / s, self.sigma * dt * (4.0 if fast else 1.0))

    def slide(self, notches: float) -> float:
        s = math.copysign(0.08 * self.sigma * abs(notches), notches)       # local units
        if self.overview():                        # the slice stays on the cube
            H, h = self.offset_range(), self.h()
            s = (min(max(h + s * 3.0 ** -self.pos.E, -H), H) - h) * 3.0 ** self.pos.E
        return self.translate(math.copysign(1.0, s) * self.F[:, 3], abs(s))

    def offset_range(self) -> float:
        return 0.5 * float(np.abs(self.F[:, 3]).sum())

    def set_offset(self, h: float) -> None:
        assert self.overview()
        h = min(max(h, -self.offset_range()), self.offset_range())
        self.pos.move(self.F[:, 3] * (h - self.h()) * 3.0 ** self.pos.E)
        self.refresh()

    def preset(self, name: str) -> None:
        ana, h = PRESETS[name]
        B = slice_basis(ana)
        self.F = look_frame(B, START)
        self.pos = Pos.from_floats(self.c + h * B[:, 3] + B[:, :3] @ START, 0)
        self.log_sigma = 0.0
        self.refresh()
        self.update_sigma()


# ---- inset map ---------------------------------------------------------------------------
INSET_PROJ = np.array([[1.0, 0.0], [0.0, 1.0], [0.45, 0.30], [-0.38, 0.52]])   # x, y, z, w -> 2D


def slice_segments(ana, h: float, n: int = 4):
    """Line segments where the hyperplane (p - c) . ana = h cuts the square faces of the unit
    n-cube (n = 4: 24 faces).  Together they are the edges of the cross-section polyhedron."""
    a = np.asarray(ana, float)
    c = np.full(4, 0.5); c[n:] = 0.0
    segs = []
    for i in range(n):
        for j in range(i + 1, n):
            others = [k for k in range(n) if k not in (i, j)]
            for bits in range(1 << len(others)):
                base = np.zeros(4)
                for b, k in enumerate(others):
                    base[k] = (bits >> b) & 1
                corners = []
                for (ui, uj) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = base.copy(); p[i] = ui; p[j] = uj
                    corners.append(p)
                s = [float((p - c) @ a - h) for p in corners]
                s = [v if v != 0.0 else 1e-12 for v in s]
                pts = []
                for e in range(4):
                    p0, p1, s0, s1 = corners[e], corners[(e + 1) % 4], s[e], s[(e + 1) % 4]
                    if (s0 < 0) != (s1 < 0):
                        pts.append(p0 + (p1 - p0) * (s0 / (s0 - s1)))
                if len(pts) == 2 and np.linalg.norm(pts[0] - pts[1]) > 1e-9:    # skip corner touches
                    segs.append((pts[0], pts[1]))
    return segs
