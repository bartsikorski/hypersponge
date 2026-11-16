"""Frames for the 3D and 4D cameras (design section 13.5), from prototype/camera_ref.py.

Camera space is (right, up, forward); a frame F is a 4x4 orthonormal matrix whose columns are
those three object-space vectors and ana, the normal of the 3D slice.  In 3D mode ana = w and
the fourth axis is switched off.
"""
from __future__ import annotations

import math

import numpy as np

START = np.array([1.45, 1.05, -1.75])          # overview eye, in slice coordinates from the target
LEASH = 30.0                                   # the eye never moves further than this from the centre
PRESETS = {                                    # 4D slices, name: (ana, slice offset h) (design 5.4)
    "menger": ((0, 0, 0, 1), -0.25),           # w = 1/4: exactly the Menger sponge
    "dust": ((0, 0, 0, 1), 0.0),               # w = 1/2: exactly Cantor dust
    "diagonal": ((1, 1, 1, 1), 0.0),
    "face diagonal": ((0, 0, 1, 1), 0.0),
    "cell diagonal": ((0, 1, 1, 1), 0.0),
    "generic": ((1, 2, 3, 4), 0.0),
}
START_PRESET = "diagonal"                      # the 4D view at start-up


def givens(i: int, j: int, angle: float) -> np.ndarray:
    """F @ givens(i, j, a): column i turns towards column j by angle a."""
    G = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def orthonormalize(F: np.ndarray) -> np.ndarray:
    """Closest orthonormal matrix (polar decomposition: U @ Vt of the SVD)."""
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
    right = np.cross(up_s, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    F = B.copy()
    F[:, :3] = B[:, :3] @ np.column_stack([right, up, fwd])
    return F


def slice_segments(ana, h: float, n: int = 4) -> list:
    """Line segments where the hyperplane (p - c) . ana = h cuts the square faces of the unit
    n-cube (n = 4: 24 faces).  Together they are the edges of the cross-section polyhedron."""
    a = np.asarray(ana, float)
    c = np.full(4, 0.5)
    c[n:] = 0.0
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
                    p = base.copy()
                    p[i] = ui
                    p[j] = uj
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
