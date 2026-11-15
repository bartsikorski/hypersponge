"""Camera frames (design sections 12.1 and 13.5): Givens rotations, orthonormalising, the slice
basis and the look-at frame.  Ported from prototype/camera_ref.py.

A frame F is a 4x4 matrix whose columns are right, up, forward and ana in object axes.  In 3D
mode ana = w and the fourth axis is switched off.
"""
from __future__ import annotations

import math

import numpy as np

START = np.array([1.45, 1.05, -1.75])     # overview eye in slice coordinates from the target
LEASH = 30.0                              # the eye never moves further than this from the centre


def givens(i: int, j: int, angle: float) -> np.ndarray:
    """F @ givens(i, j, a) turns column i of F towards column j by the angle a."""
    G = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def orthonormalize(F: np.ndarray) -> np.ndarray:
    """The nearest orthonormal matrix (U @ Vt of the SVD)."""
    U, _, Vt = np.linalg.svd(F)
    return U @ Vt


def slice_basis(ana) -> np.ndarray:
    """4x4 frame (b1, b2, b3, ana) with determinant +1: Gram-Schmidt of the object axes in index
    order, skipping the one most parallel to ana.  ana = w gives the identity."""
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
    """The frame looking from eye_s to target_s, both in slice coordinates of the basis B."""
    fwd = np.asarray(target_s, float) - np.asarray(eye_s, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up_s, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    F = B.copy()
    F[:, :3] = B[:, :3] @ np.column_stack([right, up, fwd])
    return F
