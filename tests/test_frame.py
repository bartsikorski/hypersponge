"""Frames (design 13.5), from prototype/check_camera_rules.py."""
import math
import random

import numpy as np
import pytest

from hypersponge.core.frame import START, givens, look_frame, orthonormalize, slice_basis, slice_segments


@pytest.mark.parametrize("ana", [(0, 0, 0, 1), (1, 1, 1, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 2, 3, 4),
                                 (-0.3, 0.2, 0.9, -0.1)])
def test_slice_basis_is_a_rotation_ending_in_ana(ana):
    B = slice_basis(ana)
    assert np.allclose(B.T @ B, np.eye(4), atol=1e-12) and abs(np.linalg.det(B) - 1) < 1e-12
    a = np.asarray(ana, float)
    assert np.allclose(B[:, 3], a / np.linalg.norm(a))


def test_w_gives_the_identity():
    assert np.allclose(slice_basis((0, 0, 0, 1)), np.eye(4))


def test_look_frame_looks_at_the_target():
    for ana in ((0, 0, 0, 1), (1, 2, 3, 4)):
        B = slice_basis(ana)
        F = look_frame(B, START)
        assert np.allclose(F.T @ F, np.eye(4), atol=1e-12) and abs(np.linalg.det(F) - 1) < 1e-9
        to_target = -(B[:, :3] @ START)
        assert np.allclose(F[:, 2], to_target / np.linalg.norm(to_target), atol=1e-12)
        assert np.allclose(F[:, 3], B[:, 3])
        assert F[:, 1] @ (B[:, :3] @ np.array([0, 1, 0])) > 0          # up stays up


def test_givens_turns_column_i_towards_j_and_frames_stay_orthonormal():
    F = np.eye(4) @ givens(0, 2, 0.1)
    assert np.allclose(F[:, 0], [math.cos(0.1), 0, math.sin(0.1), 0])
    rnd = random.Random(11)
    F = np.eye(4)
    for k in range(100000):
        i, j = rnd.sample(range(4), 2)
        F = F @ givens(i, j, rnd.uniform(-0.3, 0.3))
        if k % 16 == 0:
            F = orthonormalize(F)
    assert np.allclose(F.T @ F, np.eye(4), atol=1e-12)


def test_slice_segments():
    seg = slice_segments((0, 0, 0, 1), 0.0)
    assert len(seg) == 12 and all(abs(np.linalg.norm(a - b) - 1) < 1e-12 for a, b in seg)        # a cube
    seg = slice_segments(np.ones(4) / 2, 0.0)
    assert len(seg) == 12 and all(abs(np.linalg.norm(a - b) - math.sqrt(2)) < 1e-9 for a, b in seg)  # octahedron
    assert len(slice_segments(np.ones(4) / 2, 1.01)) == 0
    assert len(slice_segments(np.ones(4) / 2, 0.9)) == 6                                          # tetrahedron
    assert len(slice_segments((0, 0, 1, 0), 0.2, n=3)) == 4                                        # 3D: a square
