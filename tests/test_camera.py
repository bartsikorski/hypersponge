"""The 3D camera (design 15.2, 15.3) and frames (13.5), from prototype/check_camera_rules.py and
check_camera.py.  CPU only."""
import math
import random

import numpy as np
import pytest

from hypersponge.core.camera import CameraND
from hypersponge.core.frame import LEASH, START, givens, look_frame, orthonormalize, slice_basis
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import de_scalar, de_table, free_distance, table_rows


def orthonormal(F, tol=1e-9):
    return np.allclose(F.T @ F, np.eye(4), atol=tol) and abs(np.linalg.det(F) - 1.0) < tol


def screen(cam, P):
    """Screen position (x / z, y / z) of the world point P."""
    v = cam.F[:, :3].T @ (P - cam.eye_world())
    return v[0] / v[2], v[1] / v[2]


# ---- frames ----------------------------------------------------------------------------------
def test_givens_turns_column_i_towards_j():
    F = np.eye(4) @ givens(0, 2, 0.3)
    assert F[:, 0] == pytest.approx([math.cos(0.3), 0, math.sin(0.3), 0])


def test_frames_stay_orthonormal_after_100000_rotations():
    rng = random.Random(3)
    F = np.eye(4)
    for k in range(100000):
        i, j = rng.sample(range(4), 2)
        F = F @ givens(i, j, rng.uniform(-0.3, 0.3))
        if k % 16 == 0:
            F = orthonormalize(F)
    assert orthonormal(orthonormalize(F), 1e-12)


def test_slice_basis_and_look_frame():
    assert np.allclose(slice_basis((0, 0, 0, 1)), np.eye(4))
    for ana in ((1, 1, 1, 1), (0, 0, 1, 1), (1, 2, 3, 4)):
        B = slice_basis(ana)
        assert orthonormal(B, 1e-12)
        F = look_frame(B, START)
        assert orthonormal(F, 1e-12)
        assert np.allclose(F[:, 2], -B[:, :3] @ START / np.linalg.norm(START))


# ---- state -----------------------------------------------------------------------------------
def test_reset_is_the_overview_start():
    cam = CameraND(Rule(3, 2), 640, 360)
    assert cam.overview() and cam.depth == 0 and orthonormal(cam.F)
    assert cam.rho() == pytest.approx(np.linalg.norm(START), abs=1e-12)
    to_c = cam.c - cam.eye_world()
    assert np.allclose(to_c / np.linalg.norm(to_c), cam.F[:, 2])
    assert 1.0 <= cam.de < 3.0 or cam.depth == 0
    assert cam.de <= cam.sigma <= 3 ** 8 * cam.de


def test_rotations_in_overview_keep_rho():
    cam = CameraND(Rule(3, 2), 640, 360)
    rng = random.Random(11)
    rho0 = cam.rho()
    for _ in range(2000):
        i, j = rng.sample(range(3), 2)
        cam.rotate(i, j, rng.uniform(-0.2, 0.2))
    assert cam.rho() == pytest.approx(rho0, abs=1e-9) and cam.overview() and orthonormal(cam.F)


def test_drag_the_scene_follows_the_mouse():
    cam = CameraND(Rule(3, 2), 640, 360)
    near = cam.eye_world() + cam.F[:, 2] * (cam.rho() - 0.6)
    x0, y0 = screen(cam, near)
    cam.drag_rotate(40, 0)
    x1, _ = screen(cam, near)
    cam.drag_rotate(0, 40)
    _, y1 = screen(cam, near)
    assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and cam.overview()
    for _ in range(14):
        cam.wheel(1)
    assert not cam.overview()
    ahead = cam.eye_world() + cam.F[:, 2] * 0.01
    e0 = cam.eye_world().copy()
    x0, y0 = screen(cam, ahead)
    cam.drag_rotate(40, 0)
    x1, _ = screen(cam, ahead)
    cam.drag_rotate(0, 40)
    _, y1 = screen(cam, ahead)
    assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and np.allclose(cam.eye_world(), e0)   # turned about the eye


def test_wheel_in_then_out_returns():
    cam = CameraND(Rule(3, 2), 640, 360)
    e0 = cam.eye_world().copy()
    cam.wheel(1)
    assert np.linalg.norm(cam.eye_world() - e0) > 0.05
    cam.wheel(-1)
    assert np.linalg.norm(cam.eye_world() - e0) < 0.01 * np.linalg.norm(e0 - cam.c)


def test_the_wheel_alone_reaches_depth_100_without_entering_solid():
    """Stage 2 done-when: from the overview, the wheel flies into a hole and on to 3^-100."""
    cam = CameraND(Rule(3, 2), 1920, 1080)
    notches = 0
    while cam.depth < 100 and notches < 800:
        cam.wheel(1)
        notches += 1
        assert cam.de > 0
    assert cam.depth >= 100 and not cam.overview()
    assert len(cam.table.w) <= 24 and cam.table.dropped == 0


def test_wheel_on_a_miss_stops_beside_the_object_and_the_leash_holds():
    cam = CameraND(Rule(3, 2), 640, 360)
    d_cam = np.array([0.45, 0.30, 1.0])                             # a corner of the picture: a miss
    for _ in range(200):
        cam.wheel(1, d_cam)
    d = cam.eye_world() - cam.c
    D = cam.F[:, :3] @ (d_cam / np.linalg.norm(d_cam))
    assert abs(d @ D) < 0.2 and np.linalg.norm(d) < 3.0
    for _ in range(400):
        cam.wheel(-3)
    assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH
    for _ in range(400):
        cam.fly((0.0, 0.0, -1.0), 0.1, True)
    assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH


@pytest.mark.slow
@pytest.mark.parametrize("dive", [0, 30, 150])
def test_fuzz_never_in_solid(dive):
    """Random inputs over every 3D control, from the overview and after diving in (3,400 each,
    10,200 in all)."""
    rng = random.Random(dive + 5)
    nrng = np.random.default_rng(dive)
    cam = CameraND(Rule(3, 2), 1280, 720)
    for _ in range(dive):
        cam.wheel(1, (rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), 1.0))
    rows = 0
    for _ in range(3400):
        r = rng.random()
        d = (rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3), 1.0)
        if r < 0.30:
            cam.wheel(rng.choice((1, 1, 2, -1) if dive else (1, -1)), d)
        elif r < 0.50:
            cam.drag_rotate(rng.uniform(-60, 60), rng.uniform(-60, 60))
        elif r < 0.70:
            cam.fly(nrng.normal(0, 1, 3), rng.uniform(0.005, 0.05), rng.random() < 0.3)
        elif r < 0.82:
            cam.pan(rng.uniform(-80, 80), rng.uniform(-80, 80), cam.sigma)
        elif r < 0.92:
            cam.roll(rng.uniform(-0.5, 0.5))
        else:
            cam.update_sigma(0.05)
        assert cam.de > 0 and orthonormal(cam.F)
        assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH + 1e-9
        rows = max(rows, len(cam.table.w) + cam.table.dropped)
    assert rows <= 24


def test_rule_change_inside_solid_resets():
    cam = CameraND(Rule(3, 1), 640, 360)          # Cantor dust: mostly empty, easy to fly into
    for _ in range(40):
        cam.wheel(1, (0.05, 0.03, 1.0))
    for m in (2, 3, 1, 3):
        cam.set_rule(Rule(3, m))
        assert cam.de > 0
    for lv in (1, 3, None):
        cam.set_rule(Rule(3, 2, lv))
        assert cam.de > 0 and cam.rule == Rule(3, 2, lv)


def test_sigma_settles_and_stops_changing():
    cam = CameraND(Rule(3, 2), 640, 360)
    for _ in range(30):
        cam.wheel(1)
    changed = [cam.update_sigma(1 / 60) for _ in range(400)]
    assert any(changed) and not any(changed[-20:])
    k = cam.key()
    cam.update_sigma(1 / 60)
    assert cam.key() == k


def test_fov_and_size_change_pix():
    cam = CameraND(Rule(3, 2), 640, 360)
    p = cam.pix
    cam.set_fov(90)
    assert cam.pix == pytest.approx(2.0 / 360) and cam.pix > p
    cam.resize(1280, 720)
    assert cam.pix == pytest.approx(1.0 / 360)
    v = cam.far
    cam.set_view_range(80)
    assert cam.far == pytest.approx(2 * v)


# ---- scalar estimator and free distance (check_camera.py) --------------------------------------
def test_scalar_estimator_equals_the_vector_one():
    cam = CameraND(Rule(3, 2), 1080, 1080)
    for _ in range(60):
        cam.wheel(1, (0.11, 0.07, 1.0))
    tb = cam.table
    rows = table_rows(tb)
    rng = np.random.default_rng(5)
    q = rng.normal(0, 1, (2000, 4)) * (10.0 ** rng.uniform(-3, 2.5, 2000))[:, None]
    q[:, 3] = 0.0
    dv = de_table(q, tb, dtype=np.float64)
    ds = np.array([de_scalar(list(r), rows) for r in q])
    assert np.abs(dv - ds).max() < 1e-9


def test_free_distance_stops_before_the_surface():
    cam = CameraND(Rule(3, 2), 1080, 1080)
    for _ in range(40):
        cam.wheel(1, (0.11, 0.07, 1.0))
    rng = np.random.default_rng(7)
    for _ in range(100):
        d = rng.normal(0, 1, 3)
        D = cam.F[:, :3] @ (d / np.linalg.norm(d))
        t, hit = free_distance(cam.rows, list(D), cam.pix, cam.far)
        assert t > 0
        if hit:                                     # half of it is always empty
            assert de_table((D * 0.5 * t)[None, :], cam.table, dtype=np.float64)[0] > 0
