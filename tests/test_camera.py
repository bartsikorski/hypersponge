"""The 3D camera (design 15.2, 15.3), from the 3D parts of prototype/check_camera_rules.py and
prototype/check_camera.py.  Every rule keeps the eye in empty space."""
import math
import random

import numpy as np
import pytest

from hypersponge.core.camera import CameraND
from hypersponge.core.frame import LEASH, START
from hypersponge.core.levels import level_table, normalize
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import de_scalar, de_table, free_distance, table_rows


def cam3(m=2, max_level=None, H=720):
    return CameraND(Rule(3, m, max_level), H * 16 // 9, H)


def screen_xy(cam, P):
    v = cam.F[:, :3].T @ (P - cam.eye_world())
    return v[0] / v[2], v[1] / v[2]


def test_start_and_reset():
    cam = cam3()
    assert cam.overview() and cam.depth == 0 and abs(cam.rho() - np.linalg.norm(START)) < 1e-12
    assert np.allclose(cam.eye_world()[:3], 0.5 + START) and cam.eye_world()[3] == 0.0
    to_c = cam.c - cam.eye_world()
    assert np.allclose(cam.F[:, 2], to_c / np.linalg.norm(to_c), atol=1e-12)
    k = cam.key()
    for _ in range(20):
        cam.wheel(1)
    cam.drag_rotate(30, -10)
    assert cam.key() != k
    cam.reset()
    assert cam.key() == k


def test_overview_rotations_keep_rho_and_the_frame():
    rnd = random.Random(11)
    cam = cam3()
    rho0 = cam.rho()
    for _ in range(3000):
        i, j = rnd.sample(range(3), 2)
        cam.rotate(i, j, rnd.uniform(-0.2, 0.2))
    assert abs(cam.rho() - rho0) < 1e-9 and cam.overview()
    assert np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-12)
    assert np.allclose(cam.F[:, 3], [0, 0, 0, 1])                       # 3D: ana stays w


def test_drag_follows_the_mouse_in_both_regimes():
    cam = cam3()
    near = cam.eye_world() + cam.F[:, 2] * (cam.rho() - 0.6)           # on the near side of the cube
    x0, y0 = screen_xy(cam, near)
    cam.drag_rotate(40, 0)
    x1, _ = screen_xy(cam, near)
    cam.drag_rotate(0, 40)
    _, y1 = screen_xy(cam, near)
    assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and cam.overview()
    for _ in range(14):
        cam.wheel(1)
    assert not cam.overview()
    ahead = cam.eye_world() + cam.F[:, 2] * 0.01
    x0, y0 = screen_xy(cam, ahead)
    e0 = cam.eye_world().copy()
    cam.drag_rotate(40, 0)
    x1, _ = screen_xy(cam, ahead)
    cam.drag_rotate(0, 40)
    _, y1 = screen_xy(cam, ahead)
    assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and np.allclose(cam.eye_world(), e0)


def test_wheel_in_and_out_returns():
    cam = cam3()
    e0 = cam.eye_world().copy()
    cam.wheel(1)
    cam.wheel(-1)
    assert np.linalg.norm(cam.eye_world() - e0) < 0.01 * np.linalg.norm(e0 - cam.c)


def test_the_wheel_alone_reaches_depth_100_without_entering_solid():
    cam = cam3(H=1080)
    notches = 0
    while cam.depth < 100:
        cam.wheel(1, cam.cursor_dir((0.0, 0.0)))
        notches += 1
        assert cam.de > 0 and notches < 800
    assert not cam.overview()
    # the eye is in empty space: the exact distance estimate at the eye is positive
    tb = level_table(cam.pos, 3, 2, far=0.0)
    assert tb.de_cam > 0


def test_wheel_on_a_miss_stops_beside_the_object_and_the_leash_holds():
    cam = cam3()
    d_cam = np.array([0.45, 0.30, 1.0])
    for _ in range(200):
        cam.wheel(1, d_cam)                                            # a corner of the picture: a miss
    d = cam.eye_world() - cam.c
    D = cam.F[:, :3] @ (d_cam / np.linalg.norm(d_cam))
    assert abs(d @ D) < 0.2 and np.linalg.norm(d) < 3.0
    for _ in range(400):
        cam.wheel(-3)
    assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH
    for _ in range(400):
        cam.fly((0.0, 0.0, -1.0), 0.1, True)
    assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH


@pytest.mark.parametrize("dive", [0, 30, 150])
def test_fuzz_every_control_never_enters_solid(dive):
    rnd = random.Random(dive + 1)
    rng = np.random.default_rng(dive + 2)
    cam = cam3()
    for _ in range(dive):
        cam.wheel(1, (rnd.uniform(-0.3, 0.3), rnd.uniform(-0.2, 0.2), 1.0))
    for _ in range(1200):
        r = rnd.random()
        d = (rnd.uniform(-0.45, 0.45), rnd.uniform(-0.3, 0.3), 1.0)
        if r < 0.30:
            cam.wheel(rnd.choice((1, 1, 2, -1) if dive else (1, -1)), d)
        elif r < 0.50:
            cam.drag_rotate(rnd.uniform(-60, 60), rnd.uniform(-60, 60))
        elif r < 0.70:
            cam.fly(rng.normal(0, 1, 3), rnd.uniform(0.005, 0.05), rnd.random() < 0.3)
        elif r < 0.82:
            cam.pan(rnd.uniform(-80, 80), rnd.uniform(-80, 80), cam.grab(d[:2]))
        elif r < 0.90:
            cam.roll(rnd.uniform(-0.5, 0.5))
        else:
            cam.update_sigma(0.05)
        assert cam.de > 0 and np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-9)
        assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH + 1e-9
        assert cam.table.dropped == 0


def test_drunk_fly_in_any_direction_never_ends_in_solid():
    """Raw rule of the core: random moves of at most half the free distance."""
    rnd = random.Random(3)
    rng = np.random.default_rng(5)
    cam = cam3(H=1080)
    for _ in range(60):
        cam.wheel(1)
    p = cam.pos.copy()
    pix = cam.pix
    for _ in range(800):
        de = normalize(p, 3, 2)
        assert de > 0
        tb = level_table(p, 3, 2, far=1000.0, fine_min=0.3 * de * pix)
        rows = table_rows(tb)
        u = np.zeros(4)
        u[:3] = rng.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        fd, _ = free_distance(rows, list(u), pix, 1000.0)
        p.move(u * min(rnd.choice((0.3, 1.0, 3.0, 30.0)) * de, 0.5 * fd))


def test_scalar_estimate_equals_vector_estimate():
    cam = cam3()
    for _ in range(60):
        cam.wheel(1)
    rng = np.random.default_rng(7)
    q = np.zeros((500, 4))
    q[:, :3] = rng.normal(0, 1, (500, 3)) * (10.0 ** rng.uniform(-3, 2.5, 500))[:, None]
    dv = de_table(q, cam.table, dtype=np.float64)
    ds = np.array([de_scalar(list(r), cam.rows) for r in q])
    assert np.abs(dv - ds).max() < 1e-9 * np.abs(dv).max()


def test_rule_change_inside_solid_resets():
    cam = cam3()
    for _ in range(40):
        cam.wheel(1, (0.1, 0.05, 1.0))
    assert not cam.overview()
    cam.set_rule(Rule(3, 3))              # fewer holes: the eye may now be inside solid
    assert cam.de > 0
    cam.set_rule(Rule(3, 2, 1))           # level 1 only: deep inside, the eye is in solid
    assert cam.de > 0 and cam.rule == Rule(3, 2, 1)
    with pytest.raises(ValueError):
        cam.set_rule(Rule(2, 2))


def test_fov_and_view_distance():
    cam = cam3()
    pix0, far0 = cam.pix, cam.far
    cam.set_fov(70.0)
    assert cam.pix == pytest.approx(2 * math.tan(math.radians(35.0)) / cam.H) and cam.pix > pix0
    cam.set_view_range(80.0)
    assert cam.far == pytest.approx(2 * far0)
    cam.resize(640, 360)
    assert cam.pix == pytest.approx(2 * math.tan(math.radians(35.0)) / 360)


def test_update_sigma_settles():
    cam = cam3()
    for _ in range(30):
        cam.wheel(1)
    moving = [cam.update_sigma(1 / 60) for _ in range(600)]
    assert not moving[-1]
