"""The 2D camera (design 15.1), from prototype/check_camera2d.py."""
import random
from fractions import Fraction as F

import numpy as np
import pytest

from hypersponge.core.camera import MAX_HEIGHT, Camera2D, safe_fraction
from hypersponge.core.position import Pos
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import removal_row

H = 1080


def under_cursor(cam, cur):
    X = cam.pos.exact()
    s = F(3) ** cam.pos.E
    return [X[i] + F(cur[i] * cam.pixel) / s for i in range(2)]


def new_cam(m=2):
    return Camera2D(Rule(2, m), 1920, H)


def test_zoom_keeps_the_point_under_the_cursor():
    rnd = random.Random(5)
    cam = new_cam()
    cam.pos = Pos.from_floats([0.25, 0.25, 0, 0])
    cam.refresh()
    worst = 0.0
    rows = 0
    for _ in range(2600):
        cur = (rnd.uniform(-900, 900), rnd.uniform(-500, 500))
        before = under_cursor(cam, cur)
        deep = cam.pos.E >= 3
        cam.zoom(rnd.choice([1, 1, 1, 2, -1]), cur)
        after = under_cursor(cam, cur)
        hw = F(cam.height) / F(3) ** cam.pos.E
        if deep and cam.pos.E >= 3:                     # near the top the centre clamp may act
            worst = max(worst, max(abs(float((a - b) / hw)) for a, b in zip(after, before)))
        rows = max(rows, len(cam.table.w))
        assert 1.0 <= cam.height < 3.0 or cam.pos.E == 0
        assert cam.table.dropped == 0
    assert worst < 0.5 / H and cam.pos.E > 300 and rows <= 12      # under half a pixel


def test_zoom_out_returns_and_stops_and_pan_follows():
    cam = new_cam()
    c0 = cam.pos.exact()
    h0 = cam.height
    for _ in range(500):
        cam.zoom(1, (300, -200))
    for _ in range(500):
        cam.zoom(-1, (300, -200))
    c1 = cam.pos.exact()
    assert cam.pos.E == 0 and abs(cam.height - h0) < 1e-9
    assert all(abs(float(a - b)) < 2 * h0 / H for a, b in zip(c0, c1))          # within 2 pixels
    for _ in range(50):
        cam.zoom(-1)
    assert abs(cam.world_height() - MAX_HEIGHT) < 1e-12
    rnd = random.Random(6)
    for _ in range(400):
        cam.pan(rnd.uniform(-2000, 2000), rnd.uniform(-2000, 2000))
    assert all(0 <= x <= 1 for x in cam.pos.exact()[:2])
    cam = new_cam()
    for _ in range(300):
        cam.zoom(1, (500, 300))
    x0 = under_cursor(cam, (100, 50))
    cam.pan(37, -21)
    x1 = under_cursor(cam, (137, 29))
    hw = F(cam.height) / F(3) ** cam.pos.E
    assert all(abs(float((a - b) / hw)) < 1e-12 for a, b in zip(x0, x1))


@pytest.mark.parametrize("m", [2, 1])
def test_anchor_rule_never_ends_in_a_hole(m):
    rng = np.random.default_rng(3)
    cam = new_cam(m)
    events = 0
    while cam.pos.E < 400:
        for half in (40, 200, 400):                     # a pixel near the centre that shows material
            ys, xs = np.mgrid[-half:half, -half * 3 // 2:half * 3 // 2]
            q = np.zeros((xs.size, 4))
            q[:, 0] = (xs.ravel() + 0.5) * cam.pixel
            q[:, 1] = (ys.ravel() + 0.5) * cam.pixel
            kept = np.nonzero(removal_row(q, cam.table, cam.pixel / 3, np.float64) == -1)[0]
            if kept.size:
                break
        assert kept.size > 0, "no material in view"
        pick = int(kept[rng.integers(kept.size)])
        cur = (float(xs.ravel()[pick]) + 0.5, float(ys.ravel()[pick]) + 0.5)
        for _ in range(int(rng.integers(1, 12))):
            cam.zoom(1, cur)
            events += 1
            q = np.array([[cur[0] * cam.pixel, cur[1] * cam.pixel, 0, 0]])
            assert cam.table.covered == 0, (m, events, cam.pos.E)
            assert removal_row(q, cam.table, cam.pixel / 3, np.float64)[0] == -1, (m, events, cam.pos.E)


def test_safe_fraction_has_no_digit_one_and_moves_little():
    rnd = random.Random(9)
    for _ in range(2000):
        u = rnd.random()
        s = safe_fraction(u)
        x = F(s)
        for k in range(1, 30):
            assert int(x * 3 ** k) % 3 != 1, (u, s, k)
        assert abs(s - u) <= 1.0 / 3.0                   # inside the same first-level third or next to it


def test_rule_change_keeps_the_view_and_needs_a_2d_rule():
    cam = new_cam()
    for _ in range(60):
        cam.zoom(1, (120.5, 80.5))
    pos, h = cam.pos.copy(), cam.height
    cam.set_rule(Rule(2, 1, 5))
    assert cam.pos.E == pos.E and cam.height == h and cam.cut == 2.0
    with pytest.raises(ValueError):
        cam.set_rule(Rule(3, 2))


def test_key_changes_with_every_view_change():
    cam = new_cam()
    k0 = cam.key()
    cam.pan(1, 0)
    k1 = cam.key()
    cam.resize(800, 600)
    k2 = cam.key()
    cam.set_rule(Rule(2, 2, 4))
    k3 = cam.key()
    assert len({k0, k1, k2, k3}) == 4
    cam.resize(800, 600)
    assert cam.key() == k3
