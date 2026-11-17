"""The 4D camera (design 5.4, 15.3), from the 4D parts of prototype/check_camera_rules.py, plus
the automatic motions, which the prototype did not have."""
import math
import random

import numpy as np
import pytest

from hypersponge.core.camera import CameraND
from hypersponge.core.frame import LEASH, PRESETS, START, START_PRESET, slice_basis
from hypersponge.core.levels import level_table
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import march


def cam4(m=2, H=720, preset=None):
    cam = CameraND(Rule(4, m), H * 16 // 9, H)
    if preset:
        cam.preset(preset)
    return cam


def cam3(m=2, H=720):
    return CameraND(Rule(3, m), H * 16 // 9, H)


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets(name):
    ana, h = PRESETS[name]
    cam = cam4(preset=name)
    assert np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-12) and abs(np.linalg.det(cam.F) - 1) < 1e-9
    a = np.asarray(ana, float)
    assert np.allclose(cam.F[:, 3], a / np.linalg.norm(a), atol=1e-12)
    assert cam.overview() and cam.depth == 0
    assert abs(cam.h() - h) < 1e-12 and abs(cam.rho() - np.linalg.norm(START)) < 1e-12
    to_target = -(cam.F[:, :3] @ cam.rel()[:3])                  # looking at the slice's centre point
    assert np.allclose(to_target / np.linalg.norm(to_target), cam.F[:, 2], atol=1e-12)
    # Reset returns to the last preset chosen
    k = cam.key()
    for _ in range(10):
        cam.wheel(1)
    cam.drag_turn(30, 20)
    assert cam.key() != k
    cam.reset()
    assert cam.key() == k and cam.preset_name == name


def test_start_preset_and_3d_refuses_4d_operations():
    cam = cam4()
    assert cam.preset_name == START_PRESET == "diagonal"
    assert abs(cam.offset_range() - 1.0) < 1e-12                  # diagonal: 0.5 * 4 * 0.5
    c3 = cam3()
    k = c3.key()
    c3.drag_turn(40, 40)
    assert c3.slide(3) == 0.0 and not c3.step_auto(0.1)
    c3.set_offset(0.3)
    assert c3.key() == k
    with pytest.raises(ValueError):
        c3.preset("menger")


def test_overview_rotations_keep_rho_and_h():
    rnd = random.Random(11)
    cam = cam4()
    rho0, h0 = cam.rho(), cam.h()
    for _ in range(3000):
        i, j = rnd.sample(range(4), 2)
        cam.rotate(i, j, rnd.uniform(-0.2, 0.2))
    assert abs(cam.rho() - rho0) < 1e-9 and abs(cam.h() - h0) < 1e-9 and cam.overview()
    for _ in range(2000):                                           # plane jogs and rolls
        i, j = rnd.sample(range(4), 2)
        cam.rotate_object(i, j, rnd.uniform(-0.2, 0.2))
        cam.roll(rnd.uniform(-0.2, 0.2))
    assert abs(cam.rho() - rho0) < 1e-9 and abs(cam.h() - h0) < 1e-9 and cam.overview()
    assert np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-12)


def test_drag_turn_turns_ana_and_keeps_the_eye_in_camera_coordinates():
    cam = cam4()
    a0, rel0 = cam.ana(), cam.rel()
    cam.drag_turn(100, 0)                                          # 30 degrees in (right, ana)
    assert math.degrees(math.acos(np.clip(a0 @ cam.ana(), -1, 1))) == pytest.approx(30.0, abs=1e-6)
    assert np.allclose(cam.rel(), rel0, atol=1e-12)
    for _ in range(40):
        cam.wheel(1)
    assert not cam.overview()
    e0 = cam.eye_world().copy()
    cam.drag_turn(-50, 70)                                         # immersed: the eye stays put
    assert np.array_equal(cam.eye_world(), e0)


def test_offset_slider_and_slide_keep_the_slice_on_the_cube():
    cam = cam4()
    H = cam.offset_range()
    cam.set_offset(0.9 * H)
    assert abs(cam.h() - 0.9 * H) < 1e-12 and cam.overview()
    cam.set_offset(5.0)
    assert abs(cam.h() - H) < 1e-12
    cam.set_offset(-H)
    assert abs(cam.h() + H) < 1e-12
    cam.set_offset(0.95)
    cam.rotate(2, 3, 0.6)                                           # turning shrinks H: h follows
    assert abs(cam.h()) <= cam.offset_range() + 1e-12
    for _ in range(100):
        cam.slide(2)
    assert abs(cam.h() - cam.offset_range()) < 1e-9 and cam.overview()
    assert cam.slide(1) < 1e-12                                     # at the end: nothing left but rounding
    s = cam.slide(-1)
    assert s > 0 and abs(cam.h()) < cam.offset_range()


def test_slide_one_notch_is_8_percent_of_sigma_and_offset_does_nothing_immersed():
    cam = cam4(preset="generic")
    for _ in range(40):
        cam.wheel(1)
    assert not cam.overview()
    rel0, k = cam.rel(), cam.key()
    cam.set_offset(rel0[3] + 0.1)
    assert cam.key() == k
    sigma, E = cam.sigma, cam.depth
    t, _ = cam.free(cam.F[:, 3])
    step = cam.slide(1)
    assert step > 0 and step == pytest.approx(min(0.08 * sigma, 0.5 * t), rel=1e-9)
    assert np.allclose(cam.rel()[:3], rel0[:3], rtol=0, atol=1e-12)          # only along ana
    assert cam.h() - rel0[3] == pytest.approx(step * 3.0 ** -E, rel=1e-6)


def test_menger_preset_equals_3d_and_dust_preset_equals_3d_cantor_dust():
    """w = 1/4 has no digit 1 in base 3, w = 1/2 has only 1s: the slices are exactly the Menger
    sponge and 3D Cantor dust.  Every ray must agree."""
    for preset, m3 in (("menger", 2), ("dust", 1)):
        c4 = cam4(H=96, preset=preset)
        c3 = cam3(m=m3, H=96)
        assert np.allclose(c4.F, c3.F)
        ys, xs = np.mgrid[0:96, 0:128]
        dc = np.stack([(xs + 0.5 - 64) * c3.pix, (ys + 0.5 - 48) * c3.pix, np.ones((96, 128))], -1).reshape(-1, 3)
        dc /= np.linalg.norm(dc, axis=1)[:, None]
        t4, h4, _ = march(c4.table, dc @ c4.F[:, :3].T, c4.pix, c4.far)
        t3, h3, _ = march(c3.table, dc @ c3.F[:, :3].T, c3.pix, c3.far)
        assert h4.mean() > 0.05 and (h4 == h3).all(), preset
        assert np.allclose(t4[h4], t3[h3], rtol=1e-5), preset


def test_an_oblique_slice_zooms_to_depth_100_without_entering_solid():
    cam = cam4(H=1080, preset="generic")
    notches = 0
    while cam.depth < 100:
        cam.wheel(1, cam.cursor_dir((0.0, 0.0)))
        notches += 1
        assert cam.de > 0 and notches < 1000
    assert not cam.overview()
    assert level_table(cam.pos, 4, 2, far=0.0).de_cam > 0


def test_w_ref_spans_the_cube_in_overview_and_follows_sigma_immersed():
    cam = cam4(preset="generic")
    assert cam.rho() > 1.05 * cam.R                                  # b = 1
    ref, scale = cam.w_ref()
    assert ref == pytest.approx(cam.c[3] - cam.eye_world()[3]) and scale == pytest.approx(1.0)
    for _ in range(80):
        cam.wheel(1)
    assert cam.rho() < 0.8 * cam.R
    assert cam.w_ref() == (0.0, cam.sigma)


@pytest.mark.parametrize("dive", [0, 30, 150])
def test_fuzz_every_4d_control_never_enters_solid(dive):
    rnd = random.Random(dive + 7)
    rng = np.random.default_rng(dive + 8)
    cam = cam4(preset="generic" if dive else None)
    for _ in range(dive):
        cam.wheel(1, (rnd.uniform(-0.3, 0.3), rnd.uniform(-0.2, 0.2), 1.0))
    for _ in range(1000):
        r = rnd.random()
        d = (rnd.uniform(-0.45, 0.45), rnd.uniform(-0.3, 0.3), 1.0)
        if r < 0.22:
            cam.wheel(rnd.choice((1, 1, 2, -1) if dive else (1, -1)), d)
        elif r < 0.32:
            cam.drag_rotate(rnd.uniform(-60, 60), rnd.uniform(-60, 60))
        elif r < 0.42:
            cam.fly(rng.normal(0, 1, 3), rnd.uniform(0.005, 0.05), rnd.random() < 0.3)
        elif r < 0.48:
            cam.pan(rnd.uniform(-80, 80), rnd.uniform(-80, 80), cam.grab(d[:2]))
        elif r < 0.60:
            cam.drag_turn(rnd.uniform(-60, 60), rnd.uniform(-60, 60))
        elif r < 0.72:
            cam.slide(rnd.choice((-2, -1, 1, 2)))
        elif r < 0.80:
            cam.rotate_object(*rnd.sample(range(4), 2), rnd.uniform(-0.3, 0.3))
        elif r < 0.88:
            cam.auto_slide, cam.auto_turn = rnd.random() < 0.7, rnd.random() < 0.5
            cam.auto_plane = rnd.randrange(3)
            for _ in range(5):
                cam.step_auto(1 / 60)
        elif r < 0.92:
            cam.set_offset(rnd.uniform(-1.2, 1.2))
        elif r < 0.93 and not dive:
            cam.preset(rnd.choice(list(PRESETS)))
        elif r < 0.96:
            cam.roll(rnd.uniform(-0.5, 0.5))
        else:
            cam.update_sigma(0.05)
        assert cam.de > 0 and np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-9)
        assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH + 1e-9
        assert not cam.overview() or abs(cam.h()) <= cam.offset_range() + 1e-9
        assert cam.table.dropped == 0


def test_auto_slide_in_overview_is_a_smooth_triangle_wave():
    cam = cam4()
    cam.auto_slide = True
    edge = 0.98 * cam.offset_range()
    hs = []
    for _ in range(60 * 20):
        assert cam.step_auto(1 / 60)
        hs.append(cam.h())
    hs = np.array(hs)
    step = 2 * edge / 8.0 / 60                                       # 8 seconds each way
    assert hs.max() == pytest.approx(edge, abs=step) and hs.min() == pytest.approx(-edge, abs=step)
    assert np.abs(np.diff(hs)).max() <= step * (1 + 1e-9)
    turns = np.sum(np.diff(np.sign(np.diff(hs))) != 0)
    assert turns == 2                                                # up, down, up in 20 seconds
    # from the very end of the range it starts back inwards without a jump
    cam.set_offset(cam.offset_range())
    h0 = cam.h()
    cam.step_auto(1 / 60)
    assert h0 - step * (1 + 1e-9) <= cam.h() < h0


def test_auto_slide_immersed_moves_along_ana_and_turns_back_when_blocked():
    cam = cam4(preset="generic")
    for _ in range(60):
        cam.wheel(1)
    assert not cam.overview()
    cam.auto_slide = True
    flips, last = 0, cam._slide_dir
    for _ in range(600):
        rel0 = cam.rel()
        cam.step_auto(1 / 60)
        cam.update_sigma(1 / 60)
        assert cam.de > 0 and not cam.overview()
        assert abs(cam.rho() - np.linalg.norm(rel0[:3])) < 1e-9       # only ana changes
        flips += cam._slide_dir != last
        last = cam._slide_dir
    assert flips >= 1


def test_auto_turn_turns_ana_at_10_degrees_per_second():
    cam = cam4()
    cam.auto_turn = True
    cam.auto_plane = 2
    a0, rho0 = cam.ana(), cam.rho()
    for _ in range(60):
        cam.step_auto(1 / 60)
    assert math.degrees(math.acos(np.clip(a0 @ cam.ana(), -1, 1))) == pytest.approx(10.0, abs=1e-6)
    assert abs(cam.rho() - rho0) < 1e-9 and cam.overview()
    assert not cam4().step_auto(1 / 60)                               # nothing on: nothing moves


def test_rule_change_in_4d_keeps_or_resets_the_preset():
    cam = cam4(preset="cell diagonal")
    for _ in range(40):
        cam.wheel(1, (0.1, 0.05, 1.0))
    cam.set_rule(Rule(4, 4))
    assert cam.de > 0
    cam.set_rule(Rule(4, 2, 1))
    assert cam.de > 0 and cam.preset_name == "cell diagonal"
    if cam.overview():                                               # it reset to the preset
        assert np.allclose(cam.ana(), slice_basis((0, 1, 1, 1))[:, 3])
