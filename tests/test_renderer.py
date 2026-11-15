"""Renderer behaviour (design 16): accumulation, dynamic resolution and the long-frame guard."""
import numpy as np
import pytest

from hypersponge.core.camera import Camera2D, CameraND
from hypersponge.core.rule import Rule
from hypersponge.render import renderer as rmod
from hypersponge.render.renderer import Renderer, jitter, render_still
from hypersponge.render.styles import StyleSettings

pytestmark = pytest.mark.gpu


def test_jitter_is_inside_the_pixel():
    pts = [jitter(i) for i in range(1, 17)]
    assert all(-0.5 <= x < 0.5 and -0.5 <= y < 0.5 for x, y in pts) and len(set(pts)) == 16


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_band_mode_gives_the_same_picture(ctx, monkeypatch, mode):
    """With the guard forced on, the still picture built band by band equals the plain one."""
    cam = Camera2D(Rule(2, 2), 200, 150) if mode == "2d" else CameraND(Rule(3, 2), 200, 150)
    st = StyleSettings()
    plain = render_still(cam, st, samples=4, renderer=Renderer(ctx, dynamic=False))
    monkeypatch.setattr(rmod, "LONG_MOVING_MS", -1.0)       # every frame counts as long ...
    monkeypatch.setattr(rmod, "LONG_STILL_MS", -1.0)
    monkeypatch.setattr(rmod, "PASS_FAST_MS", -1.0)         # ... and the bands never merge again
    monkeypatch.setattr(rmod, "BAND_SLOW_MS", 1e9)
    r = Renderer(ctx, max_samples=4, dynamic=True)
    r.scale = rmod.SCALE_MIN
    r.frame(cam, st)                                         # a moving frame switches the guard on
    assert r.bands == rmod.BANDS_START
    frames = 0
    while r.frame(cam, st):
        frames += 1
        assert frames < 1000
    assert r.samples == 4 and frames == 4 * r.bands
    banded = r.picture()
    assert np.abs(banded.astype(int) - plain.astype(int)).mean() < 0.5


def test_moving_frames_roll_through_the_bands(ctx, monkeypatch):
    monkeypatch.setattr(rmod, "PASS_FAST_MS", -1.0)
    monkeypatch.setattr(rmod, "BAND_SLOW_MS", 1e9)
    cam = Camera2D(Rule(2, 2), 160, 120)
    r = Renderer(ctx, dynamic=True)
    r._start_bands()
    seen = []
    for i in range(10):
        cam.pan(1, 0)
        r.frame(cam, StyleSettings())
        seen.append(r.next_band)
    assert seen[:8] == list(range(1, 9)) and seen[8] == 1


def test_bands_double_when_slow_and_halve_when_fast(ctx):
    r = Renderer(ctx, dynamic=True)
    r._start_bands()
    r._adapt_bands(150.0, 1200.0)
    assert r.bands == 16
    r._adapt_bands(1.0, 16.0)
    assert r.bands == 8
    for _ in range(5):
        r._adapt_bands(1.0, 8.0)
    assert r.bands == 1


def test_dynamic_resolution_lowers_and_restores_the_scale(ctx):
    r = Renderer(ctx, dynamic=True)
    for _ in range(40):
        r._adapt(30.0)
    assert r.scale == pytest.approx(rmod.SCALE_MIN)
    for _ in range(80):
        r._adapt(2.0)
    assert r.scale == 1.0
