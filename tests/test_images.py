"""Golden scenes 1 to 14 and the identities (design 18.2).  Needs an OpenGL 3.3 context.

`pytest --update-golden` rewrites tests/golden/*.png; look at the new pictures before committing.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hypersponge.core.camera import Camera2D, CameraND
from hypersponge.core.position import Pos
from hypersponge.core.rule import Rule
from hypersponge.render.renderer import render_still
from hypersponge.render.styles import StyleSettings, styles_for

pytestmark = pytest.mark.gpu
GOLDEN = Path(__file__).resolve().parent / "golden"
SIZE = 256


def mad(a, b) -> float:
    """Mean absolute difference in units of 1/255."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def cam_at(xy, height, E=0, m=2, size=SIZE):
    cam = Camera2D(Rule(2, m), size, size)
    p = Pos.from_floats([xy[0], xy[1], 0, 0])
    p.rescale(E)
    cam.pos = p
    cam.height = height
    cam.refresh()
    return cam


def cam3(m=2, max_level=None, notches=0, size=SIZE):
    cam = CameraND(Rule(3, m, max_level), size, size)
    for _ in range(notches):
        cam.wheel(1, (0.12, -0.05, 1.0))
    cam.update_sigma()
    return cam


def cam4(preset, m=2, notches=0, size=SIZE):
    cam = CameraND(Rule(4, m), size, size)
    cam.preset(preset)
    for _ in range(notches):
        cam.wheel(1, (0.12, -0.05, 1.0))
    cam.update_sigma()
    return cam


SCENES = {
    "01_carpet": lambda: (cam_at((0.5, 0.5), 1.25), StyleSettings()),
    "02_carpet_deep_levels": lambda: (cam_at((0.25, 0.75), 1.4, E=150), StyleSettings(style="levels", palette="pastel")),
    "03_dust_clay": lambda: (cam_at((0.5, 0.5), 1.25, m=1), StyleSettings(style="clay")),
    "04_carpet_inverted_light": lambda: (cam_at((0.5, 0.5), 1.25),
                                         StyleSettings(inverted=True, background="light")),
    "05_sponge": lambda: (cam3(), StyleSettings()),
    "06_sponge_wheel40": lambda: (cam3(notches=40), StyleSettings()),
    "07_centre_removed_clay": lambda: (cam3(m=3), StyleSettings(style="clay")),
    "08_max_level2_axes": lambda: (cam3(max_level=2), StyleSettings(style="axes")),
    "09_diagonal": lambda: (cam4("diagonal"), StyleSettings()),
    "10_dust": lambda: (cam4("dust"), StyleSettings()),
    "11_face_diagonal_ember": lambda: (cam4("face diagonal"), StyleSettings(style="levels", palette="ember")),
    "12_cell_diagonal_axes": lambda: (cam4("cell diagonal"), StyleSettings(style="axes")),
    "13_generic_wdepth": lambda: (cam4("generic"), StyleSettings(style="wdepth")),
    "14_generic_wheel60_levels": lambda: (cam4("generic", notches=60), StyleSettings(style="levels")),
}


@pytest.mark.parametrize("name", list(SCENES))
def test_golden(renderer, update_golden, name):
    cam, st = SCENES[name]()
    img = render_still(cam, st, samples=16, renderer=renderer)
    path = GOLDEN / f"{name}.png"
    if update_golden or not path.exists():
        GOLDEN.mkdir(exist_ok=True)
        Image.fromarray(img).save(path)
        if not update_golden:
            pytest.skip(f"wrote the missing golden picture {path.name}")
    want = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape == want.shape
    assert mad(img, want) < 2.0, mad(img, want)
    assert img.std() > 10                                          # not one colour


def test_same_3d_picture_at_unit_E_and_E_plus_1(renderer):
    """One rescale(+1) without normalising leaves the 3D picture unchanged."""
    from hypersponge.core.levels import level_table
    from hypersponge.core.sdf import table_rows
    a = cam3(notches=40)
    b = cam3(notches=40)
    b.pos.rescale(+1)
    b.far *= 3.0
    b.de *= 3.0
    b.table = level_table(b.pos, 3, 2, far=b.far, fine_min=0.3 * b.de * b.pix)
    b.rows = table_rows(b.table)
    assert b.pos.E == a.pos.E + 1 and b.sigma == pytest.approx(3.0 * a.sigma)
    for style in ("sponge", "levels", "axes"):
        st = StyleSettings(style=style)
        ia = render_still(a, st, samples=4, renderer=renderer)
        ib = render_still(b, st, samples=4, renderer=renderer)
        assert mad(ia, ib) < 1.0, (style, mad(ia, ib))


@pytest.mark.parametrize("preset,m3", [("menger", 2), ("dust", 1)])
def test_4d_presets_equal_3d(renderer, preset, m3):
    """The slice w = 1/4 is the Menger sponge and w = 1/2 is 3D Cantor dust, at every depth."""
    for notches in (0, 40):
        a = cam4(preset, notches=notches)
        b = cam3(m=m3, notches=notches)
        for style in ("sponge", "axes", "clay"):
            st = StyleSettings(style=style)
            ia = render_still(a, st, samples=4, renderer=renderer)
            ib = render_still(b, st, samples=4, renderer=renderer)
            assert mad(ia, ib) < 1.0, (preset, notches, style, mad(ia, ib))


def test_presets_differ(renderer):
    pics = [render_still(cam4(p, size=96), StyleSettings(), samples=1, renderer=renderer)
            for p in ("menger", "dust", "diagonal", "face diagonal", "cell diagonal", "generic")]
    for i in range(len(pics)):
        for j in range(i):
            assert mad(pics[i], pics[j]) > 2.0, (i, j)


def test_same_picture_at_unit_E_and_E_plus_1(renderer):
    """One rescale(+1) without normalising leaves the picture unchanged."""
    a = cam_at((0.25, 0.75), 1.4, E=40)
    b = cam_at((0.25, 0.75), 1.4, E=40)
    b.pos.rescale(+1)
    b.height *= 3.0
    b.build_table()
    assert b.pos.E == a.pos.E + 1
    for style in styles_for("2d"):
        st = StyleSettings(style=style)
        ia = render_still(a, st, samples=4, renderer=renderer)
        ib = render_still(b, st, samples=4, renderer=renderer)
        assert mad(ia, ib) < 1.0, (style, mad(ia, ib))


@pytest.mark.parametrize("style", ["sponge", "levels", "clay"])
@pytest.mark.parametrize("inverted", [False, True])
def test_self_similarity_2d(renderer, style, inverted):
    """x -> 9x - 2 fixes 1/4, and 48 levels is a multiple of the palette length, so every style
    matches between depth 3^-12 and 3^-60."""
    st = StyleSettings(style=style, inverted=inverted)
    a = render_still(cam_at((0.25, 0.25), 1.3 * 3.0 ** -12), st, samples=4, renderer=renderer)
    b = render_still(cam_at((0.25, 0.25), 1.3 * 3.0 ** -60), st, samples=4, renderer=renderer)
    assert mad(a, b) < 1.0, mad(a, b)
