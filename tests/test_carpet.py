"""The 2D shader against numpy and exact membership (design 14.4, 17.3), from the carpet half of
prototype/check_shaders.py.  Needs an OpenGL 3.3 context."""
from fractions import Fraction as F

import numpy as np
import pytest

from hypersponge.core.camera import Camera2D
from hypersponge.core.position import Pos
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import removal_level, removal_row

pytestmark = pytest.mark.gpu
W, H = 320, 240


def place(xy, E=0, move=None):
    p = Pos.from_floats([xy[0], xy[1], 0, 0])
    if E:
        p.rescale(E)
    if move:
        p.move([move[0], move[1], 0, 0])
    return p


CASES = [
    ("carpet, whole square", place((0.5, 0.5)), 1.3, 2, None),
    ("carpet, corner outside view", place((0.02, 0.97)), 0.4, 2, None),
    ("carpet, level cap 3", place((0.5, 0.5)), 1.1, 2, 3),
    ("Cantor dust (m=1)", place((0.25, 0.25)), 0.6, 1, None),
    ("carpet at (1/4,1/4), depth 3^-40", place((0.25, 0.25), 40), 2.2, 2, None),
    ("carpet at (1/4,3/4), depth 3^-150", place((0.25, 0.75), 150), 1.4, 2, None),
    ("carpet, depth 3^-300, off centre", place((0.25, 0.25), 300, (0.37, -0.21)), 2.9, 2, None),
    ("carpet, on the edge of a big hole", place((1 / 3, 0.25), 80, (0.2, 0.1)), 1.2, 2, None),
]


@pytest.mark.parametrize("name,pos,height,m,max_level", CASES, ids=[c[0] for c in CASES])
def test_carpet_shader(renderer, name, pos, height, m, max_level):
    cam = Camera2D(Rule(2, m, max_level), W, H)
    cam.pos = pos.copy()
    cam.height = height
    cam.refresh()
    g = renderer.data_2d(cam).reshape(-1)
    pixel = cam.pixel
    ys, xs = np.mgrid[0:H, 0:W]
    q = np.zeros((H * W, 4), np.float32)
    q[:, 0] = ((xs + 0.5 - W / 2).astype(np.float32) * np.float32(pixel)).reshape(-1)
    q[:, 1] = ((ys + 0.5 - H / 2).astype(np.float32) * np.float32(pixel)).reshape(-1)
    tb = cam.table
    r = removal_row(q, tb, np.float32(pixel / 3))
    assert (g == r).mean() > 0.999, (g == r).mean()
    # exact membership on every 7th pixel, up to the finest row the shader may use
    used = [int(k) for k, w in zip(tb.level, tb.arrays(np.float32)[2]) if w >= np.float32(pixel / 3)]
    X = cam.pos.exact()
    s = F(3) ** cam.pos.E
    bad = tot = 0
    for j in range(0, H * W, 7):
        x = [X[i] + F(float(q[j, i])) / s for i in range(2)]
        if any(v < 0 or v > 1 for v in x):
            want = -2
        else:
            k = removal_level([min(v, F(1) - F(1, 10 ** 400)) for v in x], m, max(used) if used else 0)
            want = -1 if k == 0 else k
        got = g[j] if g[j] < 0 else int(tb.level[g[j]])
        tot += 1
        bad += want != got
    assert bad / tot < 0.005, (bad, tot)


def test_info_shader_check(ctx):
    from hypersponge.cli import shader_check
    for name, ok, detail in shader_check(ctx):
        assert ok, (name, detail)
