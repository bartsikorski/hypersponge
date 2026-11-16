"""The ray marcher against numpy (design 14.3, 17.2), from the marcher half of
prototype/check_shaders.py: hits, distances and colours of march.frag against core/sdf.py
and shade_ref.py.  Needs an OpenGL 3.3 context."""
import numpy as np
import pytest

from hypersponge.core.frame import slice_basis
from hypersponge.core.levels import level_table, normalize
from hypersponge.core.position import Pos
from hypersponge.core.sdf import de_table, march
from hypersponge.render.renderer import Program
from hypersponge.render.styles import PALETTES
from shade_ref import BG_DARK, STYLES, shade

pytestmark = pytest.mark.gpu
W, H = 320, 240
PIX = 2 * np.tan(np.radians(25.0)) / H


def frame_look(eye, target, up, basis):
    """Camera frame inside the slice spanned by basis (4x3, orthonormal columns)."""
    fwd = basis.T @ (np.asarray(target, float) - np.asarray(eye, float))
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.asarray(up, float), fwd)
    right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    return basis @ np.stack([right, upv, fwd], axis=1)


def zoom_in(eye, F3, n, m, steps, frac=0.7):
    """The wheel-zoom rule of the prototype: a fraction of the way to the surface ahead."""
    pos = Pos.from_floats(eye, 0)
    pix = 2 * np.tan(np.radians(25.0)) / 540
    fwd = F3[:, 2].astype(np.float64)
    for _ in range(steps):
        de = normalize(pos, n, m)
        tb = level_table(pos, n, m, far=1e4, fine_min=0.3 * de * pix)
        t, hit, _ = march(tb, fwd[None, :], pix, 1e4, max_steps=400)
        assert hit[0], "centre ray missed"
        pos.move(fwd * (frac * float(t[0])))
    return pos


def scenes():
    c = np.array([.5, .5, .5, .5])
    B3 = np.eye(4)[:, :3]
    out = []
    eye = np.array([.5, .5, .5, 0]) + B3 @ np.array([1.45, 1.05, -1.75])
    out.append(("3D overview", Pos.from_floats(eye), frame_look(eye, [.5, .5, .5, 0], [0, 1, 0], B3), 3, 2, 80.0,
                "sponge"))
    F3 = frame_look(eye, [.43, .61, .17, 0], [0, 1, 0], B3)
    out.append(("3D depth 3^-88", zoom_in(eye, F3, 3, 2, 80), F3, 3, 2, 240.0, "levels"))
    B = slice_basis([1, 1, 1, 1])[:, :3]
    eye = c + B @ np.array([1.5, 1.1, -1.9])
    out.append(("4D diagonal slice", Pos.from_floats(eye), frame_look(eye, c, [0, 1, 0], B), 4, 2, 80.0, "axes"))
    B = slice_basis([1, 2, 3, 4])[:, :3]
    eye = c + B @ np.array([1.5, 1.1, -1.9])
    F3 = frame_look(eye, c + B @ np.array([.11, .07, 0]), [0, 1, 0], B)
    out.append(("4D oblique, depth 3^-88", zoom_in(eye, F3, 4, 2, 80), F3, 4, 2, 240.0, "wdepth"))
    out.append(("4D oblique m=3, 3^-44", zoom_in(eye, F3, 4, 3, 40), F3, 4, 3, 240.0, "clay"))
    return out


@pytest.fixture(scope="module")
def target(ctx):
    tex = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([tex])
    return fbo, Program(ctx, "march.frag")


def draw(fbo, prog):
    fbo.use()
    fbo.viewport = (0, 0, W, H)
    prog.draw()
    return np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H * W, 4).copy()


SCENES = scenes()


@pytest.mark.parametrize("name,pos,F3,n,m,far,style", SCENES, ids=[s[0] for s in SCENES])
def test_marcher_matches_numpy(target, name, pos, F3, n, m, far, style):
    fbo, prog = target
    pos = pos.copy()
    de = normalize(pos, n, m)
    tb = level_table(pos, n, m, far=far, fine_min=0.3 * de * PIX)
    Fr = np.zeros((4, 4), dtype="f4")
    Fr[:, :3] = F3
    prog.table_key = None
    prog.set_table(tb)
    prog.prog["uFrame"].write(Fr.T.tobytes())
    prog.set(uPix=PIX, uFar=far, uRes=(W, H), uJitter=(0.0, 0.0), uMaxSteps=220, uLod=1.0, uStyle=STYLES[style],
             uPalette=PALETTES["pastel"], uBg=tuple(BG_DARK), uFog=1.0, uAo=1.0, uWRef=(0.3, 2.0), uOutMode=1)
    data = draw(fbo, prog)
    ys, xs = np.mgrid[0:H, 0:W]
    dc = np.stack([(xs + 0.5 - W / 2) * PIX, (ys + 0.5 - H / 2) * PIX, np.ones((H, W))], -1)
    dc = dc.reshape(-1, 3).astype(np.float32)
    dc /= np.linalg.norm(dc, axis=1)[:, None]
    D = (dc @ F3.T.astype(np.float32)).astype(np.float32)
    yfrac = ((ys + 0.5) / H).reshape(-1)
    t, hit, steps = march(tb, D, PIX, far, max_steps=220)
    hit = hit | ((steps >= 220) & (t <= far))
    _, row, axis = de_table(D * t[:, None], tb, lodcut=t * np.float32(PIX), want_info=True)
    gh = data[:, 1] > 0.5
    both = gh & hit
    rel = np.abs(data[:, 0] - t)[both] / t[both]
    assert (gh == hit).mean() > 0.999, (gh == hit).mean()
    assert (rel > 1e-3).mean() < 0.001, (rel > 1e-3).mean()
    assert (data[:, 2].astype(int) == row)[both].mean() > 0.99
    assert (np.floor(data[:, 3]).astype(int) == axis % 4)[both].mean() > 0.99
    prog.set(uOutMode=0)
    colg = draw(fbo, prog)
    colc = shade(tb, F3, D, dc, t, hit, PIX, far, yfrac, style=style, wref=(0.3, 2.0))
    diff = np.abs(colg[:, :3] - colc)
    assert diff.mean() < 2e-3, diff.mean()
