"""GPU checks on Mesa llvmpipe (or any OpenGL 3.3): the shaders against the CPU references.
Run:  xvfb-run -a uv run --with moderngl --with numpy --with pillow python check_shaders.py
"""
import time
from fractions import Fraction as F
import numpy as np, moderngl
from sponge_ref import *
from shade_ref import shade, PALETTES, BG_DARK, STYLES
from glsl_util import program, set_table, set_if
from render import frame_look, slice_basis, zoom_in

ctx = moderngl.create_standalone_context(require=330)
print(ctx.info["GL_RENDERER"], "|", ctx.info["GL_VERSION"])
W, H = 320, 240; fov = 50.0; pix = 2 * np.tan(np.radians(fov) / 2) / H
tex = ctx.texture((W, H), 4, dtype="f4"); fbo = ctx.framebuffer([tex])
mprog, mvao = program(ctx, "march.frag")
cprog, cvao = program(ctx, "carpet.frag")

def draw(vao):
    fbo.use(); t0 = time.time(); vao.render(moderngl.TRIANGLES, vertices=3); ctx.finish()
    return np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H * W, 4).copy(), time.time() - t0

def rays(F3):
    ys, xs = np.mgrid[0:H, 0:W]
    dc = np.stack([(xs + 0.5 - W / 2) * pix, (ys + 0.5 - H / 2) * pix, np.ones((H, W))], -1).reshape(-1, 3).astype(np.float32)
    dc /= np.linalg.norm(dc, axis=1)[:, None]
    return dc, (dc @ F3.T.astype(np.float32)).astype(np.float32), ((ys + 0.5) / H).reshape(-1)

# ---------------------------------------------------------------- 3D / 4D marcher
c = np.array([.5, .5, .5, .5]); B3 = np.eye(4)[:, :3]
scenes = []
eye = np.array([.5, .5, .5, 0]) + B3 @ np.array([1.45, 1.05, -1.75])
scenes.append(("3D overview", Pos.from_floats(eye), frame_look(eye, [.5, .5, .5, 0], [0, 1, 0], B3), 3, 2, 80.0, "sponge"))
F3 = frame_look(eye, [.43, .61, .17, 0], [0, 1, 0], B3)
scenes.append(("3D depth 3^-88", zoom_in(eye, F3, 3, 2, 80)[0], F3, 3, 2, 240.0, "levels"))
B = slice_basis([1, 1, 1, 1]); eye = c + B @ np.array([1.5, 1.1, -1.9])
scenes.append(("4D diagonal slice", Pos.from_floats(eye), frame_look(eye, c, [0, 1, 0], B), 4, 2, 80.0, "axes"))
B = slice_basis([1, 2, 3, 4]); eye = c + B @ np.array([1.5, 1.1, -1.9]); F3 = frame_look(eye, c + B @ np.array([.11, .07, 0]), [0, 1, 0], B)
scenes.append(("4D oblique, depth 3^-88", zoom_in(eye, F3, 4, 2, 80)[0], F3, 4, 2, 240.0, "wdepth"))
scenes.append(("4D oblique m=3, 3^-44", zoom_in(eye, F3, 4, 3, 40)[0], F3, 4, 3, 240.0, "clay"))
for name, pos, F3, n, m, far, style in scenes:
    de = normalize(pos, n, m); tb = level_table(pos, n, m, far=far, fine_min=0.3 * de * pix)
    Fr = np.zeros((4, 4), dtype="f4"); Fr[:, :3] = F3
    set_table(mprog, tb)
    mprog["uFrame"].write(Fr.T.tobytes())
    set_if(mprog, uPix=pix, uFar=far, uRes=(W, H), uJitter=(0.0, 0.0), uMaxSteps=220, uLod=1.0, uStyle=STYLES[style],
           uPalette=PALETTES["pastel"], uBg=tuple(BG_DARK), uFog=1.0, uAo=1.0, uWRef=(0.3, 2.0), uOutMode=1)
    data, dt = draw(mvao)
    dc, D, yfrac = rays(F3)
    t, hit, steps = march(tb, D, pix, far, max_steps=220)
    hit = hit | ((steps >= 220) & (t <= far))
    d, row, axis = de_table(D * t[:, None], tb, lodcut=t * np.float32(pix), want_info=True)
    gt = data[:, 0]; gh = data[:, 1] > 0.5; both = gh & hit
    rel = np.abs(gt - t)[both] / t[both]
    mprog["uOutMode"] = 0
    colg, dt2 = draw(mvao)
    colc = shade(tb, F3, D, dc, t, hit, pix, far, yfrac, style=style, wref=(0.3, 2.0))
    diff = np.abs(colg[:, :3] - colc)
    assert (gh == hit).mean() > 0.999 and (rel > 1e-3).mean() < 0.001 and diff.mean() < 2e-3
    print(f"{name:26s} E={pos.E:3d} rows={len(tb.w):2d} {style:7s} | hit agree {(gh == hit).mean() * 100:.2f}% | row agree "
          f"{(data[:, 2].astype(int) == row)[both].mean() * 100:.2f}% axis agree {(np.floor(data[:, 3]).astype(int) == axis % 4)[both].mean() * 100:.2f}%"
          f" | colour: mean diff {diff.mean():.1e}, pixels off by > 0.02: {(diff.max(axis=1) > 0.02).mean() * 100:.2f}% | shaded frame {dt2 * 1000:.0f} ms")

# ---------------------------------------------------------------- 2D carpet against exact membership
def carpet_case(name, pos, height, m, max_level=None):
    height = normalize_2d(pos, height); pixel = height / H
    tb = level_table(pos, 2, m, max_level, far=0.55 * np.hypot(W, H) * pixel, fine_min=pixel / 3)
    set_table(cprog, tb)
    set_if(cprog, uRes=(W, H), uPixel=pixel, uJitter=(0.0, 0.0), uSS=3, uCut=1.0 / 3.0, uStyle=0, uPalette=PALETTES["pastel"],
           uBg=tuple(BG_DARK), uInvert=0.0, uOutMode=1)
    data, dt = draw(cvao)
    g = data[:, 0].astype(int)
    ys, xs = np.mgrid[0:H, 0:W]
    q = np.zeros((H * W, 4), np.float32)
    q[:, 0] = ((xs + 0.5 - W / 2).astype(np.float32) * np.float32(pixel)).reshape(-1)
    q[:, 1] = ((ys + 0.5 - H / 2).astype(np.float32) * np.float32(pixel)).reshape(-1)
    r = removal_row(q, tb, np.float32(pixel / 3))
    # exact: every 7th pixel (Fractions are slow), level limit = finest row the shader may use
    used = [int(k) for k, w in zip(tb.level, tb.arrays(np.float32)[2]) if w >= np.float32(pixel / 3)]
    X = pos.exact(); s = F(3) ** pos.E; bad = 0; tot = 0
    for j in range(0, H * W, 7):
        x = [X[i] + F(float(q[j, i])) / s for i in range(2)]
        if any(v < 0 or v > 1 for v in x):
            want = -2
        else:
            k = removal_level([min(v, F(1) - F(1, 10 ** 400)) for v in x], m, max(used) if used else 0)
            want = -1 if k == 0 else k
        got = g[j] if g[j] < 0 else int(tb.level[g[j]])
        tot += 1; bad += (want != got)
    cmode = 0
    cprog["uOutMode"] = 0
    col, dt2 = draw(cvao)
    kinds = len(np.unique(g))
    assert (g == r).mean() > 0.999 and bad / tot < 0.005, (name, (g == r).mean(), bad, tot)
    print(f"{name:34s} E={pos.E:3d} rows={len(tb.w):2d} | GLSL = numpy on {(g == r).mean() * 100:.2f}% | exact check: {bad} of {tot} differ"
          f" | {kinds} classes in view | colour frame {dt2 * 1000:.0f} ms, mean {col[:, :3].mean():.2f}")

carpet_case("carpet, whole square", Pos.from_floats([0.5, 0.5, 0, 0]), 1.3, 2)
carpet_case("carpet, corner outside view", Pos.from_floats([0.02, 0.97, 0, 0]), 0.4, 2)
carpet_case("carpet, level cap 3", Pos.from_floats([0.5, 0.5, 0, 0]), 1.1, 2, 3)
carpet_case("Cantor dust (m=1)", Pos.from_floats([0.25, 0.25, 0, 0]), 0.6, 1)
p = Pos.from_floats([0.25, 0.25, 0, 0]); p.rescale(40); carpet_case("carpet at (1/4,1/4), depth 3^-40", p, 2.2, 2)
p = Pos.from_floats([0.25, 0.75, 0, 0]); p.rescale(150); carpet_case("carpet at (1/4,3/4), depth 3^-150", p, 1.4, 2)
p = Pos.from_floats([0.25, 0.25, 0, 0]); p.rescale(300); p.move([0.37, -0.21, 0, 0]); carpet_case("carpet, depth 3^-300, off centre", p, 2.9, 2)
p = Pos.from_floats([1 / 3, 0.25, 0, 0]); p.rescale(80); p.move([0.2, 0.1, 0, 0]); carpet_case("carpet, on the edge of a big hole", p, 1.2, 2)
print("all shader checks passed")
