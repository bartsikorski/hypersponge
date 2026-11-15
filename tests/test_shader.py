"""The 3D shaders against their numpy twins (design 8, 14.6), from the marcher half of
prototype/check_shaders.py.  Needs an OpenGL 3.3 context.

- A probe shader built from common.glsl evaluates de and deInfo at 20,000 points.
- march.frag in data mode against the numpy marcher on 6 scenes.
- march.frag colours against tests/shade_ref.py for every 3D style.
"""
import numpy as np
import pytest

from hypersponge.cli import march_check
from hypersponge.core.camera import CameraND
from hypersponge.core.rule import Rule
from hypersponge.core.sdf import de_table, march
from hypersponge.render.renderer import SHADERS, VERT
from hypersponge.render.styles import StyleSettings
from shade_ref import shade

pytestmark = pytest.mark.gpu

PROBE = """
uniform sampler2D uPoints;
uniform float uLodcut;
out vec4 fragColor;
void main() {
    vec4 q = texelFetch(uPoints, ivec2(gl_FragCoord.xy), 0);
    int row, axis;
    float d = deInfo(q, uLodcut, row, axis);
    fragColor = vec4(d, de(q, uLodcut), float(row), float(axis));
}
"""


def camera(m=2, max_level=None, notches=0, cursor=(0.12, -0.05), size=(160, 120)):
    cam = CameraND(Rule(3, m, max_level), *size)
    for _ in range(notches):
        cam.wheel(1, (cursor[0], cursor[1], 1.0))
    cam.update_sigma()
    return cam


SCENES = {
    "overview": (lambda: camera(), "sponge"),
    "40 notches": (lambda: camera(notches=40), "levels"),
    "150 notches": (lambda: camera(notches=150, cursor=(0.05, 0.02)), "axes"),
    "m = 3": (lambda: camera(m=3), "clay"),
    "max level 2": (lambda: camera(max_level=2), "axes"),
    "dust, 30 notches": (lambda: camera(m=1, notches=30), "levels"),
}


@pytest.fixture(scope="module")
def probe(ctx):
    from hypersponge.render.renderer import Program
    p = Program.__new__(Program)
    p.ctx = ctx
    p.prog = ctx.program(vertex_shader=VERT, fragment_shader="#version 330\n"
                         + (SHADERS / "common.glsl").read_text(encoding="utf-8") + "\n" + PROBE)
    buf = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4"))
    p.vao = ctx.vertex_array(p.prog, [(buf, "2f", "p")])
    p.table_key = None
    return p


@pytest.mark.parametrize("name", ["overview", "40 notches", "150 notches", "m = 3"])
def test_probe_de_and_deinfo(ctx, probe, name):
    cam = camera(**{"overview": {}, "40 notches": {"notches": 40}, "150 notches": {"notches": 150},
                    "m = 3": {"m": 3}}[name])
    W, H = 200, 100
    rng = np.random.default_rng(len(name))
    q = rng.normal(0, 1, (W * H, 4)) * (10.0 ** rng.uniform(-2, 2.5, W * H))[:, None]
    q[:, 3] = 0.0
    q = q.astype(np.float32)
    tex = ctx.texture((W, H), 4, q.tobytes(), dtype="f4")
    out = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[out])
    lodcut = np.float32(0.01)
    try:
        probe.set_table(cam.table)
        tex.use(0)
        probe.prog["uPoints"].value = 0
        probe.prog["uLodcut"].value = float(lodcut)
        fbo.use()
        fbo.viewport = (0, 0, W, H)
        probe.draw()
        g = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(-1, 4)
    finally:
        fbo.release()
        out.release()
        tex.release()
    d, row, axis = de_table(q, cam.table, lodcut=lodcut, want_info=True)
    scale = np.maximum(np.abs(d), np.abs(q).max(axis=1) * 1e-3 + 1e-6)
    err = np.abs(g[:, 0] - d) / scale
    assert err.max() < 1e-5, err.max()
    assert np.array_equal(g[:, 0], g[:, 1])                          # de and deInfo agree
    assert (g[:, 2].astype(int) == row).mean() > 0.999
    assert (g[:, 3].astype(int) == axis % 4).mean() > 0.999


@pytest.mark.parametrize("name", list(SCENES))
def test_marcher_against_numpy(renderer, name):
    cam = SCENES[name][0]()
    hits, rows = march_check(renderer, cam)
    assert hits > 0.995 and rows > 0.995, (hits, rows)


@pytest.mark.parametrize("name", list(SCENES))
def test_colours_against_shade_ref(renderer, name):
    make, style = SCENES[name]
    cam = make()
    st = StyleSettings(style=style)
    g = renderer.colour_3d(cam, st).reshape(-1, 3)
    W, H = cam.W, cam.H
    pix = cam.pix
    ys, xs = np.mgrid[0:H, 0:W]
    dc = np.stack([(xs + 0.5 - W / 2) * pix, (ys + 0.5 - H / 2) * pix, np.ones((H, W))], -1).reshape(-1, 3)
    dc = (dc / np.linalg.norm(dc, axis=1)[:, None]).astype(np.float32)
    F3 = cam.F[:, :3]
    D = (dc @ F3.T.astype(np.float32)).astype(np.float32)
    t, hit, _ = march(cam.table, D, pix, cam.far, max_steps=st.max_steps, lod=st.lod)
    want = shade(cam.table, F3, D, dc, t, hit, pix, cam.far, ((ys + 0.5) / H).reshape(-1), style=style,
                 palette=st.palette, bg=np.array(st.bg), fog=st.fog, ao_strength=st.occlusion,
                 wref=(0.0, cam.sigma), lod=st.lod)
    diff = np.abs(g - want)
    assert diff.mean() < 2e-3 and (diff.max(axis=1) > 0.02).mean() < 0.005, (diff.mean(), (diff.max(axis=1) > 0.02).mean())
