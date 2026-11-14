"""Still images through the real shaders (any OpenGL 3.3; here Mesa llvmpipe), with jittered
accumulation.  Usage: xvfb-run -a uv run --with moderngl --with numpy --with pillow python gpu_render.py OUTDIR
"""
import sys, math
from pathlib import Path
import numpy as np, moderngl
from PIL import Image
from sponge_ref import *
from shade_ref import PALETTES, BG_DARK, STYLES
from glsl_util import program, set_table, set_if
from camera_ref import CameraND

def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r

class Still:
    def __init__(self, size=(960, 720)):
        self.ctx = moderngl.create_standalone_context(require=330)
        self.W, self.H = size
        self.fbo = self.ctx.framebuffer([self.ctx.texture(size, 4, dtype="f4")])
        self.march = program(self.ctx, "march.frag"); self.carpet = program(self.ctx, "carpet.frag")

    def _accumulate(self, prog, vao, samples):
        acc = np.zeros((self.H, self.W, 3), np.float64)
        for s in range(samples):
            prog["uJitter"] = (halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5) if samples > 1 else (0.0, 0.0)
            self.fbo.use(); vao.render(moderngl.TRIANGLES, vertices=3)
            acc += np.frombuffer(self.fbo.read(components=4, dtype="f4"), dtype="f4").reshape(self.H, self.W, 4)[:, :, :3]
        img = np.clip(acc / samples, 0, 1) ** (1 / 2.2)
        return Image.fromarray((img[::-1] * 255 + 0.5).astype(np.uint8))

    def scene(self, cam: CameraND, style="sponge", palette="pastel", samples=8, fog=True):
        prog, vao = self.march
        cam.pix = 2 * math.tan(math.radians(25.0)) / self.H; cam.refresh(); cam.update_sigma()
        set_table(prog, cam.table)
        prog["uFrame"].write(cam.F.astype("f4").T.tobytes())
        set_if(prog, uPix=cam.pix, uFar=cam.far, uRes=(self.W, self.H), uMaxSteps=300, uLod=1.0, uStyle=STYLES[style],
               uPalette=PALETTES[palette], uBg=tuple(BG_DARK), uFog=1.0 if fog else 0.0, uAo=1.0, uWRef=cam.w_ref(), uOutMode=0)
        return self._accumulate(prog, vao, samples)

    def carpet2d(self, pos: Pos, height: float, m=2, max_level=None, style="sponge", palette="pastel", samples=4, invert=False, cut=None):
        prog, vao = self.carpet
        height = normalize_2d(pos, height); pixel = height / self.H
        tb = level_table(pos, 2, m, max_level, far=0.55 * math.hypot(self.W, self.H) * pixel, fine_min=pixel / 3)
        set_table(prog, tb)
        cut = cut if cut is not None else (2.0 if m == 1 else 1.0 / 3.0)
        set_if(prog, uRes=(self.W, self.H), uPixel=pixel, uSS=3, uCut=cut, uStyle=STYLES[style], uPalette=PALETTES[palette],
               uBg=tuple(BG_DARK), uInvert=1.0 if invert else 0.0, uOutMode=0)
        return self._accumulate(prog, vao, samples)

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    st = Still()
    st.carpet2d(Pos.from_floats([0.5, 0.5, 0, 0]), 1.15).save(out / "carpet_sponge.png")
    p = Pos.from_floats([0.25, 0.75, 0, 0]); p.rescale(150); st.carpet2d(p, 1.6, style="levels").save(out / "carpet_levels_depth150.png")
    st.carpet2d(Pos.from_floats([0.5, 0.5, 0, 0]), 1.15, m=1, style="clay").save(out / "dust2d_clay.png")
    st.scene(CameraND(3)).save(out / "sponge3d.png")
    cam = CameraND(3)
    for k in range(40): cam.wheel(1, (0.12, -0.05, 1.0))
    st.scene(cam).save(out / "sponge3d_wheel40.png")
    for name in ("diagonal", "dust", "generic", "face diagonal", "cell diagonal"):
        cam = CameraND(4); cam.preset(name); st.scene(cam, style="sponge").save(out / f"preset_{name.replace(' ', '_')}.png")
    cam = CameraND(4); cam.preset("diagonal"); cam.drag_turn(140, 60); st.scene(cam, style="axes").save(out / "diagonal_turned_axes.png")
    cam = CameraND(4); cam.preset("generic"); st.scene(cam, style="wdepth").save(out / "generic_wdepth.png")
    cam = CameraND(4); cam.preset("generic")
    for k in range(60): cam.wheel(1, (0.05, 0.02, 1.0))
    st.scene(cam, style="levels").save(out / "generic_wheel60_levels.png")
    cam = CameraND(4, m=3); cam.preset("diagonal"); st.scene(cam, style="sponge").save(out / "diagonal_m3.png")
    names = ["carpet_sponge", "carpet_levels_depth150", "dust2d_clay", "sponge3d", "sponge3d_wheel40", "preset_diagonal",
             "preset_dust", "preset_generic", "preset_face_diagonal", "preset_cell_diagonal", "diagonal_turned_axes",
             "generic_wdepth", "generic_wheel60_levels", "diagonal_m3"]
    from PIL import ImageDraw
    tw, th = 480, 360
    sheet = Image.new("RGB", (tw * 4, (th + 22) * 4), (16, 16, 20))
    for i, nm in enumerate(names):
        x, y = (i % 4) * tw, (i // 4) * (th + 22)
        sheet.paste(Image.open(out / f"{nm}.png").resize((tw, th), Image.LANCZOS), (x, y))
        ImageDraw.Draw(sheet).text((x + 6, y + th + 4), nm.replace("_", " "), fill=(230, 230, 230))
    sheet.save(out / "contact_sheet.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))
