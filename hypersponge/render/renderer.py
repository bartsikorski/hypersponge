"""The renderer: shader programs, targets, accumulation, dynamic resolution (design section 16).

While the view changes, each frame draws one sample into the scene target at the current render
scale.  While it is still, frames add jittered full-size samples into the accumulation target, up
to max_samples.  present() puts whichever holds the newest picture on screen with gamma 1/2.2.

Long-frame guard (16.3): when a moving frame at the lowest scale takes over LONG_MOVING_MS, or a
still sample over LONG_STILL_MS, drawing switches to bands: one horizontal band per frame through
a scissor rectangle, so that no single draw runs long enough for the driver to reset.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import moderngl
import numpy as np

from ..core.levels import MAXLEV
from .styles import PALETTES, STYLES, StyleSettings

SHADERS = Path(__file__).resolve().parent / "shaders"
VERT = "#version 330\nin vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n"

SCALE_MIN = 0.35          # dynamic resolution floor
SLOW_MS = 14.0            # above this a moving frame lowers the render scale
FAST_MS = 9.0             # below this it raises it again
SCALE_EVERY = 4           # frames between two scale changes
LONG_MOVING_MS = 100.0    # a moving frame at SCALE_MIN slower than this starts band mode
LONG_STILL_MS = 250.0     # ... and so does a still sample slower than this
LONG_BAND_MS = 100.0      # in band mode, a band slower than this doubles the number of bands
SHORT_BANDS_MS = 30.0     # all bands together faster than this halve it again
BANDS_START = 8
BANDS_MAX = 256


def shader_source(name: str, common: bool = True) -> str:
    src = "#version 330\n"
    if common:
        src += (SHADERS / "common.glsl").read_text(encoding="utf-8") + "\n"
    return src + (SHADERS / name).read_text(encoding="utf-8")


def halton(i: int, b: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def jitter(i: int) -> tuple:
    """Sub-pixel offset of accumulated sample i (1, 2, ...) in [-0.5, 0.5)."""
    return (halton(i, 2) - 0.5, halton(i, 3) - 0.5)


class Program:
    """A full-screen fragment program with upload helpers that skip unused uniforms."""

    def __init__(self, ctx, frag: str, common: bool = True):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=shader_source(frag, common))
        buf = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4"))
        self.vao = ctx.vertex_array(self.prog, [(buf, "2f", "p")])
        self.table_key = None

    def __contains__(self, name):
        return name in self.prog

    def set(self, **kw):
        for k, v in kw.items():
            if k not in self.prog:
                continue
            if isinstance(v, np.ndarray):
                self.prog[k].write(v.astype("f4").tobytes())
            else:
                self.prog[k].value = v

    def set_table(self, tb):
        """Upload a level table (design 14.2); skipped when it did not change."""
        key = tb.key()
        if key == self.table_key:
            return
        self.table_key = key
        lo, hi, w, blo, bhi = tb.arrays(np.float32)

        def pad(a, dtype="f4"):
            o = np.zeros((MAXLEV,) + a.shape[1:], dtype=dtype)
            o[:len(a)] = a
            return o.tobytes()
        p = self.prog
        p["uCount"].value = len(w)
        p["uLo"].write(pad(lo))
        p["uHi"].write(pad(hi))
        p["uW"].write(pad(w))
        if "uLevelNo" in p:
            p["uLevelNo"].write(pad(tb.level.astype("i4"), "i4"))
        p["uBoxLo"].value = tuple(float(v) for v in blo)
        p["uBoxHi"].value = tuple(float(v) for v in bhi)
        p["uOff"].value = tuple(float(v) for v in tb.off)
        p["uM"].value = int(tb.m)

    def draw(self):
        self.vao.render(moderngl.TRIANGLES, vertices=3)


class Renderer:
    def __init__(self, ctx, max_samples: int = 16, dynamic: bool = True):
        self.ctx = ctx
        self.carpet = Program(ctx, "carpet.frag")
        self.march = Program(ctx, "march.frag")
        self.blit = Program(ctx, "blit.frag", common=False)
        self.copy = Program(ctx, "copy.frag", common=False)
        self.guard = True                       # the long-frame guard may switch to bands
        self.bands = 1                          # 1 = whole frames; more = band mode
        self.band_i = {"scene": 0, "still": 0}  # next band to draw, moving and still
        self.band_ms = 0.0
        self.band_tex = self.band_fbo = None
        self.max_samples = max_samples
        self.dynamic = dynamic
        self.size = (0, 0)
        self.scene_tex = self.scene_fbo = self.accum_tex = self.accum_fbo = None
        self.out_tex = self.out_fbo = None
        self.data_tex = self.data_fbo = None
        self.scale = 1.0
        self.samples = 0
        self.key = None
        self.moving_ms = 0.0
        self.current = ("scene", 1, 1)          # which target holds the newest picture, and its size
        self._since_scale = 0

    # ---- targets -----------------------------------------------------------------------------
    def ensure_size(self, W: int, H: int) -> None:
        if (W, H) == self.size:
            return
        for name in ("scene_fbo", "scene_tex", "accum_fbo", "accum_tex", "out_fbo", "out_tex", "band_fbo",
                     "band_tex"):
            o = getattr(self, name)
            if o is not None:
                o.release()
            setattr(self, name, None)
        self.scene_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])
        self.accum_tex = self.ctx.texture((W, H), 4, dtype="f4")
        self.accum_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.accum_fbo = self.ctx.framebuffer(color_attachments=[self.accum_tex])
        self.size = (W, H)
        self.key = None
        self.samples = 0

    # ---- drawing -----------------------------------------------------------------------------
    def frame(self, cam, st: StyleSettings) -> bool:
        """Do this frame's drawing for camera cam.  Returns True if the picture changed."""
        W, H = cam.W, cam.H
        self.ensure_size(W, H)
        key = (cam.key(), st.key())
        if key != self.key:
            self.key = key
            self.samples = 0
            self.band_i["still"] = 0
            w, h = (max(1, round(W * self.scale)), max(1, round(H * self.scale))) if self.dynamic else (W, H)
            if self.bands > 1:
                self._draw_band(cam, st, self.scene_fbo, w, h, (0.0, 0.0), "scene")
            else:
                ms = self._timed(lambda: self._draw(cam, st, self.scene_fbo, w, h, (0.0, 0.0)))
                self.moving_ms = ms
                if self.dynamic:
                    self._adapt(ms)
                if self.guard and ms > LONG_MOVING_MS and (self.scale <= SCALE_MIN or not self.dynamic):
                    self._start_bands()
            self.current = ("scene", w, h)
            return True
        if self.samples < self.max_samples:
            jit = jitter(self.samples + 1)
            if self.bands > 1:
                self._ensure_band_target(W, H)
                if self.band_i["still"] == 0:
                    self.band_fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._draw_band(cam, st, self.band_fbo, W, H, jit, "still")
                if self.band_i["still"] != 0:                # the sample is not complete yet
                    return True
                if self.samples == 0:
                    self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._add(lambda: self._copy_band(W, H))
            else:
                if self.samples == 0:
                    self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
                ms = self._timed(lambda: self._add(lambda: self._draw(cam, st, self.accum_fbo, W, H, jit)))
                if self.guard and ms > LONG_STILL_MS:
                    self._start_bands()
            self.samples += 1
            self.current = ("accum", W, H)
            return True
        return False

    def _timed(self, fn) -> float:
        """Run a draw and return its time in ms, measured with ctx.finish() (design 16.3)."""
        t0 = time.perf_counter()
        fn()
        self.ctx.finish()
        return (time.perf_counter() - t0) * 1000.0

    def _add(self, fn) -> None:
        """Run a draw with additive blending (accumulation)."""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        try:
            fn()
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ---- long-frame guard --------------------------------------------------------------------
    def _start_bands(self) -> None:
        self.bands = max(self.bands, BANDS_START)
        self.band_i = {"scene": 0, "still": 0}

    def _ensure_band_target(self, W: int, H: int) -> None:
        if self.band_tex is None or self.band_tex.size != (W, H):
            if self.band_tex is not None:
                self.band_fbo.release()
                self.band_tex.release()
            self.band_tex = self.ctx.texture((W, H), 4, dtype="f4")
            self.band_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.band_fbo = self.ctx.framebuffer(color_attachments=[self.band_tex])

    def _draw_band(self, cam, st, fbo, w: int, h: int, jit, which: str) -> float:
        """Draw the next band of a w x h picture (which: "scene" while moving, "still" for an
        accumulated sample), then adapt the number of bands after the last one."""
        n = self.bands
        i = self.band_i[which] % n
        y0 = h * i // n
        y1 = h * (i + 1) // n
        ms = 0.0
        if y1 > y0:
            ms = self._timed(lambda: self._draw(cam, st, fbo, w, h, jit, scissor=(0, y0, w, y1 - y0)))
        self.band_ms = ms
        self.band_i[which] = (i + 1) % n
        if self.band_i[which] == 0 and self.guard:
            if ms > LONG_BAND_MS and n < BANDS_MAX:
                self.bands = n * 2
                self.band_i = {"scene": 0, "still": 0}
            elif ms * n < SHORT_BANDS_MS:
                self.bands = n // 2
                self.band_i = {"scene": 0, "still": 0}
        return ms

    def _copy_band(self, W: int, H: int) -> None:
        self.accum_fbo.use()
        self.accum_fbo.viewport = (0, 0, W, H)
        self.band_tex.use(0)
        self.copy.set(uTex=0)
        self.copy.draw()

    @property
    def complete(self) -> bool:
        return self.samples >= self.max_samples

    def _adapt(self, ms: float) -> None:
        self._since_scale += 1
        if self._since_scale < SCALE_EVERY:
            return
        if ms > SLOW_MS and self.scale > SCALE_MIN:
            self.scale = max(SCALE_MIN, self.scale * 0.85)
            self._since_scale = 0
        elif ms < FAST_MS and self.scale < 1.0:
            self.scale = min(1.0, self.scale * 1.1)
            self._since_scale = 0

    def _draw(self, cam, st: StyleSettings, fbo, w: int, h: int, jit, scissor=None, out_mode: int = 0) -> None:
        fbo.use()
        fbo.viewport = (0, 0, w, h)
        fbo.scissor = scissor
        try:
            self._draw_mode(cam, st, w, h, jit, out_mode)
        finally:
            fbo.scissor = None

    def _draw_mode(self, cam, st: StyleSettings, w: int, h: int, jit, out_mode: int) -> None:
        if cam.mode == "2d":
            p = self.carpet
            p.set_table(cam.table)
            p.set(uRes=(float(w), float(h)), uPixel=float(cam.height / h), uJitter=tuple(float(v) for v in jit),
                  uSS=int(st.supersample), uCut=float(cam.cut), uStyle=STYLES[st.style_for("2d")],
                  uPalette=PALETTES[st.palette], uBg=tuple(st.bg), uInvert=1.0 if st.inverted else 0.0,
                  uOutMode=out_mode)
            p.draw()
        else:
            p = self.march
            p.set_table(cam.table)
            p.prog["uFrame"].write(cam.F.astype("f4").T.tobytes())
            w_ref = cam.w_ref() if hasattr(cam, "w_ref") else (0.0, cam.sigma)
            p.set(uPix=float(2.0 * math.tan(math.radians(cam.fov) / 2.0) / h), uFar=float(cam.far),
                  uRes=(float(w), float(h)), uJitter=tuple(float(v) for v in jit), uMaxSteps=int(st.max_steps),
                  uLod=float(st.lod), uStyle=STYLES[st.style_for(cam.mode)], uPalette=PALETTES[st.palette],
                  uBg=tuple(st.bg), uFog=1.0 if st.fog else 0.0, uAo=float(st.occlusion),
                  uWRef=tuple(float(v) for v in w_ref), uOutMode=out_mode)
            p.draw()

    # ---- output ------------------------------------------------------------------------------
    def present(self, fbo, viewport=None) -> None:
        """Draw the newest picture into fbo (the window: ctx.screen) with gamma."""
        which, w, h = self.current
        W, H = self.size
        if W == 0:
            return
        tex = self.accum_tex if which == "accum" else self.scene_tex
        vp = viewport or (0, 0, W, H)
        fbo.use()
        fbo.viewport = vp
        tex.use(0)
        self.blit.set(uTex=0, uScale=1.0 / max(1, self.samples) if which == "accum" else 1.0,
                      uViewport=tuple(float(v) for v in vp), uSrcScale=(w / W, h / H), uTexSize=(float(W), float(H)))
        self.blit.draw()

    def picture(self) -> np.ndarray:
        """The newest picture as an (H, W, 3) uint8 array, top row first, without the interface."""
        W, H = self.size
        if self.out_tex is None:
            self.out_tex = self.ctx.texture((W, H), 3)
            self.out_fbo = self.ctx.framebuffer(color_attachments=[self.out_tex])
        self.present(self.out_fbo)
        data = self.out_fbo.read(components=3, alignment=1)
        return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()

    def data_3d(self, cam, st: StyleSettings | None = None) -> np.ndarray:
        """march.frag in data mode at every pixel centre, shape (H, W, 4): t, hit, row, axis +
        steps / 1024 (design 14.3).  For tests and `hypersponge info`."""
        W, H = cam.W, cam.H
        self._ensure_data_target(W, H)
        self._draw(cam, st or StyleSettings(), self.data_fbo, W, H, (0.0, 0.0), out_mode=1)
        return np.frombuffer(self.data_fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H, W, 4).copy()

    def colour_3d(self, cam, st: StyleSettings) -> np.ndarray:
        """One sample of march.frag at the pixel centres in linear RGB, (H, W, 3), before gamma."""
        W, H = cam.W, cam.H
        self._ensure_data_target(W, H)
        self._draw(cam, st, self.data_fbo, W, H, (0.0, 0.0))
        a = np.frombuffer(self.data_fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H, W, 4)
        return a[:, :, :3].copy()

    def _ensure_data_target(self, W: int, H: int) -> None:
        if self.data_tex is None or self.data_tex.size != (W, H):
            if self.data_tex is not None:
                self.data_fbo.release()
                self.data_tex.release()
            self.data_tex = self.ctx.texture((W, H), 4, dtype="f4")
            self.data_fbo = self.ctx.framebuffer(color_attachments=[self.data_tex])

    def data_2d(self, cam) -> np.ndarray:
        """removalRow at every pixel centre (cut: one third of a pixel), shape (H, W), as int."""
        W, H = cam.W, cam.H
        self._ensure_data_target(W, H)
        p = self.carpet
        p.set_table(cam.table)
        self.data_fbo.use()
        self.data_fbo.viewport = (0, 0, W, H)
        p.set(uRes=(float(W), float(H)), uPixel=float(cam.height / H), uJitter=(0.0, 0.0), uOutMode=1)
        p.draw()
        p.set(uOutMode=0)
        a = np.frombuffer(self.data_fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H, W, 4)
        return a[:, :, 0].astype(int)


def render_still(cam, st: StyleSettings | None = None, samples: int = 16, renderer: Renderer | None = None,
                 ctx=None) -> np.ndarray:
    """A still picture of camera cam (its W x H), accumulated over `samples` jittered samples."""
    if renderer is None:
        if ctx is None:
            from .context import make_headless_context
            ctx = make_headless_context()
        renderer = Renderer(ctx, max_samples=samples, dynamic=False)
    renderer.max_samples = samples
    renderer.dynamic = False
    renderer.key = None
    st = st or StyleSettings()
    renderer.frame(cam, st)
    while renderer.frame(cam, st):
        pass
    return renderer.picture()
