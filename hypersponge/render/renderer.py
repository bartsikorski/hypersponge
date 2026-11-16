"""The renderer: shader programs, targets, accumulation, dynamic resolution (design section 16).

While the view changes, each frame draws one sample into the scene target at the current render
scale.  While it is still, frames add jittered full-size samples into the accumulation target, up
to max_samples.  present() puts whichever holds the newest picture on screen with gamma 1/2.2.

Long-frame guard (design 16.3): when a moving frame at the lowest scale takes over 100 ms, or a
still sample over 250 ms, the renderer switches to bands.  Each frame then draws one horizontal
band (a scissored draw): moving frames update the scene band by band, and a still sample is built
band by band in its own target and added to the accumulation when complete.  The number of bands
doubles while one band takes over 100 ms and halves while a whole pass takes under 40 ms.
"""
from __future__ import annotations

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
LONG_MOVING_MS = 100.0    # long-frame guard: a moving frame at SCALE_MIN slower than this ...
LONG_STILL_MS = 250.0     # ... or a still sample slower than this switches to bands
BANDS_START = 8
BANDS_MAX = 64
BAND_SLOW_MS = 100.0      # a band slower than this doubles the number of bands
PASS_FAST_MS = 40.0       # a whole pass faster than this halves it


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
        self.bands = 1                          # 1: whole frames; more: the long-frame guard is on
        self.next_band = 0                      # moving frames: the band drawn next
        self.pending = 0                        # still samples: bands of the next sample drawn so far
        self.pass_ms = 0.0                      # time of the bands of that sample so far
        self.band_tex = self.band_fbo = None
        self.last_ms = 0.0                      # time of the last draw (one band in band mode)

    # ---- targets -----------------------------------------------------------------------------
    def ensure_size(self, W: int, H: int) -> None:
        if (W, H) == self.size:
            return
        for name in ("scene_fbo", "scene_tex", "accum_fbo", "accum_tex", "out_fbo", "out_tex", "band_fbo", "band_tex"):
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
            self.pending = 0
            self.pass_ms = 0.0
            w, h = (max(1, round(W * self.scale)), max(1, round(H * self.scale))) if self.dynamic else (W, H)
            if self.bands == 1:
                ms = self._timed(lambda: self._draw(cam, st, self.scene_fbo, w, h, (0.0, 0.0)))
                if self.dynamic:
                    self.moving_ms = ms
                    self._adapt(ms)
                    if ms > LONG_MOVING_MS and self.scale <= SCALE_MIN:
                        self._start_bands()
            else:
                band = self.next_band % self.bands
                self.next_band = band + 1
                ms = self._timed(lambda: self._draw(cam, st, self.scene_fbo, w, h, (0.0, 0.0), band))
                self.moving_ms = ms
                self._adapt_bands(ms, ms * self.bands)
            self.current = ("scene", w, h)
            return True
        if self.samples >= self.max_samples:
            return False
        jit = jitter(self.samples + 1)
        if self.bands == 1:
            if self.samples == 0:
                self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
            ms = self._timed(lambda: self._blended(lambda: self._draw(cam, st, self.accum_fbo, W, H, jit)))
            self.samples += 1
            self.current = ("accum", W, H)
            if self.dynamic and ms > LONG_STILL_MS:
                self._start_bands()
            return True
        # band mode: build the sample band by band in its own target, add it when complete
        if self.band_tex is None:
            self.band_tex = self.ctx.texture((W, H), 4, dtype="f4")
            self.band_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.band_fbo = self.ctx.framebuffer(color_attachments=[self.band_tex])
        band = self.pending
        ms = self._timed(lambda: self._draw(cam, st, self.band_fbo, W, H, jit, band))
        self.pending += 1
        self.pass_ms += ms
        if self.pending >= self.bands:
            if self.samples == 0:
                self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._blended(lambda: self._copy(self.band_tex, self.accum_fbo))
            self.samples += 1
            self.pending = 0
            self.current = ("accum", W, H)
            total, self.pass_ms = self.pass_ms, 0.0
            self._adapt_bands(ms, total)
        elif ms > BAND_SLOW_MS and self.pending == 1:
            self._adapt_bands(ms, ms * self.bands)       # restarts the sample with more bands
        return True

    def _timed(self, fn) -> float:
        """Run a draw; with dynamic resolution wait for the GPU and return the time in ms."""
        t0 = time.perf_counter()
        fn()
        if self.dynamic:
            self.ctx.finish()
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_ms = ms
        return ms

    def _blended(self, fn) -> None:
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        try:
            fn()
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _start_bands(self) -> None:
        self.bands = BANDS_START
        self.next_band = 0
        self.pending = 0
        self.pass_ms = 0.0

    def _adapt_bands(self, band_ms: float, pass_ms: float) -> None:
        """More bands while one band is slow, fewer while a whole pass is fast (1 = guard off)."""
        if band_ms > BAND_SLOW_MS and self.bands < BANDS_MAX:
            self.bands *= 2
        elif pass_ms < PASS_FAST_MS and self.bands > 1:
            self.bands //= 2
        else:
            return
        self.pending = 0                         # a sample in progress starts again
        self.pass_ms = 0.0

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

    def _draw(self, cam, st: StyleSettings, fbo, w: int, h: int, jit, band: int | None = None) -> None:
        """One sample of camera cam into fbo at w x h; with band set, only that band of rows."""
        fbo.use()
        fbo.viewport = (0, 0, w, h)
        if band is not None:
            y0 = band * h // self.bands
            y1 = (band + 1) * h // self.bands
            fbo.scissor = (0, y0, w, max(1, y1 - y0))
            fbo.use()
        try:
            self._uniforms(cam, st, w, h, jit).draw()
        finally:
            if band is not None:
                fbo.scissor = None
                fbo.use()

    def _uniforms(self, cam, st: StyleSettings, w: int, h: int, jit, out_mode: int = 0) -> Program:
        jit = tuple(float(v) for v in jit)
        if cam.mode == "2d":
            p = self.carpet
            p.set_table(cam.table)
            p.set(uRes=(float(w), float(h)), uPixel=float(cam.height / h), uJitter=jit,
                  uSS=int(st.supersample), uCut=float(cam.cut), uStyle=STYLES[st.style_for("2d")],
                  uPalette=PALETTES[st.palette], uBg=tuple(st.bg), uInvert=1.0 if st.inverted else 0.0,
                  uOutMode=out_mode)
            return p
        p = self.march
        p.set_table(cam.table)
        p.prog["uFrame"].write(cam.F.astype("f4").T.tobytes())
        p.set(uPix=float(cam.pix * cam.H / h), uFar=float(cam.far), uRes=(float(w), float(h)), uJitter=jit,
              uMaxSteps=int(st.max_steps), uLod=float(st.lod), uStyle=STYLES[st.style_for(cam.mode)],
              uPalette=PALETTES[st.palette], uBg=tuple(st.bg), uFog=1.0 if st.fog else 0.0,
              uAo=float(st.occlusion), uWRef=(0.0, float(cam.sigma)), uOutMode=out_mode)
        return p

    def _copy(self, tex, fbo) -> None:
        """Linear copy of a full-size texture into fbo (no gamma)."""
        W, H = self.size
        fbo.use()
        fbo.viewport = (0, 0, W, H)
        tex.use(0)
        self.blit.set(uTex=0, uScale=1.0, uViewport=(0.0, 0.0, float(W), float(H)), uSrcScale=(1.0, 1.0),
                      uTexSize=(float(W), float(H)), uRaw=1)
        self.blit.draw()

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
                      uViewport=tuple(float(v) for v in vp), uSrcScale=(w / W, h / H), uTexSize=(float(W), float(H)),
                      uRaw=0)
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

    def _data(self, cam, st=None) -> np.ndarray:
        """Data mode at full size without jitter: (H, W, 4) float32, bottom row first."""
        W, H = cam.W, cam.H
        if self.data_tex is None or self.data_tex.size != (W, H):
            if self.data_tex is not None:
                self.data_fbo.release()
                self.data_tex.release()
            self.data_tex = self.ctx.texture((W, H), 4, dtype="f4")
            self.data_fbo = self.ctx.framebuffer(color_attachments=[self.data_tex])
        self.data_fbo.use()
        self.data_fbo.viewport = (0, 0, W, H)
        p = self._uniforms(cam, st or StyleSettings(), W, H, (0.0, 0.0), out_mode=1)
        p.draw()
        p.set(uOutMode=0)
        return np.frombuffer(self.data_fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H, W, 4).copy()

    def data_2d(self, cam) -> np.ndarray:
        """removalRow at every pixel centre (cut: one third of a pixel), shape (H, W), as int."""
        return self._data(cam)[:, :, 0].astype(int)

    def data_nd(self, cam, st=None) -> np.ndarray:
        """The marcher's data output per pixel: (t, hit, row, axis + steps / 1024), (H, W, 4)."""
        return self._data(cam, st)


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
