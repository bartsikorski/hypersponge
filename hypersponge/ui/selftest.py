"""Scripted GUI session (`gui --selftest REPORT.json`, used by tests/test_gui.py), design 18.3.

Each step is a generator that runs over frames (one `yield` per frame) with synthetic input and
returns a dict of findings; a step passes when it returns without an exception and without
`ok: False`.  Every step leaves a screenshot of the window (with the interface) next to the
report, in REPORT_shots/.  The report holds every step, the OpenGL renderer string, any OpenGL
error, every internal error of the app, and the longest frame.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from imgui_bundle import hello_imgui

from ..core.sdf import removal_row
from ..render.context import describe
from ..render.styles import StyleSettings

STEP_TIMEOUT = 300.0          # seconds a step may take (llvmpipe is slow)
MAX_FRAMES = 40000


def diff(a, b) -> float:
    """Mean absolute difference of two uint8 pictures, in units of 1/255."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


class SelfTest:
    def __init__(self, app, report: str):
        from .app import InputSnapshot
        self.Input = InputSnapshot
        self.app = app
        self.report_path = Path(report)
        self.shots = self.report_path.with_name(self.report_path.stem + "_shots")
        self.shots.mkdir(parents=True, exist_ok=True)
        app.screenshot_dir = self.shots
        self.snap = InputSnapshot()
        self.frame = 0
        self.pending_shot = None
        self.report = {"passed": False, "steps": [], "gl_errors": [], "errors": []}
        self.gen = None
        self.queue = None
        self.current = None
        self.t_step = time.perf_counter()
        self.finished = False

    # ---- hooks called by the app -------------------------------------------------------------
    def input(self):
        s = self.snap
        self.snap = self.Input(mouse=s.mouse, down=s.down, keys=set(s.keys), shift=s.shift)
        return s

    def before_swap(self):
        if self.pending_shot and self.app.ctx is not None:
            from PIL import Image
            W, H = self.app.fb
            ctx = self.app.ctx
            err = ctx.error                         # errors of this frame, before the read
            if err != "GL_NO_ERROR":
                self.report["gl_errors"].append((self.current, err))
            data = ctx.screen.read(viewport=(0, 0, W, H), components=3, alignment=1)
            # moderngl's read of the default framebuffer leaves GL_INVALID_OPERATION on Mesa
            # (it selects a colour attachment the window does not have) while the pixels are
            # right; the screenshot is only for people, so that one error is dropped here
            ctx.error
            img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]
            Image.fromarray(img).save(self.shots / f"{self.pending_shot}.png")
            self.pending_shot = None

    def after_frame(self):
        if self.finished:
            return
        self.frame += 1
        err = self.app.ctx.error
        if err != "GL_NO_ERROR":
            self.report["gl_errors"].append((self.current, err))
        if self.queue is None:
            self.queue = list(self.steps())
        if self.gen is None:
            if not self.queue:
                self.finish()
                return
            self.current, fn = self.queue.pop(0)
            self.gen = fn()
            self.t_step = time.perf_counter()
            self.errors_before = len(self.app.errors)
        try:
            next(self.gen)
            if time.perf_counter() - self.t_step > STEP_TIMEOUT:
                raise TimeoutError(f"step {self.current} took over {STEP_TIMEOUT:.0f} s")
        except StopIteration as stop:
            self.end_step(stop.value or {}, None)
        except Exception:
            self.end_step({}, traceback.format_exc())
        if self.frame > MAX_FRAMES:
            self.report["errors"].append(("frames", f"over {MAX_FRAMES} frames"))
            self.finish()

    def end_step(self, info: dict, error):
        new_errors = self.app.errors[self.errors_before:]
        ok = error is None and info.get("ok", True) and not new_errors
        if new_errors and error is None:
            error = "; ".join(w for w, _ in new_errors)
        rec = dict(name=self.current, passed=bool(ok), t=round(time.perf_counter() - self.t_step, 2),
                   frame=self.frame, error=error, **{k: v for k, v in info.items() if k != "ok"})
        self.report["steps"].append(rec)
        print("[selftest]", {k: v for k, v in rec.items() if k != "error"}, flush=True)
        if error:
            print(error, flush=True)
        self.gen = None
        self.snap = self.Input()
        self.pending_shot = self.current

    def finish(self):
        a = self.app
        self.finished = True
        self.report["renderer"] = describe(a.ctx) if a.ctx is not None else None
        self.report["longest_frame_ms"] = round(a.longest_frame_ms, 1)
        self.report["errors"] += [(w, tb) for w, tb in a.errors]
        self.report["passed"] = bool(self.report["steps"]) and all(s["passed"] for s in self.report["steps"]) \
            and not self.report["gl_errors"] and not self.report["errors"]
        self.report_path.write_text(json.dumps(self.report, indent=1, default=str), encoding="utf-8")
        hello_imgui.get_runner_params().app_shall_exit = True

    # ---- helpers -----------------------------------------------------------------------------
    def settle(self, min_frames=2):
        """Frames until the picture is fully accumulated (at least min_frames)."""
        for _ in range(min_frames):
            yield
        while not self.app.renderer.complete or self.app.sigma_moving:
            yield

    def picture(self):
        return self.app.renderer.picture()

    def press(self, *keys):
        self.snap.pressed = set(keys)
        yield

    def drag_path(self, button, moves):
        """Press at the window centre, move by each (dx, dy) in turn (one per frame), release."""
        W, H = self.app.fb
        x, y = W / 2.0, H / 2.0
        down = tuple(i == button for i in range(3))
        self.snap.mouse, self.snap.down, self.snap.clicked = (x, y), down, down
        yield
        for dx, dy in moves:
            x, y = x + dx, y + dy
            self.snap.mouse, self.snap.down = (x, y), down
            yield
        self.snap.down = (False, False, False)
        yield

    def hold(self, keys, seconds):
        t0 = time.perf_counter()
        frames = 0
        while time.perf_counter() - t0 < seconds or frames < 3:
            self.snap.keys = set(keys)
            frames += 1
            yield
        self.snap.keys = set()
        yield

    def material_cursor(self, prefer=(0.18, 0.12)):
        """A cursor offset (pixels from the centre, y up) over drawn material."""
        cam = self.app.camera
        W, H = cam.W, cam.H
        xs = np.arange(-W // 2 + 8, W // 2 - 8, 4) + 0.5
        ys = np.arange(-H // 2 + 8, H // 2 - 8, 4) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        q = np.zeros((gx.size, 4))
        q[:, 0] = gx.ravel() * cam.pixel
        q[:, 1] = gy.ravel() * cam.pixel
        row = removal_row(q, cam.table, cam.cut * cam.pixel, dtype=np.float64)
        good = np.nonzero(row == -1)[0]
        if good.size == 0:
            raise AssertionError("no material in view")
        tx, ty = prefer[0] * W, prefer[1] * H
        i = good[np.argmin((gx.ravel()[good] - tx) ** 2 + (gy.ravel()[good] - ty) ** 2)]
        return float(gx.ravel()[i]), float(gy.ravel()[i])

    def to_mouse(self, cursor):
        W, H = self.app.fb
        return (W / 2.0 + cursor[0], H / 2.0 - cursor[1])

    def changed_by(self, settings_fn):
        """Apply settings_fn, settle, and report how much the picture changed."""
        before = self.picture()
        settings_fn()
        yield from self.settle()
        after = self.picture()
        d = diff(before, after)
        return {"change": round(d, 2), "ok": d > 0.5}

    # ---- the steps ---------------------------------------------------------------------------
    def steps(self):
        a = self.app
        st = self
        start = {}

        def changed(fn, need=0.5):
            """Apply fn, settle, and check that the picture changed."""
            def f():
                before = st.picture()
                fn()
                yield from st.settle()
                d = diff(before, st.picture())
                return {"change": round(d, 2), "ok": d > need}
            return f

        def set_style(**kw):
            def fn():
                for k, v in kw.items():
                    setattr(a.style, k, v)
            return fn

        # ---- 3D ------------------------------------------------------------------------------
        def startup():
            yield from st.settle(3)
            pic = st.picture()
            start["pic"] = pic
            spread = float(pic.std())
            cam = a.camera
            return {"mode": a.mode, "style": a.style.style, "spread": round(spread, 1), "overview": cam.overview(),
                    "ok": a.mode == "3d" and cam.overview() and spread > 10 and cam.depth == 0}

        def drag_3d():
            before = st.picture()
            rho = a.camera.rho()
            yield from st.drag_path(0, [(20, 0)] * 10)
            yield from st.settle()
            d = diff(before, st.picture())
            drho = abs(a.camera.rho() - rho)
            return {"change": round(d, 2), "rho_change": drho, "ok": d > 0.5 and drho < 1e-9 and a.drag is None}

        def right_drag_3d():
            before = st.picture()
            yield from st.drag_path(1, [(-12, 8)] * 5)
            yield from st.settle()
            d = diff(before, st.picture())
            return {"change": round(d, 2), "overview": a.camera.overview(), "ok": d > 0.5}

        def wheel_30():
            yield from st.press("home")
            yield from st.settle()
            W, H = a.fb
            for _ in range(30):
                st.snap.mouse = (W / 2.0 + 0.06 * W, H / 2.0 + 0.04 * H)
                st.snap.wheel = 1.0
                yield
            yield from st.settle()
            cam = a.camera
            return {"depth": cam.depth, "overview": cam.overview(), "de": cam.de,
                    "ok": not cam.overview() and cam.depth >= 3 and cam.de > 0}

        def keys_then_home():
            before = st.picture()
            for k in ("w", "a", "q"):
                yield from st.hold({k}, 1.0)
            yield from st.settle()
            moved = diff(before, st.picture())
            yield from st.press("home")
            yield from st.settle()
            d = diff(start["pic"], st.picture())
            return {"moved": round(moved, 2), "diff_to_start": round(d, 3), "ok": moved > 0.5 and d < 2.0}

        def dive():
            W, H = a.fb
            frames = 0
            while a.camera.depth < 60 and frames < 400:
                st.snap.mouse = (W / 2.0 + 0.06 * W, H / 2.0 + 0.04 * H)
                st.snap.wheel = 2.0
                frames += 1
                yield
            yield from st.settle()
            cam = a.camera
            spread = float(st.picture().std())
            return {"depth": cam.depth, "frames": frames, "rows": len(cam.table.w), "spread": round(spread, 1),
                    "ok": cam.depth >= 60 and cam.de > 0 and not cam.overview() and spread > 5}

        def rule_step(**kw):
            def f():
                a.set_rule(**kw)
                yield from st.settle()
                cam = a.camera
                return {"rule": cam.rule.label(), "depth": cam.depth, "overview": cam.overview(), "ok": cam.de > 0}
            return f

        def restore_3d():
            a.style = StyleSettings()
            a.set_rule(m=2, max_level=None)
            a.camera.set_fov(50.0)
            a.camera.set_view_range(40.0)
            yield from st.press("home")
            yield from st.settle()
            d = diff(start["pic"], st.picture())
            return {"diff_to_start": round(d, 3), "ok": d < 2.0}

        # ---- 2D ------------------------------------------------------------------------------
        def to_2d():
            a.set_mode("2d")
            yield from st.settle(3)
            start["pic2"] = st.picture()
            spread = float(start["pic2"].std())
            return {"mode": a.mode, "spread": round(spread, 1), "ok": a.mode == "2d" and spread > 10}

        def wheel_100():
            cur = st.material_cursor()
            for _ in range(100):
                st.snap.mouse = st.to_mouse(cur)
                st.snap.wheel = 1.0
                yield
            yield from st.settle()
            tb = a.camera.table
            spread = float(st.picture().std())
            return {"depth": a.camera.depth, "covered": int(tb.covered), "rows": len(tb.w), "spread": round(spread, 1),
                    "ok": a.camera.depth >= 20 and tb.covered == 0 and spread > 5}

        def drag_2d(button, moves):
            def f():
                before = st.picture()
                yield from st.drag_path(button, moves)
                yield from st.settle()
                d = diff(before, st.picture())
                return {"change": round(d, 2), "ok": d > 0.5 and a.drag is None}
            return f

        def keys_2d():
            before = st.picture()
            yield from st.hold({"d"}, 0.0)
            yield from st.hold({"w"}, 0.0)
            yield from st.settle()
            d = diff(before, st.picture())
            return {"change": round(d, 2), "ok": d > 0.5}

        def go_home():
            yield from st.press("home")
            yield from st.settle()
            return {"depth": a.camera.depth, "ok": a.camera.depth == 0}

        def quality():
            yield from st.changed_by(lambda: setattr(a.style, "quality", 0.0))
            return {"supersample": a.style.supersample}    # the converged picture hardly changes

        def restore_2d():
            a.style = StyleSettings()
            a.set_rule(m=2, max_level=None)
            yield from st.press("home")
            yield from st.settle()
            d = diff(start["pic2"], st.picture())
            return {"diff_to_start": round(d, 3), "ok": d < 2.0}

        def deep_2d():
            t0 = time.perf_counter()
            frames = 0
            worst = 0.0
            while a.camera.depth < 300 and frames < 400:
                cur = st.material_cursor()
                st.snap.mouse = st.to_mouse(cur)
                st.snap.wheel = 30.0
                yield
                frames += 1
                worst = max(worst, a.frame_ms)
            yield from st.settle()
            tb = a.camera.table
            spread = float(st.picture().std())
            return {"depth": a.camera.depth, "frames": frames, "seconds": round(time.perf_counter() - t0, 1),
                    "worst_frame_ms": round(worst, 1), "rows": len(tb.w), "covered": int(tb.covered),
                    "spread": round(spread, 1), "ok": a.camera.depth >= 300 and tb.covered == 0 and spread > 5}

        # ---- window --------------------------------------------------------------------------
        def to_3d():
            a.set_mode("3d")
            yield from st.settle()
            d = diff(start["pic"], st.picture())
            return {"mode": a.mode, "diff_to_start": round(d, 3), "ok": a.mode == "3d" and d < 2.0}

        def toggle(key, attr):
            def f():
                was = getattr(a, attr)
                yield from st.press(key)
                yield
                now = getattr(a, attr)
                st.pending_shot = f"{st.current}_on"
                yield
                yield
                yield from st.press(key)
                yield
                return {"before": was, "after": now, "ok": now != was and getattr(a, attr) == was}
            return f

        def full_screen():
            W0, H0 = a.fb
            yield from st.press("f11")
            for _ in range(240):
                yield
                if a.fb != (W0, H0):
                    break
            big = a.fb
            yield from st.settle()
            st.pending_shot = "f11_on"
            yield
            yield from st.press("f11")
            for _ in range(240):
                yield
                if a.fb == (W0, H0):
                    break
            yield from st.settle()
            return {"full": big, "back": a.fb, "ok": big[0] * big[1] > W0 * H0 and a.fb == (W0, H0)
                    and not a.full_screen}

        def screenshot():
            a.last_screenshot = None
            yield from st.press("f12")
            for _ in range(3):
                yield
            p = a.last_screenshot
            ok = p is not None and Path(p).exists() and Path(p).stat().st_size > 1000
            return {"file": str(p) if p else None, "ok": ok}

        def resize():
            W0, H0 = a.fb
            hello_imgui.change_window_size((800, 500))
            for _ in range(240):
                yield
                if a.fb != (W0, H0):
                    break
            yield from st.settle()
            changed_size = a.fb != (W0, H0)
            spread = float(st.picture().std())
            return {"before": (W0, H0), "after": a.fb, "spread": round(spread, 1), "ok": changed_size and spread > 5}

        S = [("startup", startup), ("drag_3d", drag_3d), ("right_drag_3d", right_drag_3d), ("wheel_30", wheel_30)]
        for name in ("levels", "axes", "clay"):
            S.append((f"style_{name}_3d", changed(set_style(style=name))))
        S += [("palette_ember_3d", changed(set_style(style="levels", palette="ember"))),
              ("light_background_3d", changed(set_style(background="light"))),
              ("fog_off", changed(set_style(fog=False))),
              ("occlusion_0", changed(set_style(occlusion=0.0))),
              ("quality_low_3d", changed(set_style(quality=0.0), need=0.0)),
              ("fov_70", changed(lambda: a.camera.set_fov(70.0))),
              ("view_distance_100", changed(lambda: a.camera.set_view_range(100.0), need=0.0)),
              ("m_1_3d", rule_step(m=1)),
              ("m_3_3d", rule_step(m=3)),
              ("max_level_1_3d", rule_step(m=2, max_level=1)),
              ("max_level_3_3d", rule_step(max_level=3)),
              ("restore_3d", restore_3d),
              ("keys_then_home", keys_then_home),
              ("dive_3d", dive),
              ("home_3d", go_home),
              ("mode_2d", to_2d),
              ("wheel_100", wheel_100),
              ("drag_2d", drag_2d(0, [(12, 6)] * 10)),
              ("right_drag_2d", drag_2d(1, [(-15, 10)] * 5)),
              ("keys_2d", keys_2d)]
        for name in ("levels", "clay"):
            S.append((f"style_{name}_2d", changed(set_style(style=name))))
        S += [("palette_ember_2d", changed(set_style(style="levels", palette="ember"))),
              ("inverted", changed(set_style(inverted=True))),
              ("quality_low_2d", quality),
              ("home_2d", go_home),
              # (the background shows only outside the square, so this step runs in the whole view)
              ("light_background_2d", changed(set_style(background="light"))),
              ("default_style_2d", changed(lambda: setattr(a, "style", StyleSettings()))),
              ("m_1_2d", changed(lambda: a.set_rule(m=1))),
              ("m_2_2d", changed(lambda: a.set_rule(m=2))),
              ("max_level_3_2d", changed(lambda: a.set_rule(max_level=3))),
              ("max_level_1_2d", changed(lambda: a.set_rule(max_level=1))),
              ("restore_2d", restore_2d),
              ("deep_2d", deep_2d),
              ("home_again_2d", go_home),
              ("mode_3d", to_3d),
              ("tab", toggle("tab", "ui_visible")),
              ("f1", toggle("f1", "show_help")),
              ("m_key", toggle("m", "show_inset")),
              ("f11", full_screen),
              ("f12", screenshot),
              ("resize", resize)]
        return S
