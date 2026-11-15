"""The window: hello_imgui runner, input to camera, frame loop (design sections 15.4, 16.4)."""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import moderngl
from imgui_bundle import hello_imgui, imgui

from ..core.camera import Camera2D, CameraND
from ..core.rule import Rule
from ..render.context import GL_REQUIRE, describe
from ..render.renderer import Renderer
from ..render.styles import StyleSettings, styles_for

MODE_LABELS = {"2d": "2D carpet", "3d": "3D sponge", "4d": "4D sponge"}
AVAILABLE = ("2d", "3d")           # 4D arrives in stage 3
ROLL_SPEED = math.radians(90.0)    # Q and E, radians per second


@dataclass
class InputSnapshot:
    """What the explorer needs from the input devices for one frame."""
    mouse: tuple = (0.0, 0.0)                    # framebuffer pixels, y down
    down: tuple = (False, False, False)
    clicked: tuple = (False, False, False)
    wheel: float = 0.0
    keys: set = field(default_factory=set)       # held keys (names)
    pressed: set = field(default_factory=set)    # keys pressed this frame
    shift: bool = False
    ui_mouse: bool = False                       # the mouse is over the interface
    typing: bool = False                         # a text field has the keyboard


KEYMAP = {"w": imgui.Key.w, "a": imgui.Key.a, "s": imgui.Key.s, "d": imgui.Key.d, "r": imgui.Key.r,
          "f": imgui.Key.f, "q": imgui.Key.q, "e": imgui.Key.e, "m": imgui.Key.m, "home": imgui.Key.home,
          "tab": imgui.Key.tab, "f1": imgui.Key.f1, "f11": imgui.Key.f11, "f12": imgui.Key.f12,
          "comma": imgui.Key.comma, "period": imgui.Key.period,
          "1": imgui.Key._1, "2": imgui.Key._2, "3": imgui.Key._3, "4": imgui.Key._4, "5": imgui.Key._5,
          "6": imgui.Key._6}


def read_input(fb_scale) -> InputSnapshot:
    io = imgui.get_io()
    sx, sy = fb_scale
    mp = io.mouse_pos
    snap = InputSnapshot()
    snap.mouse = (mp.x * sx, mp.y * sy)
    snap.down = tuple(bool(io.mouse_down[i]) for i in range(3))
    snap.clicked = tuple(bool(imgui.is_mouse_clicked(i)) for i in range(3))
    snap.wheel = float(io.mouse_wheel)
    snap.keys = {n for n, k in KEYMAP.items() if imgui.is_key_down(k)}
    snap.pressed = {n for n, k in KEYMAP.items() if imgui.is_key_pressed(k, False)}
    snap.shift = bool(io.key_shift)
    snap.ui_mouse = bool(io.want_capture_mouse)
    snap.typing = bool(io.want_text_input)
    return snap


class App:
    def __init__(self, mode: str = "3d", size=(1600, 900), selftest: str | None = None):
        self.mode = mode
        self.window_size = tuple(size)
        self.style = StyleSettings()
        self.cams = {"2d": Camera2D(Rule(2, 2), *size), "3d": CameraND(Rule(3, 2), *size)}
        self.ui_visible = True
        self.show_help = False
        self.show_inset = True
        self.full_screen = False
        self.fb = (0, 0)
        self.fb_scale = (1.0, 1.0)
        self.fps = 60.0
        self.frame_ms = 0.0
        self.longest_frame_ms = 0.0
        self.last_time = time.perf_counter()
        self.drag = None                 # {"button": b, "last": (x, y), "grab": g} while a drag lasts
        self.hint = True                 # first-run hint until the first drag or wheel
        self.notice = ("", 0.0)          # (text, shown until)
        self.screenshot_dir = Path.cwd()
        self.screenshot_request = False
        self.last_screenshot = None
        self.errors = []
        self.ctx = self.renderer = None
        self.selftest = None
        if selftest:
            from .selftest import SelfTest
            self.selftest = SelfTest(self, selftest)

    # ---- state -------------------------------------------------------------------------------
    @property
    def camera(self):
        return self.cams[self.mode]

    @property
    def rule(self) -> Rule:
        return self.camera.rule

    def set_rule(self, m: int | None = None, max_level="keep") -> None:
        r = self.rule
        new = Rule(r.n, r.m if m is None else int(m), r.max_level if max_level == "keep" else max_level)
        if new != r:
            self.camera.set_rule(new)

    def set_mode(self, mode: str) -> None:
        if mode in AVAILABLE and mode in self.cams:
            self.mode = mode
            if self.style.style not in styles_for(mode):
                self.style.style = "sponge"

    def notify(self, text: str, seconds: float = 4.0) -> None:
        self.notice = (text, time.perf_counter() + seconds)

    def internal_error(self, where: str) -> None:
        """Log an unexpected exception once, tell the user briefly, keep running."""
        tb = traceback.format_exc()
        self.errors.append((where, tb[-3000:]))
        print(f"[hypersponge] error in {where}:\n{tb}", flush=True)
        self.notify(f"internal error in {where} (details in the console)")

    # ---- hello_imgui callbacks ---------------------------------------------------------------
    def post_init(self):
        self.ctx = moderngl.create_context(require=GL_REQUIRE)
        self.renderer = Renderer(self.ctx)
        print(f"[hypersponge] OpenGL: {describe(self.ctx)}", flush=True)

    def setup_imgui_config(self):
        io = imgui.get_io()
        io.config_flags &= ~imgui.ConfigFlags_.nav_enable_keyboard.value
        io.config_windows_move_from_title_bar_only = True
        io.config_error_recovery = True
        io.config_error_recovery_enable_assert = False
        io.config_error_recovery_enable_tooltip = False
        io.config_error_recovery_enable_debug_log = True

    def custom_background(self):
        try:
            self._frame()
        except Exception:
            self.internal_error("render")
        finally:
            if self.ctx is not None:
                self.ctx.screen.use()

    def show_gui(self):
        try:
            from . import help as help_window
            from . import inset, panel
            if self.ui_visible:
                panel.draw_panel(self)
            if self.show_inset:
                inset.draw_inset(self)
            panel.draw_status(self)
            if self.show_help:
                help_window.draw_help(self)
        except Exception:
            self.internal_error("interface")

    def before_swap(self):
        if self.selftest is not None:
            self.selftest.before_swap()

    # ---- frame -------------------------------------------------------------------------------
    def _frame(self):
        io = imgui.get_io()
        sc = io.display_framebuffer_scale
        self.fb_scale = (sc.x or 1.0, sc.y or 1.0)
        W = int(io.display_size.x * self.fb_scale[0])
        H = int(io.display_size.y * self.fb_scale[1])
        if W < 16 or H < 16:
            return
        self.fb = (W, H)
        for cam in self.cams.values():
            cam.resize(W, H)
        now = time.perf_counter()
        dt = min(0.1, now - self.last_time)
        self.last_time = now
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)
        inp = self.selftest.input() if self.selftest else read_input(self.fb_scale)
        self.handle_input(inp, dt)
        if self.mode != "2d":
            self.camera.update_sigma(dt)
        self.renderer.frame(self.camera, self.style)
        self.renderer.present(self.ctx.screen, (0, 0, W, H))
        if self.screenshot_request:
            self.screenshot_request = False
            self.save_screenshot()
        self.frame_ms = (time.perf_counter() - now) * 1000.0
        self.longest_frame_ms = max(self.longest_frame_ms, self.frame_ms)
        if self.selftest is not None:
            self.selftest.after_frame()
        else:
            # nothing left to draw and no key held: let hello_imgui wait for input
            quiet = self.renderer.complete and not inp.keys and self.drag is None
            try:
                hello_imgui.get_runner_params().fps_idling.enable_idling = quiet
            except Exception:
                pass

    def handle_input(self, inp: InputSnapshot, dt: float) -> None:
        cam = self.camera
        W, H = self.fb
        if not inp.typing:
            p = inp.pressed
            if "tab" in p:
                self.ui_visible = not self.ui_visible
            if "f1" in p:
                self.show_help = not self.show_help
            if "m" in p:
                self.show_inset = not self.show_inset
            if "f11" in p:
                self.toggle_full_screen()
            if "f12" in p:
                self.screenshot_request = True
            if "home" in p:
                cam.reset()
        mx, my = inp.mouse
        cursor = (mx - W / 2.0, H / 2.0 - my)
        flat = self.mode == "2d"
        if self.drag is None and not inp.ui_mouse:
            for b in (0, 1):
                if inp.clicked[b]:
                    self.drag = {"button": b, "last": (mx, my)}
                    if b == 1 and not flat:          # right-drag: the point under the cursor follows
                        self.drag["grab"] = cam.grab(cursor)
                    break
        if self.drag is not None:
            b = self.drag["button"]
            if inp.down[b]:
                lx, ly = self.drag["last"]
                dx, dy = mx - lx, my - ly
                if dx or dy:
                    if flat:
                        cam.pan(dx, -dy)
                    elif b == 0:
                        cam.drag_rotate(dx, -dy)
                    else:
                        cam.pan(dx, -dy, self.drag["grab"])
                    self.hint = False
                self.drag["last"] = (mx, my)
            else:
                self.drag = None
        if inp.wheel and not inp.ui_mouse:
            cam.zoom(inp.wheel, cursor)
            self.hint = False
        if not inp.typing:
            k = inp.keys
            vx = ("d" in k) - ("a" in k)
            vy = ("w" in k) - ("s" in k)
            if flat:
                if vx or vy:
                    cam.fly(vx, vy, dt, inp.shift)
            else:
                vz, vy = vy, ("r" in k) - ("f" in k)
                if vx or vy or vz:
                    cam.fly((vx, vy, vz), dt, inp.shift)
                roll = ("q" in k) - ("e" in k)
                if roll:
                    cam.roll(roll * ROLL_SPEED * dt)

    def toggle_full_screen(self) -> None:
        if not self.full_screen:
            self.window_size = tuple(int(v) for v in imgui.get_io().display_size)
            hello_imgui.use_window_full_monitor_work_area()
        else:
            hello_imgui.change_window_size(self.window_size)
        self.full_screen = not self.full_screen

    def save_screenshot(self) -> Path | None:
        from PIL import Image
        name = datetime.now().strftime("hypersponge_%Y%m%d_%H%M%S.png")
        path = Path(self.screenshot_dir) / name
        try:
            Image.fromarray(self.renderer.picture()).save(path)
        except Exception as e:
            self.notify(f"could not save the picture: {e}")
            return None
        self.last_screenshot = path
        self.notify(f"saved {path}")
        return path


def run_app(mode: str = "3d", size=(1600, 900), selftest: str | None = None) -> int:
    app = App(mode, size, selftest)
    rp = hello_imgui.RunnerParams()
    rp.app_window_params.window_title = "Hypersponge"
    rp.app_window_params.window_geometry.size = tuple(size)
    rp.app_window_params.restore_previous_geometry = not selftest
    rp.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    gl = hello_imgui.OpenGlOptions()
    gl.major_version = 3
    gl.minor_version = 3
    gl.use_core_profile = True
    gl.glsl_version = "#version 330"
    rp.renderer_backend_options.open_gl_options = gl
    rp.fps_idling.enable_idling = False
    rp.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    rp.ini_filename = "hypersponge/imgui.ini"
    if selftest:
        rp.ini_disable = True
    rp.callbacks.post_init = app.post_init
    rp.callbacks.custom_background = app.custom_background
    rp.callbacks.show_gui = app.show_gui
    rp.callbacks.before_swap = app.before_swap
    rp.callbacks.setup_imgui_config = app.setup_imgui_config
    hello_imgui.run(rp)
    if selftest:
        try:
            rep = json.loads(Path(selftest).read_text(encoding="utf-8"))
        except Exception:
            return 1
        return 0 if rep.get("passed") else 1
    return 0
