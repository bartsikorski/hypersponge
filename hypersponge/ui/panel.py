"""The side panel and the status bar (design sections 6.1, 6.2, 17.5, 17.6)."""
from __future__ import annotations

import math
import time

from imgui_bundle import hello_imgui, imgui

from ..core.frame import PRESETS
from ..core.rule import MAX_LEVEL_LIMIT, Rule
from ..render.styles import PALETTES, STYLE_NAMES, styles_for

MODE_ORDER = ("2d", "3d", "4d")
PRESET_LABELS = {"menger": "Menger", "dust": "Dust", "diagonal": "Diagonal", "face diagonal": "Face diagonal",
                 "cell diagonal": "Cell diagonal", "generic": "Generic"}
JOGS = (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2), ("xw", 0, 3), ("yw", 1, 3), ("zw", 2, 3))
JOG_SPEED = 0.5                     # plane jogs: degrees per pixel dragged
AUTO_PLANES = ("right", "up", "forward")
ORANGE = imgui.ImVec4(1.0, 0.62, 0.2, 1.0)
GREY = imgui.ImVec4(0.7, 0.72, 0.76, 1.0)


def em(v: float) -> float:
    return hello_imgui.em_size(v)


def draw_panel(app) -> None:
    from .app import MODE_LABELS
    imgui.set_next_window_pos(imgui.ImVec2(em(0.6), em(0.6)), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(em(21), 0), imgui.Cond_.first_use_ever)
    imgui.set_next_window_bg_alpha(0.86)
    flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_focus_on_appearing
    visible, _ = imgui.begin("Hypersponge", None, flags)
    if not visible:
        imgui.end()
        return
    st = app.style
    rule = app.rule

    # ---- Mode ---------------------------------------------------------------------------------
    imgui.separator_text("Mode")
    for i, mode in enumerate(MODE_ORDER):
        if i:
            imgui.same_line()
        active = mode == app.mode
        if active:
            imgui.push_style_color(imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_active))
        if imgui.button(MODE_LABELS[mode]):
            app.set_mode(mode)
        if active:
            imgui.pop_style_color()

    # ---- Shape --------------------------------------------------------------------------------
    imgui.separator_text("Shape")
    imgui.set_next_item_width(em(8))
    changed, m = imgui.slider_int("Rule m", rule.m, 1, rule.n)
    if changed:
        app.set_rule(m=m)
    imgui.text_wrapped(Rule(rule.n, app.rule.m).label())
    unlimited = rule.max_level is None
    ch, unlimited_new = imgui.checkbox("Unlimited levels", unlimited)
    if ch:
        app.set_rule(max_level=None if unlimited_new else min(MAX_LEVEL_LIMIT, max(1, app.camera.depth + 4)))
    if app.rule.max_level is not None:
        imgui.set_next_item_width(em(8))
        changed, lv = imgui.slider_int("Max level", app.rule.max_level, 1, MAX_LEVEL_LIMIT)
        if changed:
            app.set_rule(max_level=int(lv))

    # ---- Style --------------------------------------------------------------------------------
    imgui.separator_text("Style")
    names = styles_for(app.mode)
    cur = names.index(st.style) if st.style in names else 0
    imgui.set_next_item_width(em(8))
    changed, idx = imgui.combo("Style", cur, [STYLE_NAMES[s] for s in names])
    if changed:
        st.style = names[idx]
    if st.style == "levels":
        pal = list(PALETTES)
        imgui.set_next_item_width(em(8))
        changed, idx = imgui.combo("Palette", pal.index(st.palette), [p.capitalize() for p in pal])
        if changed:
            st.palette = pal[idx]
    if imgui.radio_button("Dark", st.background == "dark"):
        st.background = "dark"
    imgui.same_line()
    if imgui.radio_button("Light background", st.background == "light"):
        st.background = "light"
    if app.mode == "2d":
        _, st.inverted = imgui.checkbox("Inverted (swap holes and material)", st.inverted)
    else:
        _, st.fog = imgui.checkbox("Fog", st.fog)
        imgui.same_line()
        imgui.set_next_item_width(em(6))
        _, st.occlusion = imgui.slider_float("Occlusion", st.occlusion, 0.0, 1.0, "%.2f")
    imgui.set_next_item_width(em(8))
    _, st.quality = imgui.slider_float("Quality", st.quality, 0.0, 1.0, "%.2f")
    if app.mode == "2d":
        imgui.set_item_tooltip(f"{st.supersample} x {st.supersample} samples per pixel while moving")
    else:
        imgui.set_item_tooltip(f"detail down to {st.lod:.2f} pixel, at most {st.max_steps} steps per ray")

    # ---- View ---------------------------------------------------------------------------------
    imgui.separator_text("View")
    if imgui.button("Reset (Home)"):
        app.camera.reset()
    imgui.same_line()
    if imgui.button("Save picture (F12)"):
        app.screenshot_request = True
    if app.mode != "2d":
        cam = app.camera
        imgui.set_next_item_width(em(8))
        changed, fov = imgui.slider_float("Field of view", cam.fov, 30.0, 90.0, "%.0f deg")
        if changed:
            cam.set_fov(fov)
        imgui.set_next_item_width(em(8))
        changed, vr = imgui.slider_float("View distance", cam.view_range, 10.0, 300.0, "%.0f",
                                         imgui.SliderFlags_.logarithmic.value)
        if changed:
            cam.set_view_range(vr)
        imgui.set_item_tooltip("how far you see, in units of the scene scale")
    _, app.show_inset = imgui.checkbox("Inset map (M)", app.show_inset)

    # ---- 4D slice -----------------------------------------------------------------------------
    if app.mode == "4d":
        draw_slice_group(app)

    # ---- Help ---------------------------------------------------------------------------------
    imgui.separator_text("Help")
    if imgui.button("Controls (F1)"):
        app.show_help = not app.show_help
    imgui.same_line()
    imgui.text_disabled("Tab hides this panel")
    imgui.end()


def draw_slice_group(app) -> None:
    """Presets, offset, plane jogs, automatic motion and read-outs (design 6.1, 17.5)."""
    cam = app.camera
    key0 = cam.key()
    imgui.separator_text("4D slice")
    for i, name in enumerate(PRESETS):
        if i % 3:
            imgui.same_line()
        active = name == cam.preset_name
        if active:
            imgui.push_style_color(imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_active))
        if imgui.button(PRESET_LABELS[name]):
            cam.preset(name)
        if active:
            imgui.pop_style_color()
        imgui.set_item_tooltip("this slice; Home comes back to it")
    H = cam.offset_range()
    imgui.begin_disabled(not cam.overview())
    imgui.set_next_item_width(em(8))
    changed, h = imgui.slider_float("Offset", cam.h(), -H, H, "%.3f")
    if changed:
        cam.set_offset(h)
    imgui.end_disabled()
    imgui.set_item_tooltip("where the slice cuts the cube" if cam.overview()
                           else "only from outside the cube; inside, use Shift + wheel")
    imgui.text("Turn in plane")
    for i, (label, a, b) in enumerate(JOGS):
        if i % 3:
            imgui.same_line()
        imgui.set_next_item_width(em(4.2))
        changed, deg = imgui.drag_float(f"##jog{label}", 0.0, JOG_SPEED, 0.0, 0.0, f"{label} %.0f")
        if changed and deg:
            cam.rotate_object(a, b, math.radians(deg))
        imgui.set_item_tooltip(f"drag to turn the object in its {label} plane")
    _, cam.auto_slide = imgui.checkbox("Auto slide", cam.auto_slide)
    imgui.same_line()
    _, cam.auto_turn = imgui.checkbox("Auto turn", cam.auto_turn)
    imgui.same_line()
    imgui.set_next_item_width(em(5.5))
    _, cam.auto_plane = imgui.combo("##autoplane", cam.auto_plane, list(AUTO_PLANES))
    imgui.set_item_tooltip("which way auto turn tips into the fourth direction")
    a = cam.ana()
    imgui.text_disabled(f"ana ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}, {a[3]:.3f})   h {cam.h():.3f}")
    if cam.key() != key0:
        app.sigma_moving = True


def depth_text(E: int) -> str:
    """'3^-41 (10^-19.6)': the power of ten is E * log10(3)."""
    if E == 0:
        return "3^0"
    return f"3^-{E} (10^-{E * math.log10(3.0):.1f})"


def status_parts(app) -> list:
    """The status line as (text, colour or None) parts (design 17.6)."""
    from .app import MODE_LABELS
    cam = app.camera
    tb = cam.table
    r = app.rule
    rule = f"{r.name()} (m = {r.m}" + (f", max level {r.max_level})" if r.max_level else ")")
    parts = [(MODE_LABELS[app.mode], None), (rule, None), (f"depth {depth_text(cam.depth)}", None)]
    if cam.mode != "2d":
        parts.append(("overview" if cam.overview() else "immersed", None))
    parts += [(f"{app.fps:.0f} fps", None), (f"scale {app.renderer.scale:.2f}" if app.renderer else "scale -", None),
             (f"rows {len(tb.w)}", None)]
    if app.renderer is not None and not app.renderer.complete:
        parts.append((f"sample {app.renderer.samples}/{app.renderer.max_samples}", GREY))
    if app.renderer is not None and app.renderer.bands > 1:
        parts.append((f"slow frames: {app.renderer.bands} bands", GREY))
    if tb.dropped:
        parts.append((f"rows dropped: {tb.dropped}", ORANGE))
    if getattr(tb, "covered", 0):
        parts.append((f"inside a level {tb.covered} hole", ORANGE))
    text, until = app.notice
    if text and time.perf_counter() < until:
        parts.append((text, ORANGE if text.startswith(("internal error", "could not")) else GREY))
    return parts


def draw_status(app) -> None:
    vp = imgui.get_main_viewport()
    pad = em(0.4)
    imgui.set_next_window_pos(imgui.ImVec2(vp.work_pos.x + pad, vp.work_pos.y + vp.work_size.y - pad),
                              imgui.Cond_.always, imgui.ImVec2(0.0, 1.0))
    imgui.set_next_window_bg_alpha(0.55)
    flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize
             | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_focus_on_appearing
             | imgui.WindowFlags_.no_nav | imgui.WindowFlags_.no_move)
    imgui.begin("##status", None, flags)
    if app.hint:
        imgui.text_colored(GREY, "Drag to pan, wheel to zoom, F1 for help" if app.mode == "2d"
                           else "Drag to turn, wheel to zoom, F1 for help")
    for i, (text, col) in enumerate(status_parts(app)):
        if i:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
        if col is None:
            imgui.text(text)
        else:
            imgui.text_colored(col, text)
    imgui.end()
