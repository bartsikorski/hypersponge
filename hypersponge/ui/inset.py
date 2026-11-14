"""The inset map, bottom right (design sections 6.3 and 17.4).  Stage 1 draws the 2D map."""
from __future__ import annotations

from imgui_bundle import imgui

BOX = 200.0               # map size, interface pixels
MARGIN = 12.0             # from the bottom right corner
SCALE_2D = 120.0          # 2D: pixels per world unit, so that the square fills most of the map ...
FIT = 176.0               # ... unless the view is wider: then the view rectangle fits in this many
DOT = 3.0                 # a view rectangle smaller than this is drawn as a dot


def geometry_2d(center, height_world, aspect, box_min, box=BOX):
    """The unit square and the view rectangle in interface pixels, y down.

    Returns (square_min, square_max, view_min, view_max, dot): dot is None, or the centre of a
    view rectangle under DOT pixels."""
    scale = min(SCALE_2D, FIT / max(1.0, height_world * max(1.0, aspect)))
    mx = box_min[0] + box / 2.0
    my = box_min[1] + box / 2.0

    def to_px(x, y):
        return (mx + (x - 0.5) * scale, my - (y - 0.5) * scale)

    sq0 = to_px(0.0, 1.0)
    sq1 = to_px(1.0, 0.0)
    cx, cy = float(center[0]), float(center[1])
    hw, hh = 0.5 * height_world * aspect, 0.5 * height_world
    v0 = to_px(cx - hw, cy + hh)
    v1 = to_px(cx + hw, cy - hh)
    dot = to_px(cx, cy) if max(v1[0] - v0[0], v1[1] - v0[1]) < DOT else None
    return sq0, sq1, v0, v1, dot


def _col(r, g, b, a=1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r, g, b, a))


def draw_inset(app) -> None:
    if app.mode != "2d":
        return
    vp = imgui.get_main_viewport()
    x1 = vp.work_pos.x + vp.work_size.x - MARGIN
    y1 = vp.work_pos.y + vp.work_size.y - MARGIN
    x0, y0 = x1 - BOX, y1 - BOX
    if x0 < vp.work_pos.x or y0 < vp.work_pos.y:
        return
    dl = imgui.get_background_draw_list()
    V = imgui.ImVec2
    dl.add_rect_filled(V(x0, y0), V(x1, y1), _col(0, 0, 0, 0.35), 8.0)
    cam = app.camera
    W, H = app.fb
    sq0, sq1, v0, v1, dot = geometry_2d(cam.center_world(), cam.world_height(), W / max(1, H), (x0, y0))
    dl.push_clip_rect(V(x0 + 2, y0 + 2), V(x1 - 2, y1 - 2), True)
    dl.add_rect_filled(V(*sq0), V(*sq1), _col(0.97, 0.80, 0.26, 0.30))
    dl.add_rect(V(*sq0), V(*sq1), _col(0.85, 0.85, 0.88, 0.8), 0.0, 1.0)
    white = _col(1.0, 1.0, 1.0, 0.95)
    if dot is not None:
        dl.add_circle_filled(V(*dot), 3.0, white)
    else:
        dl.add_rect(V(*v0), V(*v1), white, 0.0, 2.0)
    dl.pop_clip_rect()
