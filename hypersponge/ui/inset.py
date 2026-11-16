"""The inset map, bottom right (design sections 6.3 and 17.4): the 2D square and view, and the
3D cube with the eye and its view direction."""
from __future__ import annotations

import itertools

import numpy as np
from imgui_bundle import imgui

from ..core.frame import slice_segments
from ..render.styles import AXIS_COL

BOX = 200.0               # map size, interface pixels
MARGIN = 12.0             # from the bottom right corner
SCALE_2D = 120.0          # 2D: pixels per world unit, so that the square fills most of the map ...
FIT = 176.0               # ... unless the view is wider: then the view rectangle fits in this many
DOT = 3.0                 # a view rectangle smaller than this is drawn as a dot
INSET_PROJ = np.array([[1.0, 0.0], [0.0, 1.0], [0.45, 0.30], [-0.38, 0.52]])   # x, y, z, w -> map
SCALE_ND = 70.0           # 3D and 4D: pixels per world unit
ARROW = 0.35              # length of the view line, world units


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


def cube_edges(n: int = 3) -> list:
    """The 12 edges of the unit cube (n = 3) or the 32 of the tesseract (n = 4), as 4-vector pairs."""
    edges = []
    for corner in itertools.product((0.0, 1.0), repeat=n):
        for i in range(n):
            if corner[i] == 0.0:
                a = np.zeros(4)
                a[:n] = corner
                b = a.copy()
                b[i] = 1.0
                edges.append((a, b))
    return edges


def nd_to_px(p, n, box_min, box=BOX) -> np.ndarray:
    """Map pixels (y down) of a point of the 3D cube or the tesseract."""
    c = np.full(4, 0.5)
    c[n:] = 0.0
    s = (np.asarray(p, float) - c) @ INSET_PROJ * SCALE_ND
    return np.array([box_min[0] + box / 2.0 + s[0], box_min[1] + box / 2.0 - s[1]])


def geometry_nd(eye, forward, n, box_min, box=BOX):
    """Map pixels (y down) of the cube's edges, the eye (kept on the map's border when it is
    outside) and the end of the view line."""
    mid = np.array([box_min[0] + box / 2.0, box_min[1] + box / 2.0])

    def to_px(p):
        return nd_to_px(p, n, box_min, box)
    edges = [(to_px(a), to_px(b)) for a, b in cube_edges(n)]
    e = to_px(eye)
    tip = to_px(np.asarray(eye, float) + ARROW * np.asarray(forward, float))
    lim = box / 2.0 - 6.0
    off = e - mid
    k = max(abs(off[0]), abs(off[1])) / lim
    if k > 1.0:                                   # outside the map: on its border, view line kept
        e = mid + off / k
        tip = e + (tip - to_px(eye))
    return edges, e, tip


def geometry_slice(ana, h, box_min, box=BOX):
    """4D: map pixels of the slice outline (design 17.4) and of the four axes from the corner
    (0, 0, 0, 0), as (outline segments, [(start, end, label position)] for x, y, z, w)."""
    outline = [(nd_to_px(a, 4, box_min, box), nd_to_px(b, 4, box_min, box)) for a, b in slice_segments(ana, h)]
    o = nd_to_px(np.zeros(4), 4, box_min, box)
    axes = []
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        end = nd_to_px(e, 4, box_min, box)
        d = (end - o) / max(1e-9, float(np.linalg.norm(end - o)))
        axes.append((o, end, end + d * 8.0))
    return outline, axes


def _col(r, g, b, a=1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r, g, b, a))


def draw_inset(app) -> None:
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
    if cam.mode != "2d":
        edges, e, tip = geometry_nd(cam.eye_world(), cam.F[:, 2], cam.n, (x0, y0))
        dl.push_clip_rect(V(x0 + 2, y0 + 2), V(x1 - 2, y1 - 2), True)
        grey = _col(0.8, 0.8, 0.85, 0.55 if cam.n == 3 else 0.35)
        for a, b in edges:
            dl.add_line(V(*a), V(*b), grey, 1.0)
        white = _col(1.0, 1.0, 1.0, 0.95)
        if cam.n == 4:
            outline, axes = geometry_slice(cam.ana(), cam.h(), (x0, y0))
            for (a, b, lab), col, name in zip(axes, AXIS_COL, "xyzw"):
                c = _col(*col, 0.9)
                dl.add_line(V(*a), V(*b), c, 1.5)
                ts = imgui.calc_text_size(name)
                dl.add_text(V(lab[0] - ts.x / 2, lab[1] - ts.y / 2), c, name)
            for a, b in outline:
                dl.add_line(V(*a), V(*b), white, 2.0)
        dl.add_line(V(*e), V(*tip), white, 2.0)
        dl.add_circle_filled(V(*e), 3.5, _col(0.97, 0.80, 0.26, 1.0))
        dl.pop_clip_rect()
        return
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
