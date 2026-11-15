"""Command line (design section 18.1): gui (default), render, info."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

MODES = ("2d", "3d", "4d")
AVAILABLE_MODES = ("2d",)          # 3D arrives in stage 2, 4D in stage 3


class UsageError(Exception):
    pass


def _size(s: str):
    try:
        w, h = s.lower().split("x")
        w, h = int(w), int(h)
    except Exception:
        raise UsageError(f"size must look like 1280x720, not {s!r}")
    if not (16 <= w <= 16384 and 16 <= h <= 16384):
        raise UsageError(f"size {s} is out of range")
    return w, h


def _pair(s: str, what: str):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        raise UsageError(f"{what} must look like X,Y, not {s!r}")


def _check_mode(mode: str):
    if mode not in AVAILABLE_MODES:
        raise UsageError(f"mode {mode} is not built yet (this version has: {', '.join(AVAILABLE_MODES)})")


def make_camera(args, size):
    """The camera a render command describes."""
    from .core.camera import Camera2D
    from .core.rule import Rule
    _check_mode(args.mode)
    n = int(args.mode[0])
    m = args.m if args.m is not None else 2
    try:
        rule = Rule(n, m, args.max_level)
    except ValueError as e:
        raise UsageError(str(e))
    cam = Camera2D(rule, *size)
    center = _pair(args.center, "--center") if args.center else (0.5, 0.5)
    height = args.height if args.height else None
    if height is not None and not height > 0:
        raise UsageError("--height must be positive")
    cam.set_view(center, height or cam.world_height())
    cursor = _pair(args.cursor, "--cursor") if args.cursor else (0.0, 0.0)
    for _ in range(max(0, args.dive)):
        cam.zoom(1, cursor)
    return cam


def style_from_args(args):
    from .render.styles import PALETTES, STYLES, StyleSettings, styles_for
    if args.style not in STYLES:
        raise UsageError(f"unknown style {args.style!r}; styles: {', '.join(STYLES)}")
    if args.style not in styles_for(args.mode):
        raise UsageError(f"style {args.style} is not available in {args.mode}; there: {', '.join(styles_for(args.mode))}")
    if args.palette not in PALETTES:
        raise UsageError(f"unknown palette {args.palette!r}; palettes: {', '.join(PALETTES)}")
    return StyleSettings(style=args.style, palette=args.palette, background="light" if args.light else "dark",
                         inverted=bool(args.invert))


def cmd_render(args) -> int:
    from PIL import Image
    from .render.renderer import render_still
    size = _size(args.size)
    st = style_from_args(args)
    cam = make_camera(args, size)
    t0 = time.perf_counter()
    img = render_still(cam, st, samples=max(1, args.samples))
    Image.fromarray(img).save(args.output)
    print(f"{args.output}: {size[0]}x{size[1]}, {args.mode}, {cam.rule.name()}, depth 3^-{cam.depth}, "
          f"{time.perf_counter() - t0:.2f} s")
    return 0


def shader_check(ctx) -> list:
    """The GPU's 2D shader against the numpy twin on three views.  Returns (name, ok, detail)."""
    from .core.camera import Camera2D
    from .core.rule import Rule
    from .core.sdf import removal_row
    from .render.renderer import Renderer
    r = Renderer(ctx, dynamic=False)
    results = []
    views = [("whole carpet", Camera2D(Rule(2, 2), 320, 240), 0, (0.0, 0.0)),
             ("carpet at depth 3^-150", Camera2D(Rule(2, 2), 320, 240), 150, None),
             ("Cantor dust, depth 3^-30", Camera2D(Rule(2, 1), 320, 240), 30, None)]
    for name, cam, depth, _ in views:
        if depth:
            cam.pos.rescale(depth)
            cam.pos.move([0.37, -0.21, 0.0, 0.0])
            cam.height = 1.4
            cam.refresh()
        g = r.data_2d(cam)
        W, H = cam.W, cam.H
        ys, xs = np.mgrid[0:H, 0:W]
        q = np.zeros((H * W, 4), np.float32)
        q[:, 0] = ((xs + 0.5 - W / 2).astype(np.float32) * np.float32(cam.height / H)).reshape(-1)
        q[:, 1] = ((ys + 0.5 - H / 2).astype(np.float32) * np.float32(cam.height / H)).reshape(-1)
        want = removal_row(q, cam.table, np.float32(cam.height / H / 3)).reshape(H, W)
        agree = float((g == want).mean())
        results.append((name, agree > 0.999, f"{agree * 100:.2f}% of pixels agree"))
    return results


def cmd_info(args) -> int:
    from . import __version__
    from .render.context import describe, make_headless_context
    print(f"Hypersponge {__version__}, Python {sys.version.split()[0]}, numpy {np.__version__}")
    try:
        ctx = make_headless_context()
    except Exception as e:
        print(f"OpenGL: FAIL\n{e}")
        return 1
    print(f"OpenGL: {describe(ctx)}")
    ok = True
    for name, good, detail in shader_check(ctx):
        print(f"  2D shader, {name}: {'ok' if good else 'WRONG'} ({detail})")
        ok &= good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_gui(args) -> int:
    _check_mode(args.mode)
    from .ui.app import run_app
    return run_app(mode=args.mode, size=_size(args.size), selftest=args.selftest)


def build_parser():
    p = argparse.ArgumentParser(prog="hypersponge", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="open the window (the default)")
    g.add_argument("--mode", choices=MODES, default="2d")
    g.add_argument("--size", default="1600x900")
    g.add_argument("--selftest", metavar="REPORT.json", help="run the scripted session and write a report")
    r = sub.add_parser("render", help="render a still without a window")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--mode", choices=MODES, default="2d")
    r.add_argument("--m", type=int, default=None, help="rule m (default 2)")
    r.add_argument("--max-level", type=int, default=None)
    r.add_argument("--style", default="sponge")
    r.add_argument("--palette", default="pastel")
    r.add_argument("--light", action="store_true", help="light background")
    r.add_argument("--invert", action="store_true", help="2D: swap holes and material")
    r.add_argument("--size", default="1280x720")
    r.add_argument("--samples", type=int, default=16)
    r.add_argument("--dive", type=int, default=0, help="wheel notches towards --cursor before rendering")
    r.add_argument("--cursor", default=None, help="pixels from the centre, y up (default 0,0)")
    r.add_argument("--center", default=None, help="2D: view centre in world coordinates")
    r.add_argument("--height", type=float, default=None, help="2D: view height in world units")
    sub.add_parser("info", help="OpenGL details and a check of the shaders on this GPU")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["gui"] + argv
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        return {"gui": cmd_gui, "render": cmd_render, "info": cmd_info}[args.cmd](args)
    except UsageError as e:
        print(f"hypersponge: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"hypersponge: error: {e}", file=sys.stderr)
        return 1
