"""Command line (design section 18.1): gui (default), render, info, bench."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

MODES = ("2d", "3d", "4d")
AVAILABLE_MODES = ("2d", "3d")    # 4D arrives in stage 3


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
    from .core.camera import Camera2D, CameraND
    from .core.rule import Rule
    _check_mode(args.mode)
    n = int(args.mode[0])
    m = args.m if args.m is not None else 2
    try:
        rule = Rule(n, m, args.max_level)
    except ValueError as e:
        raise UsageError(str(e))
    cursor = _pair(args.cursor, "--cursor") if args.cursor else (0.0, 0.0)
    if n == 3:
        if args.center or args.height:
            raise UsageError("--center and --height are for the 2D mode")
        cam = CameraND(rule, *size)
        for _ in range(max(0, args.dive)):
            cam.zoom(1, cursor)
        cam.update_sigma()
        return cam
    cam = Camera2D(rule, *size)
    center = _pair(args.center, "--center") if args.center else (0.5, 0.5)
    height = args.height if args.height else None
    if height is not None and not height > 0:
        raise UsageError("--height must be positive")
    cam.set_view(center, height or cam.world_height())
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


def march_check(renderer, cam, st=None):
    """march.frag in data mode against the numpy marcher (core/sdf.py) for every pixel of cam.
    Returns (hit agreement, row agreement on pixels both call hits) as fractions."""
    from .core.sdf import march
    from .render.styles import StyleSettings
    st = st or StyleSettings()
    g = renderer.data_3d(cam, st).reshape(-1, 4)
    W, H = cam.W, cam.H
    pix = cam.pix
    ys, xs = np.mgrid[0:H, 0:W]
    dc = np.stack([(xs + 0.5 - W / 2) * pix, (ys + 0.5 - H / 2) * pix, np.ones((H, W))], -1).reshape(-1, 3)
    dc = (dc / np.linalg.norm(dc, axis=1)[:, None]).astype(np.float32)
    D = (dc @ cam.F[:, :3].T.astype(np.float32)).astype(np.float32)
    t, hit, _ = march(cam.table, D, pix, cam.far, max_steps=st.max_steps, lod=st.lod)
    t = t.astype(np.float32)
    from .core.sdf import de_table
    _, row, _ = de_table(D * t[:, None], cam.table, lodcut=np.float32(st.lod) * t * np.float32(pix), want_info=True)
    gh = g[:, 1] > 0.5
    both = gh & hit
    rows = float((g[:, 2].astype(int) == row)[both].mean()) if both.any() else 1.0
    return float((gh == hit).mean()), rows


def shader_check_3d(ctx) -> list:
    """The GPU's 3D ray marcher against the numpy twin on two views.  Returns (name, ok, detail)."""
    from .core.camera import CameraND
    from .core.rule import Rule
    from .render.renderer import Renderer
    r = Renderer(ctx, dynamic=False)
    results = []
    deep = CameraND(Rule(3, 2), 160, 120)
    for _ in range(40):
        deep.wheel(1, (0.12, -0.05, 1.0))
    deep.update_sigma()
    for name, cam in (("sponge overview", CameraND(Rule(3, 2), 160, 120)), ("inside, depth 3^-%d" % deep.depth, deep)):
        hits, rows = march_check(r, cam)
        results.append((name, hits > 0.995 and rows > 0.99,
                        f"{hits * 100:.2f}% of pixels agree on the hit, {rows * 100:.2f}% on the level"))
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
    for name, good, detail in shader_check_3d(ctx):
        print(f"  3D shader, {name}: {'ok' if good else 'WRONG'} ({detail})")
        ok &= good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench_scenes(size):
    """The eight fixed scenes of `hypersponge bench`: (name, camera, style)."""
    from .core.camera import Camera2D, CameraND
    from .core.position import Pos
    from .core.rule import Rule
    from .render.styles import StyleSettings
    W, H = size

    def dive(cam, notches, cursor=(0.0, 0.0)):
        for _ in range(notches):
            cam.zoom(1, cursor)
        cam.update_sigma()
        return cam

    carpet = Camera2D(Rule(2, 2), W, H)
    deep2d = Camera2D(Rule(2, 2), W, H)
    p = Pos.from_floats([0.25, 0.75, 0.0, 0.0])
    p.rescale(150)
    deep2d.pos, deep2d.height = p, 1.4
    deep2d.refresh()
    q1 = StyleSettings(quality=1.0)
    return [
        ("2D carpet, whole", carpet, StyleSettings()),
        ("2D carpet, depth 3^-150, Levels", deep2d, StyleSettings(style="levels")),
        ("3D overview", CameraND(Rule(3, 2), W, H), StyleSettings()),
        ("3D overview, quality 1", CameraND(Rule(3, 2), W, H), q1),
        ("3D overview, m = 3, Clay", CameraND(Rule(3, 3), W, H), StyleSettings(style="clay")),
        ("3D, 40 wheel notches", dive(CameraND(Rule(3, 2), W, H), 40, (0.12 * H, -0.05 * H)), StyleSettings()),
        ("3D, 40 notches, quality 1", dive(CameraND(Rule(3, 2), W, H), 40, (0.12 * H, -0.05 * H)), q1),
        ("3D, depth 3^-100", dive(CameraND(Rule(3, 2), W, H), 489), StyleSettings(style="levels")),
    ]


def cmd_bench(args) -> int:
    """Frame times on this GPU (design 8): the GPU draw of one full-size frame, measured with
    ctx.finish(), and the CPU time of one camera update (table, unit, free distance)."""
    import statistics
    from .render.context import describe, make_headless_context
    from .render.renderer import Renderer
    sizes = [_size(args.size)] if args.size else [(1920, 1080), (3840, 2160)]
    ctx = make_headless_context()
    print(f"OpenGL: {describe(ctx)}")
    r = Renderer(ctx, dynamic=False)
    r.guard = False
    for size in sizes:
        print(f"\n{size[0]}x{size[1]}, one sample per pixel, median of {args.frames} frames")
        print(f"{'scene':34s} {'depth':>7s} {'rows':>4s} {'GPU ms':>8s} {'fps':>7s} {'camera ms':>9s}")
        for name, cam, st in bench_scenes(size):
            r.ensure_size(*size)
            draw = lambda: r._draw(cam, st, r.scene_fbo, size[0], size[1], (0.0, 0.0))   # noqa: E731
            r._timed(draw)
            ms = statistics.median(r._timed(draw) for _ in range(args.frames))
            t0 = time.perf_counter()
            for _ in range(5):
                cam.refresh()
                if cam.mode != "2d":
                    cam.free(cam.F[:, 2])
            cpu = (time.perf_counter() - t0) / 5 * 1000.0
            print(f"{name:34s} {'3^-' + str(cam.depth):>7s} {len(cam.table.w):4d} {ms:8.2f} {1000.0 / ms:7.0f} {cpu:9.2f}")
    return 0


def cmd_gui(args) -> int:
    _check_mode(args.mode)
    from .ui.app import run_app
    return run_app(mode=args.mode, size=_size(args.size), selftest=args.selftest)


def build_parser():
    p = argparse.ArgumentParser(prog="hypersponge", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="open the window (the default)")
    g.add_argument("--mode", choices=MODES, default="3d")
    g.add_argument("--size", default="1600x900")
    g.add_argument("--selftest", metavar="REPORT.json", help="run the scripted session and write a report")
    r = sub.add_parser("render", help="render a still without a window")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--mode", choices=MODES, default="3d")
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
    b = sub.add_parser("bench", help="frame times for 8 fixed scenes on this GPU")
    b.add_argument("--size", default=None, help="one size, for example 1920x1080 (default: 1080p and 4K)")
    b.add_argument("--frames", type=int, default=20, help="frames timed per scene (median)")
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
        return {"gui": cmd_gui, "render": cmd_render, "info": cmd_info, "bench": cmd_bench}[args.cmd](args)
    except UsageError as e:
        print(f"hypersponge: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"hypersponge: error: {e}", file=sys.stderr)
        return 1
