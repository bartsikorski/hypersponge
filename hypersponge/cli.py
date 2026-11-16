"""Command line (design section 18.1): gui (default), render, info, bench."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

MODES = ("2d", "3d", "4d")
AVAILABLE_MODES = MODES


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
    from .core.camera import FOV, Camera2D, CameraND
    from .core.rule import Rule
    _check_mode(args.mode)
    n = int(args.mode[0])
    m = args.m if args.m is not None else 2
    try:
        rule = Rule(n, m, args.max_level)
    except ValueError as e:
        raise UsageError(str(e))
    cursor = _pair(args.cursor, "--cursor") if args.cursor else (0.0, 0.0)
    if n == 2:
        cam = Camera2D(rule, *size)
        center = _pair(args.center, "--center") if args.center else (0.5, 0.5)
        height = args.height if args.height else None
        if height is not None and not height > 0:
            raise UsageError("--height must be positive")
        cam.set_view(center, height or cam.world_height())
        for _ in range(max(0, args.dive)):
            cam.zoom(1, cursor)
        return cam
    if args.center or args.height:
        raise UsageError("--center and --height are for the 2D mode")
    preset = getattr(args, "preset", None)
    if preset is not None and n != 4:
        raise UsageError("--preset is for the 4D mode")
    fov = args.fov if args.fov is not None else FOV
    if not 10.0 <= fov <= 150.0:
        raise UsageError("--fov must be between 10 and 150 degrees")
    cam = CameraND(rule, *size, fov=fov)
    if n == 4:
        from .core.frame import PRESETS
        name = (preset or "diagonal").replace("-", " ").replace("_", " ").lower()
        if name not in PRESETS:
            raise UsageError(f"unknown preset {preset!r}; presets: {', '.join(p.replace(' ', '-') for p in PRESETS)}")
        cam.preset(name)
    d = cam.cursor_dir(cursor)
    for _ in range(max(0, args.dive)):
        cam.wheel(1, d)
    cam.update_sigma()
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
                         inverted=bool(args.invert), fog=not args.no_fog)


def cmd_render(args) -> int:
    from PIL import Image
    from .render.renderer import render_still
    size = _size(args.size)
    st = style_from_args(args)
    cam = make_camera(args, size)
    t0 = time.perf_counter()
    img = render_still(cam, st, samples=max(1, args.samples))
    Image.fromarray(img).save(args.output)
    where = "" if cam.mode == "2d" else (", overview" if cam.overview() else ", immersed")
    print(f"{args.output}: {size[0]}x{size[1]}, {args.mode}, {cam.rule.name()}, depth 3^-{cam.depth}{where}, "
          f"{time.perf_counter() - t0:.2f} s")
    return 0


def shader_check(ctx) -> list:
    """The GPU's shaders against the numpy twins.  Returns (name, ok, detail)."""
    from .core.camera import Camera2D, CameraND
    from .core.rule import Rule
    from .core.sdf import march, removal_row
    from .render.renderer import Renderer
    from .render.styles import StyleSettings
    r = Renderer(ctx, dynamic=False)
    results = []
    views = [("whole carpet", Camera2D(Rule(2, 2), 320, 240), 0),
             ("carpet at depth 3^-150", Camera2D(Rule(2, 2), 320, 240), 150),
             ("Cantor dust, depth 3^-30", Camera2D(Rule(2, 1), 320, 240), 30)]
    for name, cam, depth in views:
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
        results.append(("2D shader, " + name, agree > 0.999, f"{agree * 100:.2f}% of pixels agree"))
    overview = CameraND(Rule(3, 2), 160, 120)
    deep = CameraND(Rule(3, 2), 160, 120)
    oblique = CameraND(Rule(4, 2), 160, 120)
    oblique.preset("generic")
    for cam in (deep, oblique):
        for _ in range(120):
            cam.wheel(1, (0.12, -0.05, 1.0))
        cam.update_sigma()
    for name, cam in (("3D shader, sponge overview", overview), (f"3D shader, sponge at depth 3^-{deep.depth}", deep),
                      (f"4D shader, oblique slice at depth 3^-{oblique.depth}", oblique)):
        st = StyleSettings()
        data = r.data_nd(cam, st).reshape(-1, 4)
        W, H = cam.W, cam.H
        ys, xs = np.mgrid[0:H, 0:W]
        dc = np.stack([(xs + 0.5 - W / 2) * cam.pix, (ys + 0.5 - H / 2) * cam.pix, np.ones((H, W))], -1)
        dc = dc.reshape(-1, 3).astype(np.float32)
        dc /= np.linalg.norm(dc, axis=1)[:, None]
        D = (dc @ cam.F[:, :3].T.astype(np.float32)).astype(np.float32)
        t, hit, n_steps = march(cam.table, D, cam.pix, cam.far, max_steps=st.max_steps, lod=st.lod)
        hit = hit | ((n_steps >= st.max_steps) & (t <= cam.far))
        gh = data[:, 1] > 0.5
        both = gh & hit
        rel = np.abs(data[:, 0] - t)[both] / np.maximum(t[both], 1e-30)
        agree = float((gh == hit).mean())
        close = float((rel < 1e-3).mean()) if both.any() else 1.0
        results.append((name, agree > 0.999 and close > 0.999,
                        f"{agree * 100:.2f}% of rays agree on hit, {close * 100:.2f}% of hit distances within 0.1%"))
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
        print(f"  {name}: {'ok' if good else 'WRONG'} ({detail})")
        ok &= good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


BENCH_SCENES = [                    # (name, mode, rule m, style, quality, wheel notches)
    ("2D carpet", "2d", 2, "sponge", 0.5, 0),
    ("2D carpet, 3^-150", "2d", 2, "levels", 0.5, 150),
    ("3D overview", "3d", 2, "sponge", 0.5, 0),
    ("3D overview, quality 1", "3d", 2, "sponge", 1.0, 0),
    ("3D overview, Cantor dust", "3d", 1, "clay", 0.5, 0),
    ("3D immersed, 40 notches", "3d", 2, "sponge", 0.5, 40),
    ("3D immersed, 300 notches", "3d", 2, "levels", 0.5, 300),
    ("3D immersed, 300 notches, q 1", "3d", 2, "axes", 1.0, 300),
    ("4D diagonal slice, overview", "4d", 2, "sponge", 0.5, 0),
    ("4D generic slice, 60 notches", "4d", 2, "wdepth", 0.5, 60),
]


def bench_camera(mode, m, notches, size):
    from .core.camera import Camera2D, CameraND
    from .core.rule import Rule
    if mode == "2d":
        cam = Camera2D(Rule(2, m), *size)
        if notches:
            cam.pos.rescale(notches)
            cam.pos.move([0.37, -0.21, 0.0, 0.0])
            cam.height = 1.4
            cam.refresh()
        return cam
    cam = CameraND(Rule(int(mode[0]), m), *size)
    if mode == "4d":
        cam.preset("diagonal" if not notches else "generic")
    for _ in range(notches):
        cam.wheel(1, (0.12, -0.05, 1.0))
    cam.update_sigma()
    return cam


def cmd_bench(args) -> int:
    """Frame times on this GPU: a moving frame (one full-size sample) and a still sample."""
    from .render.context import describe, make_headless_context
    from .render.renderer import Renderer
    from .render.styles import StyleSettings
    sizes = [_size(args.size)] if args.size else [(1920, 1080), (3840, 2160)]
    ctx = make_headless_context()
    print(f"OpenGL: {describe(ctx)}")
    r = Renderer(ctx, dynamic=False)
    print(f"{'scene':32s} {'size':>10s} {'depth':>6s} {'frame ms':>9s} {'fps':>6s}")
    for size in sizes:
        for name, mode, m, style, q, notches in BENCH_SCENES:
            cam = bench_camera(mode, m, notches, size)
            st = StyleSettings(style=style, quality=q)
            r.ensure_size(*size)
            times = []
            for i in range(args.frames + 2):
                t0 = time.perf_counter()
                r._draw(cam, st, r.scene_fbo, size[0], size[1], (0.0, 0.0))
                ctx.finish()
                if i >= 2:                       # the first two include shader warm-up
                    times.append((time.perf_counter() - t0) * 1000.0)
            ms = float(np.median(times))
            print(f"{name:32s} {size[0]:>5d}x{size[1]:<4d} {cam.depth:>6d} {ms:>9.1f} {1000.0 / ms:>6.0f}", flush=True)
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
    r.add_argument("--m", type=int, default=None, help="rule m, 1 to n (default 2)")
    r.add_argument("--max-level", type=int, default=None)
    r.add_argument("--style", default="sponge")
    r.add_argument("--palette", default="pastel")
    r.add_argument("--light", action="store_true", help="light background")
    r.add_argument("--invert", action="store_true", help="2D: swap holes and material")
    r.add_argument("--no-fog", action="store_true", help="3D and 4D: no fog")
    r.add_argument("--size", default="1280x720")
    r.add_argument("--samples", type=int, default=16)
    r.add_argument("--dive", type=int, default=0, help="wheel notches towards --cursor before rendering")
    r.add_argument("--cursor", default=None, help="pixels from the centre, y up (default 0,0)")
    r.add_argument("--fov", type=float, default=None, help="3D and 4D: field of view in degrees (default 50)")
    r.add_argument("--preset", default=None,
                   help="4D: menger, dust, diagonal (default), face-diagonal, cell-diagonal or generic")
    r.add_argument("--center", default=None, help="2D: view centre in world coordinates")
    r.add_argument("--height", type=float, default=None, help="2D: view height in world units")
    sub.add_parser("info", help="OpenGL details and a check of the shaders on this GPU")
    b = sub.add_parser("bench", help="frame times for 10 fixed scenes (1080p and 4K unless --size)")
    b.add_argument("--size", default=None)
    b.add_argument("--frames", type=int, default=10, help="timed frames per scene")
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
