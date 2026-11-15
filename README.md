# Hypersponge

A small desktop explorer for one fractal family in three modes: the Sierpinski carpet (2D), the
Menger sponge (3D) and the 4D Menger sponge seen through a moving 3D slice. All three zoom
without limit.

**Status: stage 2 of 4.** The 2D carpet and the 3D Menger sponge work: fly around and into the
sponge, zoom without limit in both, switch styles, rule and max level, make stills from the
command line. The 4D slice (stage 3) comes next; its design and a verified prototype are in this
repository.

![Fourteen stills rendered by the prototype](prototype/images/contact_sheet.png)

These stills come from the prototype: the real fragment shaders on a software OpenGL renderer,
driven by the prototype camera. Top row: the carpet, the carpet at depth 3^-150, Cantor dust,
the Menger sponge. The rest are the 3D sponge from inside and slices of the 4D sponge.

## Run it (Windows)

Needs any GPU with OpenGL 3.3. Double-click `Install Hypersponge.bat` once: it installs
[uv](https://docs.astral.sh/uv/) if needed, Python 3.12 and the packages, checks OpenGL, and puts
a Hypersponge icon on the Desktop and in the Start menu. Run it again after an update. From a
terminal in the repository folder:

```
run.bat                                     the window (same as: uv run hypersponge)
run.bat render -o sponge.png --dive 200 --cursor 30,-12
run.bat render -o deep.png --mode 2d --dive 1500 --cursor 137.5,-120.5
run.bat info                                OpenGL details and the shader checks
run.bat bench                               frame times on this GPU
uv run pytest -q                            the tests
```

In 3D: left-drag to turn the sponge (or look around once inside), the wheel to fly towards the
point under the cursor, right-drag to slide, W A S D R F to fly, Q E to roll. In 2D: drag or
W A S D to pan, the wheel to zoom about the cursor. Everywhere: Home to reset, Tab to hide the
panel, F1 for the controls, M for the inset map, F11 for full screen, F12 to save a PNG.

## What is here

| Path | Content |
| --- | --- |
| [`hypersponge/`](hypersponge) | The program: `core/` (rule, positions, level table, distance estimator, frames, cameras), `render/` (shaders, renderer, styles), `ui/` (window, panel, inset map, help, scripted session), `cli.py` |
| [`tests/`](tests) | pytest: the prototype checks, both shaders against numpy, golden pictures, the camera rules, the command line, and a scripted GUI session |
| [`docs/design-notes.md`](docs/design-notes.md) | The full design: the maths, rendering, camera and controls, interface, code layout, test plan, stages, and reference sections for whoever writes the code |
| [`prototype/`](prototype/README.md) | Reference code (numpy and GLSL) and seven check scripts. Everything the design claims as verified is checked here |

## The idea in short

- One rule defines every shape. A point of the unit cube is removed at level k when at least m
  of its coordinates lie in the middle third at that level. With m = 2 this gives the carpet in
  2D, the Menger sponge in 3D and its analogue in 4D.
- A position is stored as base-3 digits, so zoom has no limit. Each frame the CPU builds a short
  table of nearby walls relative to the camera, and a float32 shader does the rest.
- 3D and 4D use distance-estimator ray marching. 4D is shown as a 3D slice that can be turned
  into the fourth direction and slid along it.

## Plan

1. Skeleton, core and the 2D carpet.
2. The 3D sponge.
3. The 4D slice.
4. Extras, picked after stage 3.

Section 19 of the design notes holds a brief for each stage.

## Run the prototype checks

Needs [uv](https://docs.astral.sh/uv/). The shader check needs any OpenGL 3.3; on Linux without
a GPU, Mesa llvmpipe under `xvfb-run` is enough.

```
cd prototype
./run_all.sh
```

## Stack

Python 3.12, uv, numpy, moderngl, Dear ImGui (imgui-bundle), OpenGL 3.3. Windows first. The
same stack as [mandelscope](https://github.com/bartsikorski/mandelscope), without its deep-zoom
machinery.

## Licence

MIT, see [LICENSE](LICENSE).
