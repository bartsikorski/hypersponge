# Changelog

## 0.2.0 (stage 2, unreleased)

The 3D Menger sponge, with unlimited zoom. The program now starts in 3D.

- Fly around and into the sponge: left-drag turns it (or looks around once inside), right-drag
  slides, the wheel moves 20% of the way to the surface under the cursor, W A S D R F fly, Q E
  roll, Home returns to the overview. The camera never enters the solid.
- Styles Sponge, Levels, Axes and Clay with exact face normals, a camera light, ambient
  occlusion and fog; panel fields for fog, occlusion, field of view and view distance.
- Rule m = 1, 2, 3 and max levels in 3D; the camera resets if a change would put it in solid.
- 3D inset map (the cube, the eye and the view direction), 3D column in the controls window,
  "overview" or "immersed" in the status bar.
- Long-frame guard: on a slow GPU the picture is drawn in horizontal bands.
- One-click start on Windows: `Install Hypersponge.bat` sets up and puts a Hypersponge icon on the
  Desktop and in the Start menu (`tools/make_shortcuts.ps1`, icon from `tools/make_icon.py`).
- `hypersponge bench` prints frame times for 8 scenes at 1080p and 4K; `hypersponge info` also
  checks the 3D shader. On an RTX 5060 Ti the 1080p overview takes 1.3 ms.
- Tests: the 3D camera rules and a 10,200-input fuzz, the probe shader and the marcher against
  numpy, the shading against `tests/shade_ref.py`, golden pictures 5 to 8, 3D stills from the
  command line, and 3D steps in the scripted session (the wheel alone reaches depth 3^-100).

## 0.1.0 (stage 1, unreleased)

The first program: the 2D Sierpinski carpet with unlimited zoom.

- `hypersponge` (or `run.bat`) opens the window: drag or W A S D to pan, the wheel to zoom about
  the cursor, Home to reset. The panel has the Mode, Shape, Style, View and Help groups; the 3D
  and 4D buttons are shown disabled until stages 2 and 3.
- Rule m = 1 (Cantor dust) and m = 2 (carpet), max level 1 to 12 or unlimited.
- Styles Sponge, Levels (palettes Pastel, Ember, Ice) and Clay, dark or light background, and
  Inverted (holes and material swapped).
- Status bar with the depth as 3^-E, frames per second, render scale and table rows; it says
  when the whole view is inside one hole. Inset map of the square and the view (M). Controls
  window (F1), hide the panel (Tab), full screen (F11), save a PNG without the interface (F12).
- `hypersponge render` makes stills without a window, down to any depth (`--dive`, `--cursor`,
  `--center`, `--height`); `hypersponge info` checks the 2D shader against numpy on this GPU.
- Accumulation of 16 jittered samples while the view is still; dynamic resolution while moving.
- Tests: the prototype checks as pytest, golden pictures 1 to 4, the 2D identities, and a
  scripted GUI session (`hypersponge gui --selftest report.json`) that zooms to depth 3^-300.
  CI on Ubuntu with Mesa llvmpipe under Xvfb.
