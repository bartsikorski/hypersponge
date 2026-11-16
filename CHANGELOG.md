# Changelog

## 0.2.0 (stage 2, unreleased)

The 3D Menger sponge. The program now starts in 3D.

- Fly around and into the sponge: left-drag turns it (overview) or looks around (inside), the
  wheel moves 20% of the way to the surface under the cursor, right-drag slides, W A S D R F fly,
  Q E roll, Home returns to the overview. The camera never enters the solid and zoom has no limit.
- Styles Sponge, Levels, Axes and Clay with fog and occlusion; field of view and view distance
  sliders; rule m = 1 to 3 and max level; 3D inset map with the eye and its view direction.
- Long-frame guard: on a slow GPU frames are drawn in bands so the window stays responsive.
- `hypersponge bench` prints frame times for 8 scenes at 1080p and 4K; `info` also checks the
  3D shader; `render` makes 3D stills (`--dive`, `--cursor`, `--fov`, `--no-fog`).
- `Install Hypersponge.bat`: one double-click sets everything up and puts a Hypersponge icon on
  the Desktop and in the Start menu.
- Tests: the 3D camera rules and fuzzing, frames, the marcher against numpy, golden scenes 5 to
  8, the 3D identity, the long-frame guard, and the 3D steps of the scripted session.

## 0.1.0 (stage 1)

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
