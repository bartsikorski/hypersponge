# Hypersponge prototype

Reference code written while preparing the design (12 and 14 November 2026). Everything here
runs and passes, and the design notes quote these results. It is the starting point for
`hypersponge/core`, `hypersponge/render/shaders` and the tests. It is not the program.

Run every check with `./run_all.sh` (needs uv; the shader check needs any OpenGL 3.3, and
`xvfb-run` on a machine without a display). A full run takes about four minutes.

## Reference code

| File | What it is | Design section |
| --- | --- | --- |
| `sponge_ref.py` | Exact ground truth with Fractions; `Pos`, a position as base-3 digits; `level_table`, the camera-relative table of walls; `normalize` and `normalize_2d`, the unit rules; `de_table`, `removal_row` and `march`, float32 numpy twins of the GLSL; `de_scalar` and `free_distance` for the camera | 3, 13 |
| `level_table_simple.py` | The first `level_table`: every level with big integers. Exact but slow when deep. Kept as the reference for the fast version | 13.3 |
| `shaders/common.glsl` | Table uniforms, `sort4`, `wallDist`, `boxDist`, `de`, `deInfo`, `removalRow` | 3.5, 14.2 |
| `shaders/march.frag` | 3D and 4D ray marcher with shading, occlusion, fog and five styles | 4.3, 14.3 |
| `shaders/carpet.frag` | 2D carpet with 3 x 3 supersampling and three styles | 4.2, 14.4 |
| `shade_ref.py` | numpy twin of the shading in `march.frag`; palettes and colour constants | 17 |
| `glsl_util.py` | Shader loading and uniform upload | 14.1 |
| `camera2d_ref.py` | `Camera2D`: zoom about the cursor, the anchor rule, the cover rule, pan | 15.1 |
| `camera_ref.py` | `CameraND`: regimes, pivots, the motion rule, wheel, drags, slice turns and slides, presets, scene scale; `slice_basis`, `look_frame`, `slice_segments` for the inset map | 15.2, 15.3, 17.4 |
| `gpu_render.py` | Stills through the real shaders with jittered accumulation; made `images/contact_sheet.png` | 16.5 |
| `render.py` | The first, slow CPU renderer; made `images/cpu_contact_sheet.png`. `check_camera.py` uses its helpers | 3.7 |

## Checks

| Script | What it checks |
| --- | --- |
| `check_estimator.py` | Kept-cell counts for all 9 rules; the estimator is a lower bound of the true distance for 10 rule and dimension cases |
| `check_table.py` | The float32 table against exact rationals at depths 3^0 to 3^-200, including positions beside huge walls; rescale and move are exact; rebase invariance; far-limit pruning |
| `check_levels.py` | The fast `level_table` equals `level_table_simple` on 1,584 positions; timing down to depth 3^-10000 |
| `check_camera.py` | Scalar functions equal the vector ones; ray timing; 1,500 random 4D moves never enter solid |
| `check_camera_rules.py` | Presets and frames; overview rotations keep rho and h; the scene follows the mouse; wheel in and out; offset slider; slice outlines; the Menger preset equals 3D mode ray for ray; a fuzz of 9,000 random inputs over every control |
| `check_camera2d.py` | Zoom keeps the point under the cursor; zoom in and out returns; the anchor rule never ends inside a hole down to depth 3^-400 |
| `check_shaders.py` | `march.frag` against the numpy marcher and shading on 5 scenes, every pixel; `carpet.frag` against exact base-3 membership on 8 views down to depth 3^-300 |

## Stills

```
xvfb-run -a uv run --with moderngl --with numpy --with pillow python gpu_render.py gpu_out
```

writes 14 stills and a contact sheet into `gpu_out/` (ignored by git).

## Porting notes

- `de_table` reports the outer cube's axis as `axis + 4`; the shader returns `axis`. Port it as `axis % 4`.
- `CameraND(3)` reuses the `menger` preset, which leaves the unused w coordinate of the eye at -0.25. It has no effect because the axis is switched off; the port should start 3D with w = 0.
- `CameraND.eye_world` is `Pos.to_floats` in the design.
- The checks print results and assert. The stages turn them into pytest files (design section 8).
