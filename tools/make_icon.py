"""Makes hypersponge/assets/hypersponge.ico: the Menger sponge on a rounded dark tile.
Run:  uv run python tools/make_icon.py   (needs OpenGL; on Linux without a GPU use xvfb-run)"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hypersponge.core.camera import CameraND
from hypersponge.core.rule import Rule
from hypersponge.render.renderer import render_still
from hypersponge.render.styles import StyleSettings

OUT = Path(__file__).resolve().parents[1] / "hypersponge" / "assets" / "hypersponge.ico"
N = 512

cam = CameraND(Rule(3, 2, 3), N, N, fov=36.0)
img = render_still(cam, StyleSettings(fog=False), samples=32)
pic = Image.fromarray(img).convert("RGBA")
mask = Image.new("L", (N, N), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, N - 1, N - 1), radius=N // 7, fill=255)
pic.putalpha(mask)
sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
pic.resize((256, 256), Image.LANCZOS).save(OUT, sizes=sizes)
print(OUT, OUT.stat().st_size, "bytes", np.asarray(pic).shape)
