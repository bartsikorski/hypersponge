"""Renders the program icon, hypersponge/assets/hypersponge.ico, from the 3D overview.
Run from the repository folder:  uv run python tools/make_icon.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

from hypersponge.core.camera import CameraND
from hypersponge.core.rule import Rule
from hypersponge.render.context import make_headless_context
from hypersponge.render.renderer import Renderer, render_still
from hypersponge.render.styles import StyleSettings

OUT = Path(__file__).resolve().parents[1] / "hypersponge" / "assets" / "hypersponge.ico"


def main():
    cam = CameraND(Rule(3, 2), 512, 512, fov=34.0)
    r = Renderer(make_headless_context(), dynamic=False)
    img = render_still(cam, StyleSettings(), samples=16, renderer=r)
    hit = r.data_3d(cam)[::-1, :, 1] > 0.5                                    # the sponge, not the sky
    mask = hit.astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    pad = 6
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, 511), min(ys.max() + pad, 511))
    rgba = Image.fromarray(np.dstack([img, mask])).crop(box)
    side = max(rgba.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(rgba, ((side - rgba.size[0]) // 2, (side - rgba.size[1]) // 2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sq.resize((256, 256), Image.LANCZOS).save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
