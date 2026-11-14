"""Styles, palettes and the colour constants of design section 17.  All colours are linear RGB."""
from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

STYLES = {"sponge": 0, "levels": 1, "axes": 2, "clay": 3, "wdepth": 4}
STYLE_NAMES = {"sponge": "Sponge", "levels": "Levels", "axes": "Axes", "clay": "Clay", "wdepth": "W-depth"}
STYLE_MODES = {"sponge": ("2d", "3d", "4d"), "levels": ("2d", "3d", "4d"), "axes": ("3d", "4d"),
               "clay": ("2d", "3d", "4d"), "wdepth": ("4d",)}

PALETTES = {
    "pastel": np.array([[0.93, 0.90, 0.82], [0.95, 0.55, 0.20], [0.25, 0.62, 0.85], [0.55, 0.80, 0.35],
                        [0.85, 0.35, 0.55], [0.98, 0.82, 0.30], [0.45, 0.45, 0.90], [0.30, 0.75, 0.70]]),
    "ember": np.array([[0.98, 0.93, 0.80], [0.99, 0.80, 0.45], [0.98, 0.60, 0.25], [0.90, 0.38, 0.18],
                       [0.70, 0.22, 0.20], [0.48, 0.14, 0.24], [0.30, 0.10, 0.28], [0.60, 0.50, 0.55]]),
    "ice": np.array([[0.95, 0.97, 0.99], [0.78, 0.90, 0.97], [0.55, 0.80, 0.93], [0.35, 0.66, 0.88],
                     [0.22, 0.50, 0.80], [0.18, 0.36, 0.66], [0.16, 0.24, 0.50], [0.50, 0.58, 0.72]]),
}
BACKGROUNDS = {"dark": (0.035, 0.04, 0.06), "light": (0.92, 0.93, 0.95)}
AXIS_COL = np.array([[0.90, 0.30, 0.25], [0.35, 0.75, 0.35], [0.30, 0.50, 0.95], [0.98, 0.80, 0.25]])
SPONGE_BASE = (0.97, 0.80, 0.26)
SPONGE_CAVITY = (0.45, 0.22, 0.03)
CLAY = (0.85, 0.85, 0.88)


def styles_for(mode: str) -> list:
    return [s for s, modes in STYLE_MODES.items() if mode in modes]


@dataclass
class StyleSettings:
    style: str = "sponge"
    palette: str = "pastel"
    background: str = "dark"
    fog: bool = True
    occlusion: float = 1.0
    inverted: bool = False        # 2D only
    quality: float = 0.5          # 0 .. 1

    def key(self):
        return astuple(self)

    def style_for(self, mode: str) -> str:
        """The style actually drawn: a style the mode does not have falls back to Sponge."""
        return self.style if mode in STYLE_MODES.get(self.style, ()) else "sponge"

    @property
    def bg(self) -> tuple:
        return BACKGROUNDS[self.background]

    @property
    def lod(self) -> float:
        """3D and 4D detail cut in pixels: 2 at quality 0, 1 at 0.5, 0.5 at 1."""
        return 2.0 ** (1.0 - 2.0 * self.quality)

    @property
    def max_steps(self) -> int:
        """3D and 4D step limit: 150 at quality 0, 300 at 0.5, 600 at 1."""
        return int(round(150 * 4.0 ** self.quality))

    @property
    def supersample(self) -> int:
        """2D samples per axis per pixel: 2 at quality 0, 3 at 0.5, 4 at 1."""
        return 2 + int(round(2.0 * self.quality))
