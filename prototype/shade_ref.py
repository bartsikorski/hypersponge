"""numpy copy of the shading in shaders/march.frag (linear RGB, before gamma)."""
import numpy as np
from sponge_ref import de_table

PALETTES = {
    "pastel": np.array([[0.93, 0.90, 0.82], [0.95, 0.55, 0.20], [0.25, 0.62, 0.85], [0.55, 0.80, 0.35],
                        [0.85, 0.35, 0.55], [0.98, 0.82, 0.30], [0.45, 0.45, 0.90], [0.30, 0.75, 0.70]]),
    "ember": np.array([[0.98, 0.93, 0.80], [0.99, 0.80, 0.45], [0.98, 0.60, 0.25], [0.90, 0.38, 0.18],
                       [0.70, 0.22, 0.20], [0.48, 0.14, 0.24], [0.30, 0.10, 0.28], [0.60, 0.50, 0.55]]),
    "ice": np.array([[0.95, 0.97, 0.99], [0.78, 0.90, 0.97], [0.55, 0.80, 0.93], [0.35, 0.66, 0.88],
                     [0.22, 0.50, 0.80], [0.18, 0.36, 0.66], [0.16, 0.24, 0.50], [0.50, 0.58, 0.72]]),
}
AXIS_COL = np.array([[0.90, 0.30, 0.25], [0.35, 0.75, 0.35], [0.30, 0.50, 0.95], [0.98, 0.80, 0.25]])
LIGHT = np.array([-0.45, 0.65, -0.62]) / np.linalg.norm([-0.45, 0.65, -0.62])
BG_DARK = np.array([0.035, 0.04, 0.06])
BG_LIGHT = np.array([0.92, 0.93, 0.95])
STYLES = {"sponge": 0, "levels": 1, "axes": 2, "clay": 3, "wdepth": 4}


def shade(tb, F3, D, dc, t, hit, pix, far, yfrac, style="sponge", palette="pastel", bg=BG_DARK,
          fog=True, ao_strength=1.0, wref=(0.0, 1.0), lod=1.0):
    """D: (N,4) ray directions, dc: (N,3) the same in camera space, yfrac: gl_FragCoord.y / height."""
    N = len(t)
    f32 = np.float32
    bgc = np.asarray(bg)[None, :] * (1.0 - 0.35 * (1.0 - yfrac[:, None]))
    out = bgc.copy()
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return out
    p = D[idx] * t[idx, None]
    foot = t[idx] * f32(pix)
    d, row, axis = de_table(p, tb, lodcut=f32(lod) * foot, want_info=True)
    axis = axis % 4
    N4 = np.zeros((len(idx), 4), f32)
    N4[np.arange(len(idx)), axis] = np.where(D[idx, axis] > 0, -1.0, 1.0)
    n = N4 @ F3.astype(f32)
    nl = np.linalg.norm(n, axis=1)
    n = np.where(nl[:, None] > 1e-4, n / np.maximum(nl, 1e-30)[:, None], -dc[idx])
    ns = (n @ F3.T.astype(f32)).astype(f32)
    occ = np.zeros(len(idx), f32)
    h = f32(1.5) * foot
    for _ in range(5):
        occ += np.clip((h - de_table(p + ns * h[:, None], tb, lodcut=f32(lod) * foot)) / h, 0, 1)
        h = h * f32(3.0)
    ao = np.clip(1.0 - 0.95 * ao_strength * occ / 5.0, 0, 1)
    dif = 0.5 + 0.5 * (n @ LIGHT)
    s = STYLES[style]
    if s == 0:
        a = (ao * ao)[:, None]
        col = (np.array([0.45, 0.22, 0.03]) * (1 - a) + np.array([0.97, 0.80, 0.26]) * a) * (0.16 + 0.92 * dif[:, None] ** 1.5)
    else:
        if s == 1:
            levno = np.where(row >= 0, tb.level[np.maximum(row, 0)] % 8, 0)
            base = PALETTES[palette][levno]
        elif s == 2:
            base = AXIS_COL[axis]
        elif s == 4:
            x = 4.0 * (p[:, 3] - wref[0]) / wref[1]
            k = (0.5 + 0.5 * x / (1.0 + np.abs(x)))[:, None]
            base = np.array([0.25, 0.45, 0.95]) * (1 - k) + np.array([0.98, 0.55, 0.15]) * k
        else:
            base = np.tile(np.array([0.85, 0.85, 0.88]), (len(idx), 1))
        col = base * (0.35 + 0.65 * dif[:, None]) * (0.25 + 0.75 * ao[:, None])
    if fog:
        x = 3.0 * t[idx] / far
        fg = (1.0 - np.exp(-x * x))[:, None]
        col = col * (1 - fg) + bgc[idx] * fg
    out[idx] = col
    return out
