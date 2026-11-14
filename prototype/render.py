"""CPU sample renderer for the Hypersponge design (slow; numpy).  Not the product renderer."""
import sys, time
import numpy as np
from PIL import Image
from sponge_ref import *

PAL = np.array([[0.93,0.90,0.82],[0.95,0.55,0.20],[0.25,0.62,0.85],[0.55,0.80,0.35],[0.85,0.35,0.55],
                [0.98,0.82,0.30],[0.45,0.45,0.90],[0.30,0.75,0.70]])
AXC = np.array([[0.90,0.30,0.25],[0.35,0.75,0.35],[0.30,0.50,0.95],[0.98,0.80,0.25]])

def frame_look(eye, target, up, basis):
    """Camera frame inside the slice spanned by `basis` (4x3, orthonormal columns)."""
    fwd = basis.T @ (np.asarray(target, float) - np.asarray(eye, float)); fwd /= np.linalg.norm(fwd)
    upv = np.asarray(up, float); right = np.cross(upv, fwd); right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    R = np.stack([right, upv, fwd], axis=1)          # slice coords of camera axes
    return basis @ R                                 # 4x3: right, up, forward in 4D

def render(eye, F3, n, m, size=(320, 240), fov=50.0, style="sponge", far=1e4, fog=0.0, max_level=None,
           max_steps=220, bg=(0.035, 0.04, 0.06)):
    W, H = size
    pos = eye.copy() if isinstance(eye, Pos) else Pos.from_floats(eye, 0)
    de = normalize(pos, n, m, max_level)
    pix = 2 * np.tan(np.radians(fov) / 2) / H
    tb = level_table(pos, n, m, max_level, far=far, fine_min=0.3 * de * pix)
    ys, xs = np.mgrid[0:H, 0:W]
    cx = (xs + 0.5 - W / 2) * pix; cy = -(ys + 0.5 - H / 2) * pix
    dc = np.stack([cx, cy, np.ones_like(cx)], -1).reshape(-1, 3); dc /= np.linalg.norm(dc, axis=1)[:, None]
    D = (dc @ F3.T).astype(np.float32)               # (N,4)
    t0 = time.time()
    t, hit, steps = march(tb, D, pix, far, max_steps=max_steps)
    hit |= (steps >= max_steps) & (t < far)
    img = np.tile(np.array(bg), (W * H, 1)) * (1.0 - 0.35 * (ys.reshape(-1, 1) / H))
    idx = np.nonzero(hit)[0]
    p = D[idx] * t[idx, None]; foot = t[idx] * np.float32(pix)
    d, lev, axis = de_table(p, tb, lodcut=foot, want_info=True)
    ax = axis % 4
    N4 = np.zeros((len(idx), 4), np.float32)
    N4[np.arange(len(idx)), ax] = -np.sign(D[idx, ax] + 1e-20)
    n3 = N4 @ F3                                      # camera-space normal (unnormalised projection)
    n3 /= (np.linalg.norm(n3, axis=1)[:, None] + 1e-12)
    ns = (n3 @ F3.T).astype(np.float32)               # in-slice unit normal in 4D
    occ = np.zeros(len(idx), np.float32)
    for i in range(1, 6):
        h = foot * np.float32(1.5 * 3 ** (i - 1))
        di = de_table(p + ns * h[:, None], tb, lodcut=foot)
        occ += np.clip((h - di) / h, 0, 1) / 5
    ao = np.clip(1 - 0.95 * occ, 0, 1)
    L = np.array([-0.45, 0.65, -0.62]); L /= np.linalg.norm(L)
    dif = 0.5 + 0.5 * (n3 @ L)
    levno = np.where(lev >= 0, tb.level[np.maximum(lev, 0)], 0)
    if style == "sponge":
        base = np.array([0.97, 0.80, 0.26]); cav = np.array([0.45, 0.22, 0.03])
        a = (ao ** 2.0)[:, None]
        col = (cav * (1 - a) + base * a) * (0.16 + 0.92 * dif[:, None] ** 1.5)
    elif style == "levels":
        col = PAL[levno % len(PAL)] * (0.35 + 0.65 * dif[:, None]) * (0.25 + 0.75 * ao[:, None])
    elif style == "axes":
        col = AXC[ax] * (0.35 + 0.65 * dif[:, None]) * (0.25 + 0.75 * ao[:, None])
    else:
        col = np.array([0.85, 0.85, 0.88]) * (0.25 + 0.75 * dif[:, None]) * (0.2 + 0.8 * ao[:, None])
    if fog > 0:
        fg = 1 - np.exp(-(t[idx] / fog) ** 2)
        col = col * (1 - fg[:, None]) + img[idx] * fg[:, None]
    img[idx] = col
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    info = dict(E=pos.E, de=de, levels=len(tb.w), hit=float(hit.mean()), steps=float(steps.mean()), sec=time.time() - t0)
    return (img.reshape(H, W, 3) * 255).astype(np.uint8), info

def zoom_in(eye, F3, n, m, steps, frac=0.7, fov=50.0, H=540):
    """The wheel-zoom rule: move a fraction of the way to the surface under the centre ray."""
    pos = Pos.from_floats(eye, 0); pix = 2 * np.tan(np.radians(fov) / 2) / H
    fwd = F3[:, 2].astype(np.float64); log = []
    for i in range(steps):
        de = normalize(pos, n, m)
        tb = level_table(pos, n, m, far=1e4, fine_min=0.3 * de * pix)
        t, hit, st = march(tb, fwd[None, :], pix, 1e4, max_steps=400)
        assert hit[0], "centre ray missed"
        pos.move(fwd * (frac * float(t[0])))
        log.append((pos.E, float(t[0]), len(tb.w)))
    return pos, log

def slice_basis(normal):
    nrm = np.asarray(normal, float); nrm /= np.linalg.norm(nrm)
    M = np.linalg.qr(np.column_stack([nrm, np.eye(4)[:, :3] + 0.01 * np.arange(12).reshape(4, 3)]))[0]
    M[:, 0] *= np.sign(M[:, 0] @ nrm)
    return M[:, 1:4]

if __name__ == "__main__":
    which = sys.argv[1]; size = tuple(int(v) for v in sys.argv[2].split("x")); out = sys.argv[3]
    c = np.array([0.5, 0.5, 0.5, 0.5])
    if which == "sponge3d":
        B = np.eye(4)[:, :3]; eye = np.array([0.5, 0.5, 0.5, 0.0]) + B @ np.array([1.45, 1.05, -1.75])
        img, info = render(eye, frame_look(eye, [0.5, 0.5, 0.5, 0], [0, 1, 0], B), 3, 2, size, style="sponge")
    elif which == "inside3d":
        B = np.eye(4)[:, :3]; eye = np.array([0.5, 0.5, 0.5, 0.0]) + B @ np.array([0.02, 0.013, 0.007])
        img, info = render(eye, frame_look(eye, [1.0, 0.86, 0.68, 0], [0, 1, 0], B), 3, 2, size, fov=75, style="sponge", fog=40.0, far=160)
    elif which.startswith("diag4d"):
        m = int(which[-1]); B = slice_basis([1, 1, 1, 1]); eye = c + B @ np.array([1.5, 1.1, -1.9])
        img, info = render(eye, frame_look(eye, c, [0, 1, 0], B), 4, m, size, style="axes" if "ax" in which else "sponge")
    elif which.startswith("w4d"):
        w = float(which.split("_")[1]); B = np.eye(4)[:, :3]
        eye = np.array([0.5, 0.5, 0.5, w]) + B @ np.array([1.45, 1.05, -1.75])
        img, info = render(eye, frame_look(eye, [0.5, 0.5, 0.5, w], [0, 1, 0], B), 4, 2, size, style="sponge")
    elif which.startswith("tilt4d"):
        ang = np.radians(float(which.split("_")[1])); B = np.eye(4)[:, :3].copy()
        B[:, 0] = [np.cos(ang), 0, 0, np.sin(ang)]            # x axis turned towards w
        eye = c + B @ np.array([1.45, 1.05, -1.75])
        img, info = render(eye, frame_look(eye, c, [0, 1, 0], B), 4, 2, size, style="axes")
    elif which.startswith("deep3d"):
        k = int(which.split("_")[1]); B = np.eye(4)[:, :3]
        eye = np.array([0.5, 0.5, 0.5, 0.0]) + B @ np.array([1.45, 1.05, -1.75])
        F3 = frame_look(eye, [0.43, 0.61, 0.17, 0], [0, 1, 0], B)
        pos, log = zoom_in(eye, F3, 3, 2, k)
        print("zoom log (E, distance to surface in local units, levels):", log[:3], "...", log[-3:])
        img, info = render(pos, F3, 3, 2, size, style="sponge", fog=60.0, far=240.0)
    elif which.startswith("deep4d"):
        k = int(which.split("_")[1]); B = slice_basis([1, 2, 3, 4]); eye = c + B @ np.array([1.5, 1.1, -1.9])
        F3 = frame_look(eye, c + B @ np.array([0.11, 0.07, 0.0]), [0, 1, 0], B)
        pos, log = zoom_in(eye, F3, 4, 2, k)
        print("zoom log:", log[:3], "...", log[-3:])
        img, info = render(pos, F3, 4, 2, size, style="levels", fog=60.0, far=240.0)
    print(which, info)
    Image.fromarray(img).save(out)
