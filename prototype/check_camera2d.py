"""Checks of the 2D camera rules (CPU only)."""
import random, time
from fractions import Fraction as F
import numpy as np
from sponge_ref import *
from camera2d_ref import Camera2D, MAX_HEIGHT
random.seed(5)

def under_cursor(cam, cur):
    X = cam.pos.exact(); s = F(3) ** cam.pos.E
    return [X[i] + F(cur[i] * cam.pixel) / s for i in range(2)]

# the point under the cursor stays put while zooming, at every depth
cam = Camera2D()
cam.pos = Pos.from_floats([0.25, 0.25, 0, 0]); cam.refresh()
worst = 0.0; rows = 0; t0 = time.time(); n = 0; holes = 0
for k in range(2600):
    cur = (random.uniform(-900, 900), random.uniform(-500, 500))
    before = under_cursor(cam, cur); deep = cam.pos.E >= 3
    cam.zoom(random.choice([1, 1, 1, 2, -1]), cur)
    after = under_cursor(cam, cur)
    hw = F(cam.height) / F(3) ** cam.pos.E
    if deep and cam.pos.E >= 3:                    # near the top the centre clamp may act
        worst = max(worst, max(abs(float((a - b) / hw)) for a, b in zip(after, before)))
    rows = max(rows, len(cam.table.w)); n += 1; holes += cam.table.covered > 0
    assert 1.0 <= cam.height < 3.0 or cam.pos.E == 0
    assert cam.table.dropped == 0
print(f"zoom about the cursor: {n} wheel events to depth 3^-{cam.pos.E}; cursor point moves at most {worst * 1080:.2f} pixel per event; "
      f"at most {rows} table rows; view inside one hole on {holes} events; {(time.time() - t0) / n * 1e3:.2f} ms per event")
assert worst < 0.5 / 1080 and cam.pos.E > 300 and rows <= 12     # under half a pixel (the anchor rule's shift)

# zoom in then out returns; zoom-out stops at the widest view; the centre stays in the square
cam = Camera2D(); c0 = cam.pos.exact(); h0 = cam.height
for _ in range(500): cam.zoom(1, (300, -200))
for _ in range(500): cam.zoom(-1, (300, -200))
c1 = cam.pos.exact()
assert cam.pos.E == 0 and abs(cam.height - h0) < 1e-9 and all(abs(float(a - b)) < 2 * h0 / 1080 for a, b in zip(c0, c1))   # within 2 pixels
for _ in range(50): cam.zoom(-1)
assert abs(cam.world_height() - MAX_HEIGHT) < 1e-12
for _ in range(400): cam.pan(random.uniform(-2000, 2000), random.uniform(-2000, 2000))
X = cam.pos.exact(); assert all(0 <= x <= 1 for x in X[:2])
cam = Camera2D()
for _ in range(300): cam.zoom(1, (500, 300))
x0 = under_cursor(cam, (100, 50)); cam.pan(37, -21); x1 = under_cursor(cam, (137, 29))
hw = F(cam.height) / F(3) ** cam.pos.E
assert all(abs(float((a - b) / hw)) < 1e-12 for a, b in zip(x0, x1))
print("zoom in and out returns to the start within 2 pixels; widest view is 4 units; panning keeps the grabbed point under the mouse; centre stays in the square")

# zoom anchor rule: aiming at material that is visible on screen never ends inside a hole
rng = np.random.default_rng(3)
for m in (2, 1):
    cam = Camera2D(m=m); events = 0; shift = 0.0
    while cam.pos.E < 400:
        for half in (40, 200):                     # a pixel near the centre that shows material
            ys, xs = np.mgrid[-half:half, -half * 3 // 2:half * 3 // 2]
            q = np.zeros((xs.size, 4)); q[:, 0] = (xs.ravel() + 0.5) * cam.pixel; q[:, 1] = (ys.ravel() + 0.5) * cam.pixel
            kept = np.nonzero(removal_row(q, cam.table, cam.pixel / 3, np.float64) == -1)[0]
            if kept.size:
                break
        assert kept.size > 0, "no material in view"
        pick = int(kept[rng.integers(kept.size)])
        cur = (float(xs.ravel()[pick]) + 0.5, float(ys.ravel()[pick]) + 0.5)
        for _ in range(int(rng.integers(1, 12))):
            q4 = np.array([cur[0], cur[1], 0.0, 0.0]) * cam.pixel
            shift = max(shift, float(np.abs(cam.anchor_shift(q4)).max() / cam.pixel))
            cam.zoom(1, cur); events += 1
            q = np.array([[cur[0] * cam.pixel, cur[1] * cam.pixel, 0, 0]])
            assert cam.table.covered == 0 and removal_row(q, cam.table, cam.pixel / 3, np.float64)[0] == -1, (m, events, cam.pos.E)
    print(f"zoom anchor rule, m={m}: {events} wheel events on visible material to depth 3^-{cam.pos.E}, never inside a hole; "
          f"largest shift {shift:.2f} pixel")
print("all 2D camera checks passed")
