"""Checks of the camera rules (CPU only)."""
import math, random, time
import numpy as np
from sponge_ref import *
from camera_ref import *
rng = np.random.default_rng(11); random.seed(11)

# frames and presets
for name, (ana, h) in PRESETS.items():
    B = slice_basis(ana)
    assert np.allclose(B.T @ B, np.eye(4), atol=1e-12) and abs(np.linalg.det(B) - 1) < 1e-12
    cam = CameraND(4); cam.preset(name)
    assert np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-12) and abs(np.linalg.det(cam.F) - 1) < 1e-9
    assert cam.overview() and abs(cam.h() - h) < 1e-12 and abs(cam.rho() - np.linalg.norm(START)) < 1e-12
    # the camera looks at the slice's centre point
    to_target = -(cam.F[:, :3] @ cam.rel()[:3]); assert np.allclose(to_target / np.linalg.norm(to_target), cam.F[:, 2], atol=1e-12)
assert np.allclose(slice_basis((0, 0, 0, 1)), np.eye(4))
print("presets: frames orthonormal, determinant +1, eye in overview at the stated offset, looking at the centre")

# overview rotations keep rho and h; 100,000 rotations keep the frame orthonormal
cam = CameraND(4)
rho0, h0 = cam.rho(), cam.h()
for k in range(3000):
    i, j = random.sample(range(4), 2)
    cam.rotate(i, j, random.uniform(-0.2, 0.2))
assert abs(cam.rho() - rho0) < 1e-9 and abs(cam.h() - h0) < 1e-9 and cam.overview()
for k in range(2000):                              # plane jogs (object axes) and rolls
    i, j = random.sample(range(4), 2)
    cam.rotate_object(i, j, random.uniform(-0.2, 0.2)); cam.roll(random.uniform(-0.2, 0.2))
assert abs(cam.rho() - rho0) < 1e-9 and abs(cam.h() - h0) < 1e-9 and cam.overview()
assert np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-12)
F = np.eye(4)
for k in range(100000):
    i, j = random.sample(range(4), 2)
    F = F @ givens(i, j, random.uniform(-0.3, 0.3))
    if k % 16 == 0: F = orthonormalize(F)
assert np.allclose(F.T @ F, np.eye(4), atol=1e-12)
print("rotations: rho and h unchanged after 3,000 overview rotations (drift %.1e); frame orthonormal after 100,000" % abs(cam.rho() - rho0))

# the scene follows the mouse: a point ahead moves right on screen for a drag to the right
def screen_x(cam, P):
    v = cam.F[:, :3].T @ (P - cam.eye_world()); return v[0] / v[2], v[1] / v[2]
cam = CameraND(3); near = cam.eye_world() + cam.F[:, 2] * (cam.rho() - 0.6)      # a point on the near side of the cube
x0, y0 = screen_x(cam, near); cam.drag_rotate(40, 0); x1, _ = screen_x(cam, near); cam.drag_rotate(0, 40); _, y1 = screen_x(cam, near)
assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and cam.overview()
for _ in range(14): cam.wheel(1)
assert not cam.overview()
ahead = cam.eye_world() + cam.F[:, 2] * 0.01
x0, y0 = screen_x(cam, ahead); e0 = cam.eye_world().copy(); cam.drag_rotate(40, 0); x1, _ = screen_x(cam, ahead); cam.drag_rotate(0, 40); _, y1 = screen_x(cam, ahead)
assert x1 > x0 + 0.01 and y1 > y0 + 0.01 and np.allclose(cam.eye_world(), e0)
print("drag: the scene follows the mouse in overview (orbit) and immersed (look around); immersed rotation leaves the eye fixed")

# wheel in then out returns; offset slider; slice segments
cam = CameraND(4); e0 = cam.eye_world().copy(); cam.wheel(1); cam.wheel(-1)
assert np.linalg.norm(cam.eye_world() - e0) < 0.01 * np.linalg.norm(e0 - cam.c)
H = cam.offset_range(); cam.set_offset(0.9 * H); assert abs(cam.h() - 0.9 * H) < 1e-12 and cam.overview(); cam.set_offset(-H)
assert abs(H - 1.0) < 1e-12                                     # diagonal: 0.5 * 4 * 0.5
seg = slice_segments((0, 0, 0, 1), 0.0); assert len(seg) == 12 and all(abs(np.linalg.norm(a - b) - 1) < 1e-12 for a, b in seg)
seg = slice_segments(np.ones(4) / 2, 0.0); assert len(seg) == 12 and all(abs(np.linalg.norm(a - b) - math.sqrt(2)) < 1e-9 for a, b in seg)
assert len(slice_segments(np.ones(4) / 2, 1.01)) == 0 and len(slice_segments(np.ones(4) / 2, 0.9)) == 6    # tetrahedron near a corner
assert len(slice_segments((0, 0, 1, 0), 0.2, n=3)) == 4          # 3D: a square
print("wheel in/out returns; offset slider exact; slice outline: cube 12 edges, octahedron 12 edges of length sqrt 2, tetrahedron 6")

# the Menger preset (w = 1/4) gives the same picture as 3D mode
c4 = CameraND(4, height=96); c4.preset("menger"); c3 = CameraND(3, height=96)
ys, xs = np.mgrid[0:96, 0:128]
dc = np.stack([(xs + 0.5 - 64) * c3.pix, (ys + 0.5 - 48) * c3.pix, np.ones((96, 128))], -1).reshape(-1, 3); dc /= np.linalg.norm(dc, axis=1)[:, None]
t4, h4, _ = march(c4.table, dc @ c4.F[:, :3].T, c4.pix, c4.far); t3, h3, _ = march(c3.table, dc @ c3.F[:, :3].T, c3.pix, c3.far)
assert (h4 == h3).all() and np.allclose(t4[h4], t3[h3], rtol=1e-5)
print("Menger preset: every ray of the 4D slice at w = 1/4 equals 3D mode")

# fuzz: every control at random, 3D and 4D, from the overview and after diving in; never in solid
for n in (3, 4):
    for dive in (0, 30, 150):
        cam = CameraND(n, height=720); t0 = time.perf_counter(); Emax = 0; ops = 1500; imm = 0; rows = 0; far_out = 0.0
        for k in range(dive):
            cam.wheel(1, (random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2), 1.0))
        for k in range(ops):
            r = random.random()
            d = (random.uniform(-0.45, 0.45), random.uniform(-0.3, 0.3), 1.0)
            if r < 0.30: cam.wheel(random.choice((1, 1, 2, -1) if dive else (1, -1)), d)
            elif r < 0.45: cam.drag_rotate(random.uniform(-60, 60), random.uniform(-60, 60))
            elif r < 0.60: cam.fly(rng.normal(0, 1, 3), random.uniform(0.005, 0.05), random.random() < 0.3)
            elif r < 0.70: cam.pan(random.uniform(-80, 80), random.uniform(-80, 80), cam.sigma)
            elif r < 0.75: cam.rotate(0, 1, random.uniform(-0.5, 0.5))
            elif n == 4 and r < 0.87: cam.drag_turn(random.uniform(-60, 60), random.uniform(-60, 60))
            elif n == 4 and r < 0.97: cam.slide(random.choice((-2, -1, 1, 2)))
            elif n == 4 and r < 0.972 and not dive: cam.preset(random.choice(list(PRESETS)))
            elif r < 0.985: cam.rotate_object(*random.sample(range(n), 2), random.uniform(-0.3, 0.3))
            else: cam.update_sigma(0.05)
            assert cam.de > 0 and np.allclose(cam.F.T @ cam.F, np.eye(4), atol=1e-9)
            dist = np.linalg.norm(cam.eye_world() - cam.c); far_out = max(far_out, dist)
            assert dist <= LEASH + 1e-9
            assert not (n == 4 and cam.overview()) or abs(cam.h()) <= cam.offset_range() + 1e-9
            Emax = max(Emax, cam.pos.E); imm += not cam.overview(); rows = max(rows, len(cam.table.w) + cam.table.dropped)
        print(f"fuzz {n}D, {dive:3d} wheel notches first: {ops} random inputs, never in solid, {imm} immersed, "
              f"deepest unit 3^-{Emax}, most rows {rows}, furthest {far_out:.1f}, {(time.perf_counter() - t0) / ops * 1e3:.2f} ms per input")

# nothing under the cursor: the wheel approaches the point of the ray nearest the centre and stops; the leash holds
cam = CameraND(3); cam.drag_rotate(0, 0)
for _ in range(200): cam.wheel(1, (0.45, 0.30, 1.0))            # a corner of the picture, a miss
d = cam.eye_world() - cam.c; D = cam.F[:, :3] @ (np.array([0.45, 0.30, 1.0]) / np.linalg.norm([0.45, 0.30, 1.0]))
assert abs(d @ D) < 0.2 and np.linalg.norm(d) < 3.0
for _ in range(400): cam.wheel(-3)
assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH
for _ in range(400): cam.fly((0.0, 0.0, -1.0), 0.1, True)
assert np.linalg.norm(cam.eye_world() - cam.c) <= LEASH
cam = CameraND(4); cam.set_offset(0.95); cam.rotate(2, 3, 0.6); assert abs(cam.h()) <= cam.offset_range() + 1e-12
for _ in range(100): cam.slide(2)
assert abs(cam.h()) <= cam.offset_range() + 1e-12 and cam.overview()
print("wheel on a miss stops beside the object; the eye stays within 30 units; in overview the slice always meets the cube")
print("all camera checks passed")
