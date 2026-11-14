"""The fast level_table against the straightforward one (every level with big integers)."""
import random, time
import numpy as np
from sponge_ref import *
from level_table_simple import level_table_simple
random.seed(21); rng = np.random.default_rng(21)

def random_pos(n, E, kind):
    O = []
    for i in range(4):
        if i >= n or E == 0:
            O.append(0); continue
        d = [random.choice("012") for _ in range(E)]
        if kind == "wall":                          # a long run of 0s or 2s: beside a wall of a coarse level
            j = random.randrange(E); c = random.choice("02")
            stop = E if random.random() < 0.7 else random.randrange(j, E + 1)
            d[j:stop] = c * (stop - j)
        elif kind == "hole":                        # digit 1 on every axis at one coarse level
            d[random.randrange(E)] = "1"
        elif kind == "carpet":                      # no digit 1 at all
            d = [random.choice("02") for _ in range(E)]
        O.append(int("".join(d), 3))
    if kind == "outside" and E > 0:
        O[random.randrange(n)] += random.choice((-1, 1)) * 3 ** E * random.randrange(1, 3)
    f = rng.random(4); f[n:] = 0.0
    if kind == "wall":
        f[:n] = np.where(rng.random(n) < 0.5, f[:n] * 1e-6, 1 - f[:n] * 1e-6)
    return Pos(O, f, E)

cases = 0; t_new = t_old = 0.0; rows = 0
for n in (2, 3, 4):
    for m in range(1, n + 1):
        for E in (0, 1, 5, 35, 36, 37, 40, 80, 150, 300):
            for kind in ("random", "wall", "hole", "carpet", "outside", "normal"):
                for rep in range(3):
                    pos = random_pos(n, E, "wall" if kind == "normal" else kind)
                    if kind == "normal":
                        try: normalize(pos, n, m)
                        except ValueError: continue
                    far = random.choice((0.0, 3.0, 1.0e3, 1.0e7))
                    ml = random.choice((None, None, max(1, E // 2), E + 3))
                    fm = random.choice((None, 1e-3))
                    cover = n == 2 and random.random() < 0.5
                    t0 = time.perf_counter(); a = level_table(pos, n, m, ml, far, fm, cover=cover); t1 = time.perf_counter()
                    b = level_table_simple(pos, n, m, ml, far, fm, cover=cover); t2 = time.perf_counter()
                    t_new += t1 - t0; t_old += t2 - t1; cases += 1; rows = max(rows, len(a.w))
                    assert a.covered == b.covered and a.dropped == b.dropped, (n, m, E, kind)
                    if not (cover and a.covered):   # with an early stop the camera estimate covers fewer levels
                        # deep inside solid (further than 3^35 units from any hole) the fast version reports -BIG
                        assert a.de_cam == b.de_cam or (a.de_cam < -1e16 and b.de_cam < -1e16), (n, m, E, kind, a.de_cam, b.de_cam)
                    for x, y in ((a.level, b.level), (a.w, b.w), (a.lo, b.lo), (a.hi, b.hi), (a.box_lo, b.box_lo), (a.box_hi, b.box_hi)):
                        assert x.shape == y.shape and np.array_equal(x, y), (n, m, E, kind, far, ml)
print(f"{cases} tables (2D, 3D, 4D; every rule; depths 3^0 to 3^-300; random, beside walls, inside holes, outside the cube, "
      f"normalised): identical to the straightforward version; total time {t_new:.1f} s against {t_old:.1f} s")
for E in (100, 400, 1000, 3000, 10000):
    p = Pos.from_floats([0.25, 0.25, 0.25, 0.25]); p.rescale(E); p.move([0.3, 0.1, 0.2, 0.4])
    t0 = time.perf_counter()
    for _ in range(5): tb = level_table(p, 3, 2, far=40.0, fine_min=1e-3)
    t1 = time.perf_counter()
    if E <= 1000:
        tb2 = level_table_simple(p, 3, 2, far=40.0, fine_min=1e-3); t2 = time.perf_counter()
        assert np.array_equal(tb.lo, tb2.lo)
        print(f"depth 3^-{E}: {(t1 - t0) / 5 * 1e3:.1f} ms per table ({(t2 - t1) * 1e3:.0f} ms for the straightforward version), {len(tb.w)} rows")
    else:
        print(f"depth 3^-{E}: {(t1 - t0) / 5 * 1e3:.1f} ms per table, {len(tb.w)} rows")
print("all level table checks passed")
