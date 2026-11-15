// Hypersponge shared GLSL: level table uniforms and the distance estimator (design sections 3.5, 14.2).
// Prepended (after "#version 330") to march.frag, carpet.frag and the test probe.
#define MAXLEV 64
const float BIG = 1.0e30;
uniform int   uCount;            // active rows, widest first
uniform vec4  uLo[MAXLEV];       // camera-relative lower wall of the middle third, per axis
uniform vec4  uHi[MAXLEV];       // camera-relative upper wall
uniform float uW[MAXLEV];        // width of the middle third (local units)
uniform int   uLevelNo[MAXLEV];  // absolute level number of the row (for colours)
uniform vec4  uBoxLo;            // camera-relative faces of the outer cube
uniform vec4  uBoxHi;
uniform vec4  uOff;              // 1.0 = axis switched off (2D: z and w, 3D: w)
uniform int   uM;                // rule: removed when >= uM coordinates are "middle"

vec4 sort4(vec4 a) {
    vec2 lo = min(a.xz, a.yw), hi = max(a.xz, a.yw);
    float t0 = min(lo.x, lo.y), t2 = max(lo.x, lo.y);
    float t1 = min(hi.x, hi.y), t3 = max(hi.x, hi.y);
    return vec4(t0, min(t1, t2), max(t1, t2), t3);
}

// signed distance to the middle slab of row i on each axis (negative inside)
vec4 wallDist(vec4 q, int i) {
    float w = uW[i];
    vec4 u = q - uLo[i];
    vec4 a = max(uLo[i] - q, q - uHi[i]);                  // exact near the camera
    vec4 v = mod(u + w, 3.0 * w) - w;                      // other periods, for far points
    a = mix(a, max(-v, v - w), greaterThan(abs(u - 0.5 * w), vec4(1.5 * w)));
    return mix(a, vec4(BIG), uOff);
}

float boxDist(vec4 q, out vec4 b) {
    b = mix(max(uBoxLo - q, q - uBoxHi), vec4(-BIG), uOff);
    return length(max(b, 0.0)) + min(max(max(b.x, b.y), max(b.z, b.w)), 0.0);
}

// distance estimate; rows narrower than lodcut are skipped
float de(vec4 q, float lodcut) {
    vec4 b;
    float d = boxDist(q, b);
    for (int i = 0; i < uCount; i++) {
        if (uW[i] < lodcut) break;
        d = max(d, -sort4(wallDist(q, i))[uM - 1]);
    }
    return d;
}

// same value; also the row (-1 = outer cube) and the axis (0..3) that set it
float deInfo(vec4 q, float lodcut, out int row, out int axis) {
    vec4 b;
    float d = boxDist(q, b);
    row = -1;
    float bm = max(max(b.x, b.y), max(b.z, b.w));
    axis = b.x == bm ? 0 : (b.y == bm ? 1 : (b.z == bm ? 2 : 3));
    for (int i = 0; i < uCount; i++) {
        if (uW[i] < lodcut) break;
        vec4 a = wallDist(q, i);
        float am = sort4(a)[uM - 1];
        if (-am > d) {
            d = -am; row = i;
            axis = a.x == am ? 0 : (a.y == am ? 1 : (a.z == am ? 2 : 3));
        }
    }
    return d;
}

// 2D/any: first row that removes the point (-1 = kept, -2 = outside the cube)
int removalRow(vec4 q, float lodcut) {
    vec4 b;
    if (boxDist(q, b) > 0.0) return -2;
    for (int i = 0; i < uCount; i++) {
        if (uW[i] < lodcut) break;
        if (sort4(wallDist(q, i))[uM - 1] < 0.0) return i;
    }
    return -1;
}
