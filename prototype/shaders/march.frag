// Hypersponge 3D / 4D ray marcher with shading (prototype, verified against the numpy reference).
uniform mat4  uFrame;        // columns: right, up, forward, ana (object axes)
uniform float uPix;          // angle of one pixel = 2 tan(fov/2) / height
uniform float uFar;          // view distance (local units)
uniform vec2  uRes;          // render target size in pixels
uniform vec2  uJitter;       // sub-pixel offset in [-0.5, 0.5) for accumulation
uniform int   uMaxSteps;
uniform float uLod;          // level-of-detail cut in pixels (1.0)
uniform int   uStyle;        // 0 sponge, 1 levels, 2 axes, 3 clay, 4 w-depth
uniform vec3  uPalette[8];   // Levels style
uniform vec3  uBg;           // background (linear RGB)
uniform float uFog;          // 1.0 = fog on
uniform float uAo;           // occlusion strength 0..1
uniform vec2  uWRef;        // w-depth style: the neutral w offset and the scale (local units)
uniform int   uOutMode;      // 0 colour, 1 data: (t, hit, row, axis + steps/1024)
out vec4 fragColor;

const vec3 AXIS_COL[4] = vec3[4](vec3(0.90, 0.30, 0.25), vec3(0.35, 0.75, 0.35),
                                 vec3(0.30, 0.50, 0.95), vec3(0.98, 0.80, 0.25));
const vec3 LIGHT = normalize(vec3(-0.45, 0.65, -0.62));   // camera space: up and to the left, towards the viewer

void main() {
    vec2 pxy = gl_FragCoord.xy + uJitter - 0.5 * uRes;
    vec3 dc = normalize(vec3(pxy * uPix, 1.0));
    vec4 D = uFrame * vec4(dc, 0.0);
    float t = 0.0;
    bool hit = false;
    int steps = 0;
    for (int i = 0; i < uMaxSteps; i++) {
        float foot = t * uPix;
        float d = de(D * t, uLod * foot);
        steps++;
        if (d < 0.5 * foot) { hit = true; break; }
        t += max(d, 0.0);
        if (t > uFar) break;
    }
    if (steps >= uMaxSteps && t <= uFar) hit = true;       // out of steps counts as a hit
    float foot = t * uPix;
    vec4 p = D * t;
    int row, axis;
    deInfo(p, uLod * foot, row, axis);
    if (uOutMode == 1) {
        fragColor = vec4(t, hit ? 1.0 : 0.0, float(row), float(axis) + float(steps) / 1024.0);
        return;
    }
    vec3 bg = uBg * (1.0 - 0.35 * (1.0 - gl_FragCoord.y / uRes.y));
    if (!hit) { fragColor = vec4(bg, 1.0); return; }

    // exact normal: the face's object axis, signed to face the ray, projected into the slice
    vec4 N4 = vec4(0.0);
    N4[axis] = D[axis] > 0.0 ? -1.0 : 1.0;
    vec3 n = vec3(dot(N4, uFrame[0]), dot(N4, uFrame[1]), dot(N4, uFrame[2]));
    float nl = length(n);
    n = nl > 1.0e-4 ? n / nl : -dc;
    vec4 ns = uFrame * vec4(n, 0.0);                       // in-slice unit normal, object axes

    float occ = 0.0;
    float h = 1.5 * foot;
    for (int j = 0; j < 5; j++) {
        occ += clamp((h - de(p + ns * h, uLod * foot)) / h, 0.0, 1.0);
        h *= 3.0;
    }
    float ao = clamp(1.0 - 0.95 * uAo * occ / 5.0, 0.0, 1.0);
    float dif = 0.5 + 0.5 * dot(n, LIGHT);

    vec3 col;
    if (uStyle == 0) {
        col = mix(vec3(0.45, 0.22, 0.03), vec3(0.97, 0.80, 0.26), ao * ao) * (0.16 + 0.92 * pow(dif, 1.5));
    } else {
        vec3 base;
        if (uStyle == 1) base = uPalette[row >= 0 ? uLevelNo[row] % 8 : 0];
        else if (uStyle == 2) base = AXIS_COL[axis];
        else if (uStyle == 4) {                            // w offset from the reference, soft-limited
            float x = 4.0 * (p.w - uWRef.x) / uWRef.y;
            base = mix(vec3(0.25, 0.45, 0.95), vec3(0.98, 0.55, 0.15), 0.5 + 0.5 * x / (1.0 + abs(x)));
        }
        else base = vec3(0.85, 0.85, 0.88);
        col = base * (0.35 + 0.65 * dif) * (0.25 + 0.75 * ao);
    }
    if (uFog > 0.5) {
        float x = 3.0 * t / uFar;
        col = mix(col, bg, 1.0 - exp(-x * x));
    }
    fragColor = vec4(col, 1.0);
}
