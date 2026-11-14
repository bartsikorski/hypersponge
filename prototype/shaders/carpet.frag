// Hypersponge 2D carpet (prototype, verified against exact base-3 membership).
uniform vec2  uRes;          // render target size in pixels
uniform float uPixel;        // size of one pixel in local units = view height / uRes.y
uniform vec2  uJitter;       // sub-pixel offset for accumulation
uniform int   uSS;           // samples per axis per pixel (3)
uniform float uCut;          // rows narrower than uCut pixels are skipped (1/3; for rule m = 1 use 2.0)
uniform int   uStyle;        // 0 sponge, 1 levels, 3 clay
uniform vec3  uPalette[8];
uniform vec3  uBg;
uniform float uInvert;       // 1.0 swaps holes and material
uniform int   uOutMode;      // 0 colour, 1 data: removalRow at the pixel centre
out vec4 fragColor;

vec3 shade(int row) {
    if (row == -2) return uBg;
    bool kept = (row == -1) != (uInvert > 0.5);
    // hole size on screen: 0 = one pixel, 1 = 3^6 pixels or more (continuous while zooming)
    float size = row < 0 ? 0.0 : clamp(log(uW[row] / uPixel) / 6.59, 0.0, 1.0);
    if (uStyle == 0) {
        vec3 hole = mix(vec3(0.62, 0.36, 0.08), vec3(0.20, 0.09, 0.01), size);
        return kept ? vec3(0.97, 0.80, 0.26) : hole;
    }
    if (uStyle == 1) {
        vec3 hole = row < 0 ? uBg * 1.5 : uPalette[uLevelNo[row] % 8] * 0.55;
        return kept ? vec3(0.93, 0.90, 0.82) : hole;
    }
    return kept ? vec3(0.85, 0.85, 0.88) : uBg * 0.45;
}

void main() {
    vec2 c = gl_FragCoord.xy + uJitter - 0.5 * uRes;
    if (uOutMode == 1) {
        fragColor = vec4(float(removalRow(vec4(c * uPixel, 0.0, 0.0), uPixel / 3.0)));
        return;
    }
    vec3 acc = vec3(0.0);
    for (int j = 0; j < uSS; j++)
        for (int i = 0; i < uSS; i++) {
            vec2 o = (vec2(float(i), float(j)) + 0.5) / float(uSS) - 0.5;
            acc += shade(removalRow(vec4((c + o) * uPixel, 0.0, 0.0), uCut * uPixel));
        }
    fragColor = vec4(acc / float(uSS * uSS), 1.0);
}
