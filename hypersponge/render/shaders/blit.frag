// Hypersponge blit: scene or accumulation target to the window, with gamma (design section 14.5).
uniform sampler2D uTex;
uniform float uScale;        // 1 / accumulated samples, or 1
uniform vec4  uViewport;     // destination x, y, width, height in pixels
uniform vec2  uSrcScale;     // part of the texture in use (render scale)
uniform vec2  uTexSize;      // texture size in pixels
uniform int   uRaw;          // 1: copy linear values without gamma (adds a banded sample into accum)
out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy - uViewport.xy) / uViewport.zw * uSrcScale;
    vec2 half_texel = 0.5 / uTexSize;
    uv = clamp(uv, half_texel, uSrcScale - half_texel);    // never sample outside the drawn part
    vec3 c = texture(uTex, uv).rgb * uScale;
    if (uRaw == 1) { fragColor = vec4(c, 1.0); return; }
    fragColor = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
