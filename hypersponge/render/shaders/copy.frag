// Hypersponge copy: a finished banded sample from the scene target, added into the accumulation
// target with blending (design section 16.3, long-frame guard).  No gamma: linear values.
uniform sampler2D uTex;
out vec4 fragColor;

void main() {
    fragColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
}
