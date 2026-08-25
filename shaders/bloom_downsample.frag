// ============================================================================
// bloom_downsample.frag — Phase 11.9.5 Bloom (13-tap downsample)
// ============================================================================
// 13-tap "Next Generation Post Processing" downsampling kernel by Jorge
// Jimenez (Call of Duty: Advanced Warfare, SIGGRAPH 2014). Eliminates the
// aliasing / pulsing artifacts that plague naive 2x2 box downsampling on
// HDR content with very bright pixels.
//
// Sample layout (offsets in src texels):
//   A   B   C
//     D   E
//   F   G   H
//     I   J
//   K   L   M
// Weights:
//   center (G):                 0.125
//   inner ring 4 (D,E,I,J):     0.5 / 4
//   row centers (B,L):          0.5 / 4
//   column centers (F,H):       0.5 / 4
//   corners (A,C,K,M):          0.125 / 4 each row corner
// (See Jimenez paper Eq. 6 — total weight = 1.0)
//
// Inputs:
//   set=0 binding=0: srcImage  (input mip; full res for first downsample,
//                               then mip N-1 for subsequent ones)
// Push constant:
//   vec2 srcTexel = 1.0 / srcSize
// Output: R16G16B16A16_SFLOAT
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D srcImage;

layout(push_constant) uniform DownsamplePush {
    vec2 srcTexel;   // 1.0 / srcSize
    vec2 _pad;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 t = pc.srcTexel;

    // 5×5 area, 13 taps, named per the Jimenez diagram.
    vec3 a = texture(srcImage, fragTexCoord + vec2(-2.0,  2.0) * t).rgb;
    vec3 b = texture(srcImage, fragTexCoord + vec2( 0.0,  2.0) * t).rgb;
    vec3 c = texture(srcImage, fragTexCoord + vec2( 2.0,  2.0) * t).rgb;

    vec3 d = texture(srcImage, fragTexCoord + vec2(-1.0,  1.0) * t).rgb;
    vec3 e = texture(srcImage, fragTexCoord + vec2( 1.0,  1.0) * t).rgb;

    vec3 f = texture(srcImage, fragTexCoord + vec2(-2.0,  0.0) * t).rgb;
    vec3 g = texture(srcImage, fragTexCoord                       ).rgb;
    vec3 h = texture(srcImage, fragTexCoord + vec2( 2.0,  0.0) * t).rgb;

    vec3 i = texture(srcImage, fragTexCoord + vec2(-1.0, -1.0) * t).rgb;
    vec3 j = texture(srcImage, fragTexCoord + vec2( 1.0, -1.0) * t).rgb;

    vec3 k = texture(srcImage, fragTexCoord + vec2(-2.0, -2.0) * t).rgb;
    vec3 l = texture(srcImage, fragTexCoord + vec2( 0.0, -2.0) * t).rgb;
    vec3 m = texture(srcImage, fragTexCoord + vec2( 2.0, -2.0) * t).rgb;

    // Weighted average: center "+" cluster (4 inner samples) gets 0.5 weight,
    // surrounding 9-tap box gets 0.5 weight (split evenly across the 4 quads).
    vec3 result =
          (d + e + i + j) * 0.25 * 0.5
        + (a + b + f + g) * 0.25 * 0.125
        + (b + c + g + h) * 0.25 * 0.125
        + (f + g + k + l) * 0.25 * 0.125
        + (g + h + l + m) * 0.25 * 0.125;

    outColor = vec4(result, 1.0);
}
