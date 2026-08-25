// ============================================================================
// bloom_upsample.frag — Phase 11.9.5 Bloom (9-tap tent upsample, additive)
// ============================================================================
// 3×3 tent ("hat") filter upsampling — Jorge Jimenez COD:AW upsampling pass.
// The input is the smaller mip; we run this fragment shader at the larger
// mip's resolution and bilinearly upsample with a smoothing kernel that
// removes the pixel-grid blockiness of plain bilinear upsample.
//
// Output is BLENDED ADDITIVELY onto the destination via VkPipelineColorBlend
// (srcFactor=ONE, dstFactor=ONE, op=ADD), so each upsample pass contributes
// to the dst image which already holds the next-finer downsample result.
// After the last upsample, mip 0 of the bloom pyramid contains the sum of
// all blurred bright-pass mips and is what `composite.frag` samples.
//
// Inputs:
//   set=0 binding=0: srcImage (smaller mip)
// Push constant:
//   vec2 srcTexel = 1.0 / srcSize
//   float scatter = strength of the upsample blur (0.5..1.0 typical)
//   float _pad
// Output: R16G16B16A16_SFLOAT (RGB + alpha 1)
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D srcImage;

layout(push_constant) uniform UpsamplePush {
    vec2 srcTexel;
    float scatter;  // multiplier on the kernel offsets (1.0 = unit pixel offsets)
    float _pad;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    // Tent kernel:
    //   1 2 1
    //   2 4 2  / 16
    //   1 2 1
    vec2 d = pc.srcTexel * pc.scatter;

    vec3 s00 = texture(srcImage, fragTexCoord + vec2(-d.x,  d.y)).rgb;
    vec3 s10 = texture(srcImage, fragTexCoord + vec2( 0.0,  d.y)).rgb;
    vec3 s20 = texture(srcImage, fragTexCoord + vec2( d.x,  d.y)).rgb;

    vec3 s01 = texture(srcImage, fragTexCoord + vec2(-d.x,  0.0)).rgb;
    vec3 s11 = texture(srcImage, fragTexCoord                   ).rgb;
    vec3 s21 = texture(srcImage, fragTexCoord + vec2( d.x,  0.0)).rgb;

    vec3 s02 = texture(srcImage, fragTexCoord + vec2(-d.x, -d.y)).rgb;
    vec3 s12 = texture(srcImage, fragTexCoord + vec2( 0.0, -d.y)).rgb;
    vec3 s22 = texture(srcImage, fragTexCoord + vec2( d.x, -d.y)).rgb;

    vec3 sum =       s00 + 2.0 * s10 +       s20
              + 2.0 * s01 + 4.0 * s11 + 2.0 * s21
              +       s02 + 2.0 * s12 +       s22;
    sum *= 1.0 / 16.0;

    // Output additively blended (pipeline blend state handles the +).
    outColor = vec4(sum, 1.0);
}
