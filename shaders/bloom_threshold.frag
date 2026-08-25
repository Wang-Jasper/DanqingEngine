// ============================================================================
// bloom_threshold.frag — Phase 11.9.5 Bloom (bright pass extraction)
// ============================================================================
// Reads the HDR lighting RT and outputs only the parts that exceed the
// `threshold` luminance. A `softKnee` value smooths the transition near the
// threshold to avoid harsh quantization artifacts. This matches Unity URP's
// PostProcessing v2 / HDRP Bloom curve.
//
// Inputs:
//   set=0 binding=0: hdrImage (R16G16B16A16_SFLOAT)
// Push constant:
//   float threshold      // pixels with luminance < threshold are clamped to 0
//   float softKnee       // 0..1, soft transition width as fraction of threshold
//   float _pad0, _pad1
// Output: R16G16B16A16_SFLOAT (only RGB used, alpha=1)
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D hdrImage;

layout(push_constant) uniform ThresholdPush {
    float threshold;
    float softKnee;
    float _pad0;
    float _pad1;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 hdr = texture(hdrImage, fragTexCoord).rgb;
    float l = luma(hdr);

    // Soft knee curve (Unity HDRP Bloom): smoothly fade in around the threshold.
    //   knee  = threshold * softKnee + 1e-5
    //   soft  = clamp(l - threshold + knee, 0, 2*knee)
    //   soft  = soft * soft / (4 * knee)   // quadratic ramp from 0
    //   contribution = max(soft, l - threshold) / max(l, 1e-5)
    float knee = pc.threshold * pc.softKnee + 1e-5;
    float soft = clamp(l - pc.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    float contribution = max(soft, l - pc.threshold);
    contribution /= max(l, 1e-5);

    outColor = vec4(hdr * contribution, 1.0);
}
