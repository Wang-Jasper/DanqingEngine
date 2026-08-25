// ============================================================================
// fxaa.frag — Phase 11.9.3 FXAA Pass
// ============================================================================
// Simplified FXAA based on Timothy Lottes "FXAA WhitePaper" (NVIDIA, 2009)
// and Geeks3D's FXAA 3.11 PRESET 12 ~ console quality.
//
// Edge detection in PERCEPTUAL luma space (Rec.709 weights on sqrt(rgb), i.e.
// a cheap gamma-2.0 decode of the linear sample we get from the sRGB-format
// `srcImage`). This is required for Lottes' EDGE_MIN / EDGE_MAX_THR thresholds
// to behave as designed; running them on raw linear luma collapses the dynamic
// range of mid-tones and causes most edges to be classified as flat regions
// (FXAA effectively becomes a no-op).
//
// 1 sample of long-edge gradient, 2 secondary samples along the gradient,
// 4-tap subpixel quality. Output is sampled in normalized UV from `srcImage`
// (kept in LINEAR space) and written to the LDR destination — destination is
// sRGB so the hardware encoder takes care of gamma on store.
//
// Push constant:
//   vec2  invResolution  — 1/textureSize(srcImage)
//   int   enableFxaa     — 0 = passthrough (source unchanged), 1 = FXAA
//   float edgeThreshold  — relative-luma threshold (Lottes EDGE_MAX_THR).
//                          0.063 = extreme (PRESET 39), 0.166 = default
//                          (PRESET 12 console quality). Lower = more pixels
//                          treated as edges → stronger AA.
//   float subpixelBlend  — final mix factor between FXAA-smoothed and raw
//                          color. 0 disables visible smoothing, 1 = full
//                          FXAA, Lottes default 0.75.
//   int   debugShowEdges — 0 = normal output, 1 = paint detected edges red
//                          so the user can see exactly where FXAA is acting.
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D srcImage;

layout(push_constant) uniform FxaaPush {
    vec2  invResolution;  // x = 1/width, y = 1/height
    int   enableFxaa;
    int   _pad0;
    float edgeThreshold;  // relative-luma cutoff (0.063..0.250)
    float subpixelBlend;  // 0..1 final blend strength
    int   debugShowEdges; // 0/1 — paint detected edges red for diagnostics
    int   _pad1;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

// FXAA must do edge detection in PERCEPTUAL (gamma) space — Lottes' thresholds
// (EDGE_MIN=0.0312, EDGE_MAX_THR=0.125) assume sRGB-encoded luma. The source
// image is a sRGB-format texture, so the hardware sampler returns LINEAR rgb;
// we therefore apply a cheap gamma-2.0 approximation (sqrt) before computing
// luma. The final color sample is left in linear space because the output
// attachment is also sRGB and the hardware re-encodes on store.
float luma(vec3 cLinear) { return dot(sqrt(max(cLinear, vec3(0.0))), vec3(0.299, 0.587, 0.114)); }

void main() {
    vec3 colorCenter = texture(srcImage, fragTexCoord).rgb;

    // Passthrough — used when the user disables FXAA via the UI.
    if (pc.enableFxaa == 0) {
        outColor = vec4(colorCenter, 1.0);
        return;
    }

    // ------ FXAA edge detection in luma space ----------------------------
    float lumaC  = luma(colorCenter);
    float lumaN  = luma(textureOffset(srcImage, fragTexCoord, ivec2( 0, -1)).rgb);
    float lumaS  = luma(textureOffset(srcImage, fragTexCoord, ivec2( 0,  1)).rgb);
    float lumaW  = luma(textureOffset(srcImage, fragTexCoord, ivec2(-1,  0)).rgb);
    float lumaE  = luma(textureOffset(srcImage, fragTexCoord, ivec2( 1,  0)).rgb);

    float lumaMin = min(lumaC, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    float lumaMax = max(lumaC, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    float range   = lumaMax - lumaMin;

    // Skip pixels in flat regions (cheap early-out, ~50% of pixels).
    // EDGE_MIN follows Lottes' guidance of (EDGE_MAX_THR * 0.25) so that the
    // dark-pixel floor scales with the user-driven quality slider.
    float edgeMaxThr = pc.edgeThreshold;
    float edgeMin    = edgeMaxThr * 0.25;
    if (range < max(edgeMin, lumaMax * edgeMaxThr)) {
        outColor = vec4(colorCenter, 1.0);
        return;
    }

    // Diagnostic mode — paint pixels classified as "edge" red so the user can
    // visually inspect where FXAA is actually doing work. Toggle from the UI.
    if (pc.debugShowEdges != 0) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // ------ Edge orientation (horizontal vs vertical) --------------------
    float lumaNW = luma(textureOffset(srcImage, fragTexCoord, ivec2(-1, -1)).rgb);
    float lumaNE = luma(textureOffset(srcImage, fragTexCoord, ivec2( 1, -1)).rgb);
    float lumaSW = luma(textureOffset(srcImage, fragTexCoord, ivec2(-1,  1)).rgb);
    float lumaSE = luma(textureOffset(srcImage, fragTexCoord, ivec2( 1,  1)).rgb);

    float edgeH = abs(-2.0 * lumaW + lumaNW + lumaSW)
                + abs(-2.0 * lumaC + lumaN  + lumaS) * 2.0
                + abs(-2.0 * lumaE + lumaNE + lumaSE);
    float edgeV = abs(-2.0 * lumaN + lumaNW + lumaNE)
                + abs(-2.0 * lumaC + lumaW  + lumaE) * 2.0
                + abs(-2.0 * lumaS + lumaSW + lumaSE);
    bool isHorizontal = edgeH >= edgeV;

    // ------ Subpixel offset along the edge gradient ----------------------
    float luma1 = isHorizontal ? lumaN : lumaW;
    float luma2 = isHorizontal ? lumaS : lumaE;
    float gradient1 = luma1 - lumaC;
    float gradient2 = luma2 - lumaC;
    bool is1Steepest = abs(gradient1) >= abs(gradient2);
    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    float stepLen = isHorizontal ? pc.invResolution.y : pc.invResolution.x;
    float lumaLocal;
    if (is1Steepest) {
        stepLen   = -stepLen;
        lumaLocal = 0.5 * (lumaC + luma1);
    } else {
        lumaLocal = 0.5 * (lumaC + luma2);
    }

    vec2 currentUv = fragTexCoord;
    if (isHorizontal) currentUv.y += stepLen * 0.5;
    else              currentUv.x += stepLen * 0.5;

    // Two-tap edge walk (sufficient for console-quality preset).
    vec2 offset = isHorizontal ? vec2(pc.invResolution.x, 0.0)
                                : vec2(0.0, pc.invResolution.y);
    vec2 uv1 = currentUv - offset;
    vec2 uv2 = currentUv + offset;

    float lumaEnd1 = luma(texture(srcImage, uv1).rgb) - lumaLocal;
    float lumaEnd2 = luma(texture(srcImage, uv2).rgb) - lumaLocal;
    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    float dist1 = isHorizontal ? (fragTexCoord.x - uv1.x) : (fragTexCoord.y - uv1.y);
    float dist2 = isHorizontal ? (uv2.x - fragTexCoord.x) : (uv2.y - fragTexCoord.y);
    bool dirIs1 = dist1 < dist2;
    float distFinal = min(dist1, dist2);
    float edgeLen = dist1 + dist2;
    float pixelOffset = -distFinal / edgeLen + 0.5;

    bool lumaCSmaller = lumaC < lumaLocal;
    bool correctVariation = ((dirIs1 ? lumaEnd1 : lumaEnd2) < 0.0) != lumaCSmaller;
    float finalOffset = correctVariation ? pixelOffset : 0.0;

    vec2 finalUv = fragTexCoord;
    if (isHorizontal) finalUv.y += finalOffset * stepLen;
    else              finalUv.x += finalOffset * stepLen;

    // Blend the FXAA-smoothed color back with the original linear sample
    // according to the user-driven sub-pixel strength. With subpixelBlend = 1
    // we get the full Lottes result; with 0 we keep the un-smoothed pixel
    // even on detected edges (useful for A/B comparison in the UI).
    vec3 smoothed = texture(srcImage, finalUv).rgb;
    vec3 outRgb   = mix(colorCenter, smoothed, clamp(pc.subpixelBlend, 0.0, 1.0));
    outColor = vec4(outRgb, 1.0);
}
