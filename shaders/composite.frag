// ============================================================================
// composite.frag — Phase 11.9.1 / 11.9.2 / 11.9.4 / 11.9.5 Composite Pass
// ============================================================================
// Reads the linear HDR lighting RT (FP16) and writes the LDR sRGB swap-chain-
// adjacent target (B8G8R8A8_SRGB). Applies, in order:
//   1. SSAO modulation                           (binding 1, R8 AO)
//   2. Bloom additive contribution               (binding 2, FP16 bloom mip0) — Phase 11.9.5
//   3. exposure                                  (push constant)
//   4. tone mapping  (Linear / Reinhard / ACES) (push constant)
//   5. linear color is written to a sRGB attachment, so the hardware does
//      gamma 2.2 encoding for free — DO NOT pow(1/2.2) here.
//
// `ldrImage` is the same handle as `EditorUI::offscreenImage` and is bound to
// `ImGui::Image()` inside the Viewport panel.
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D hdrImage;
// Phase 11.9.4: SSAO factor (R8_UNORM in [0,1]; 1 = fully lit). Always bound;
// when SSAO is disabled the texture is filled with constant 1.0 (cheap no-op
// in shader, no need for #ifdef).
layout(set = 0, binding = 1) uniform sampler2D aoImage;
// Phase 11.9.5: bloom pyramid mip 0 (R16G16B16A16_SFLOAT). After the bloom
// upsample chain, mip 0 holds the accumulated blurred bright-pass. Always
// bound; when bloomIntensity == 0 the contribution is multiplied out.
layout(set = 0, binding = 2) uniform sampler2D bloomImage;

layout(push_constant) uniform CompositePush {
    float exposure;       // multiplied onto linear HDR before tone mapping
    int   tonemapMode;    // 0 = Linear, 1 = Reinhard, 2 = ACES Filmic (Narkowicz)
    float bloomIntensity; // 0 = bloom disabled, 1 = full strength (URP default ~0.3)
    int   _pad1;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

vec3 tonemap_reinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

// Krzysztof Narkowicz ACES Filmic (single-shader approximation).
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 tonemap_aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdrImage, fragTexCoord).rgb;
    // Phase 11.9.4: apply screen-space AO (lit value × ao). Modulating the lit
    // result is a slight simplification vs. AO-only-on-ambient (URP/Built-in
    // do the latter), but visual difference is small for typical scenes and
    // it keeps lighting.frag completely untouched.
    float ao = texture(aoImage, fragTexCoord).r;
    hdr *= ao;

    // Phase 11.9.5: add bloom contribution BEFORE exposure + tonemap so high
    // bloom pixels participate in the same tone-mapping curve as the rest of
    // the scene (matches Unity URP / HDRP behavior).
    vec3 bloom = texture(bloomImage, fragTexCoord).rgb;
    hdr += bloom * pc.bloomIntensity;

    hdr *= pc.exposure;

    vec3 ldr;
    if (pc.tonemapMode == 1) {
        ldr = tonemap_reinhard(hdr);
    } else if (pc.tonemapMode == 2) {
        ldr = tonemap_aces(hdr);
    } else {
        // Linear — clamp to [0,1] so the sRGB encoder does not see negatives.
        ldr = clamp(hdr, 0.0, 1.0);
    }

    // Output target is sRGB; hardware applies linear→sRGB encoding on store.
    outColor = vec4(ldr, 1.0);
}
