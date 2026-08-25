// ============================================================================
// ssao.frag — Phase 11.9.4-improve  view-space SSAO (URP-equivalent)
// ============================================================================
// View-space hemisphere SSAO. Reads:
//   set=0 binding=0: gPosition  (RGB = WORLD-space position, A = mask)
//   set=0 binding=1: gNormal    (RGB = WORLD-space normal,   A = roughness)
//   set=0 binding=2: SsaoUBO    (camera view + proj matrices, screen size)
//
// For every visible pixel we:
//   1. Read the world-space position P and normal N from the GBuffer.
//   2. Transform (P, N) into view space using `view`.
//   3. Generate NUM_SAMPLES tangent-space hemisphere samples whose direction
//      is rotated by an interleaved-gradient-noise (IGN) angle that varies
//      smoothly across pixels — much better spatial distribution than the
//      old per-pixel hash, dramatically reducing noise BEFORE blurring.
//   4. For each sample point P_view + tbn * sample * radius, project it back
//      to clip space with `proj`, derive screen UV, and sample gPosition at
//      that UV. The depth (view-space Z) of the occluder vs. the sample tells
//      us whether the sample is occluded.
//   5. Range-falloff weights distant occluders to avoid haloing.
//
// Output: R8_UNORM AO factor in [0,1] (1 = fully lit, 0 = fully occluded).
// A separable bilateral blur (ssao_blur.frag) cleans up the remaining noise;
// the composite pass multiplies the blurred AO onto the lit HDR.
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D gPosition;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform SsaoUBO {
    mat4 view;
    mat4 proj;
    vec2 screenSize;   // xy = render-target pixel size (half-res in our pipe)
    vec2 _pad;
} ubo;

layout(push_constant) uniform SsaoPush {
    float radius;     // view-space sampling radius (world units)
    float intensity;  // 0 = disabled, 1 = full effect
    float bias;       // depth bias to suppress self-occlusion
    int   _pad;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out float outAO;

const int NUM_SAMPLES = 16;

// 16-sample Halton/Hammersley-style fixed kernel in tangent space.
// All vectors are inside the +Z hemisphere; lengths are biased toward small
// values (squared distribution) so dense near-surface samples dominate.
const vec3 KERNEL[16] = vec3[](
    vec3( 0.5381,  0.1856, 0.4319),
    vec3( 0.1379,  0.2486, 0.4430),
    vec3( 0.3371,  0.5679, 0.0057),
    vec3(-0.6999, -0.0451, 0.7140),
    vec3( 0.0689, -0.1598, 0.8547),
    vec3( 0.0560,  0.0069, 0.1843),
    vec3(-0.0146,  0.1402, 0.0762),
    vec3( 0.0100, -0.1924, 0.2790),
    vec3(-0.3577, -0.5301, 0.4358),
    vec3(-0.3169,  0.1063, 0.0158),
    vec3( 0.0103, -0.5869, 0.0046),
    vec3(-0.0897, -0.4940, 0.3287),
    vec3( 0.7119, -0.0154, 0.0918),
    vec3(-0.0533,  0.0596, 0.5411),
    vec3( 0.0352, -0.0631, 0.5460),
    vec3(-0.4776,  0.2847, 0.0271)
);

// Interleaved Gradient Noise (Jorge Jimenez, Crysis 3). Returns ~uniform
// pseudo-random in [0,1) that varies SMOOTHLY across nearby pixels — exactly
// what bilateral-blurred SSAO wants. Frame index keeps the pattern stable.
float ign(vec2 pixel) {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

void main() {
    vec4 posSample = texture(gPosition, fragTexCoord);
    if (posSample.a == 0.0) {
        outAO = 1.0; // sky / unwritten — never darken
        return;
    }
    vec3 P_world = posSample.rgb;
    vec3 N_world = normalize(texture(gNormal, fragTexCoord).rgb);

    // To view space.
    vec3 P_view = (ubo.view * vec4(P_world, 1.0)).xyz;
    vec3 N_view = normalize((ubo.view * vec4(N_world, 0.0)).xyz);

    // Build a TBN that orients the kernel's +Z hemisphere along N_view, then
    // randomly rotates around N_view with an IGN-derived angle so each pixel
    // gets a slightly different sampling pattern (key to bilateral blur
    // recovering smooth AO).
    vec2 pixel = fragTexCoord * ubo.screenSize;
    float angle = ign(pixel) * 6.2831853;
    vec3 randomVec = vec3(cos(angle), sin(angle), 0.0);
    // Gram-Schmidt
    vec3 tangent = normalize(randomVec - N_view * dot(randomVec, N_view));
    vec3 bitangent = cross(N_view, tangent);
    mat3 TBN = mat3(tangent, bitangent, N_view);

    float occlusion = 0.0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        // Transform tangent-space sample to view space.
        vec3 sampleVS = TBN * KERNEL[i];
        sampleVS = P_view + sampleVS * pc.radius;

        // Project to clip → NDC → UV.
        vec4 clip = ubo.proj * vec4(sampleVS, 1.0);
        if (clip.w <= 0.0) continue;
        vec3 ndc = clip.xyz / clip.w;
        // GBuffer is rendered with Vulkan's flipped Y viewport, so the same
        // mapping that lighting.frag uses applies: NDC (-1..1) → UV (0..1)
        // with `vec2(0.5, 0.5)` shift; Y is NOT flipped here (Vulkan NDC.y
        // already matches UV.y after the viewport flip in geometry pass).
        vec2 sUV = ndc.xy * 0.5 + 0.5;
        if (sUV.x < 0.0 || sUV.x > 1.0 || sUV.y < 0.0 || sUV.y > 1.0) continue;

        // Sample the occluder at sUV and bring it into view space.
        vec4 occluderWorld = texture(gPosition, sUV);
        if (occluderWorld.a == 0.0) continue; // sky there
        float occluderViewZ = (ubo.view * vec4(occluderWorld.rgb, 1.0)).z;

        // In view space (Vulkan / RH convention), the camera looks down -Z,
        // so a SMALLER (more negative) z = closer to camera = potential
        // occluder. We compare the sample's depth (sampleVS.z) against the
        // GBuffer's depth at that UV.
        float sampleDepth = sampleVS.z;
        float depthDelta = occluderViewZ - sampleDepth;

        // depthDelta > bias means the GBuffer pixel is in FRONT of the
        // hemisphere sample → occluder.
        // Range falloff: only count occluders within `radius` of P_view to
        // prevent haloing around silhouettes.
        float rangeCheck = smoothstep(0.0, 1.0,
                              pc.radius / max(abs(P_view.z - occluderViewZ), 0.0001));
        occlusion += (depthDelta > pc.bias ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion /= float(NUM_SAMPLES);
    float ao = 1.0 - clamp(occlusion, 0.0, 1.0);
    outAO = mix(1.0, ao, pc.intensity);
}
