// lighting.frag — Lighting Pass: PBR Cook-Torrance for point, directional and
// spot lights.
#version 450

// --- G-Buffer texture inputs ---
layout(set = 0, binding = 0) uniform sampler2D gPosition;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;
// Hi-Z pyramid (R32_SFLOAT, mipmapped); only used by debugMode == 5
layout(set = 0, binding = 7) uniform sampler2D gHiZ;
// Directional shadow map (D32_SFLOAT; sampler2DShadow enables hardware PCF)
layout(set = 0, binding = 8) uniform sampler2DShadow uDirShadowMap;
// Directional light-space VP + config params
layout(set = 0, binding = 9) uniform DirShadowUBO {
    mat4  lightViewProj;
    int   enabled;       // 0 = bypass shadow lookup (no directional projection)
    float invMapSize;    // 1.0 / shadowMapResolution, for PCF offsets
    float _pad0;
    float _pad1;
} dirShadow;

// Spot shadow map array (max 4 D32_SFLOAT, sampler2DShadow)
layout(set = 0, binding = 10) uniform sampler2DShadow uSpotShadowMap[4];
// Spot light-space VP array + active slot count + invMapSize
layout(set = 0, binding = 11) uniform SpotShadowUBO {
    mat4  spotLightVP[4];
    ivec4 validSlots;   // x = active slot count (0..4)
    vec4  params;       // x = invMapSize
} spotShadow;

// Point cubemap shadow map array (max 4 D32_SFLOAT, samplerCubeShadow)
layout(set = 0, binding = 12) uniform samplerCubeShadow uPointShadowMap[4];
// Point shadow params: per-slot lightPos.xyz + range.w, plus validSlots
layout(set = 0, binding = 13) uniform PointShadowUBO {
    vec4  lightPosRange[4]; // xyz = world pos, w = range
    ivec4 validSlots;       // x = active slot count (0..4)
    // Unity-style per-light shadow bias multiplier:
    //   params[k].x = normalBiasMul (normal-offset bias)
    //   params[k].y = depthBiasMul (range-aware depth bias)
    //   params[k].zw = reserved
    vec4  params[4];
} pointShadow;

// --- Lighting UBO ---
layout(set = 0, binding = 3) uniform LightingUBO {
    vec3 viewPos;
    int  lightCount;
    int  debugMode;
    int  dirLightCount;
    int  spotLightCount;
    float exposure;
} ubo;

// --- Point light SSBO ---
struct PointLight {
    vec3  position;
    float radius;
    vec3  color;
    float intensity;
    int   shadowSlot;
    float _pad[3];
};
layout(std430, set = 0, binding = 4) readonly buffer PointLightSSBO {
    PointLight pointLights[];
};

// --- Directional light SSBO ---
struct DirectionalLight {
    vec3  direction;
    float intensity;
    vec3  color;
    float _pad;
};
layout(std430, set = 0, binding = 5) readonly buffer DirLightSSBO {
    DirectionalLight dirLights[];
};

// --- Spot light SSBO ---
struct SpotLight {
    vec3  position;
    float radius;
    vec3  direction;
    float intensity;
    vec3  color;
    float innerCos;
    float outerCos;
    int   shadowSlot;
    float _pad[2];
};
layout(std430, set = 0, binding = 6) readonly buffer SpotLightSSBO {
    SpotLight spotLights[];
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

// Directional shadow PCF (4x4 = 16 samples). Projects worldPos/N/L into light
// space and samples a 4x4 grid via sampler2DShadow; hardware bilinear PCF makes
// each sample a 4-tap filter, so 4x4 is effectively 64 taps. Returns 0 = fully
// shadowed, 1 = fully lit; middle values are penumbra.
// Bias strategy: the shadow pass writes front-facing depth with slope-scaled
// depth bias via back-face culling; this normal bias only compensates float
// precision at grazing angles (NdotL ~ 0 -> 0.015) and stays 0 for NdotL ~ 1,
// so a cube flush against a plane never Peter Pans.
float sampleDirShadow(vec3 worldPos, vec3 N, vec3 L) {
    if (dirShadow.enabled == 0) return 1.0;

    // Normal bias is strictly 0 for NdotL ~ 1 to avoid Peter Panning
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float normalBias = 0.015 * (1.0 - NdotL);
    vec3 biasedPos = worldPos + N * normalBias;

    vec4 lightSpace = dirShadow.lightViewProj * vec4(biasedPos, 1.0);
    if (lightSpace.w <= 0.0) return 1.0;
    vec3 projCoord = lightSpace.xyz / lightSpace.w;

    // Vulkan: clip-space xy in [-1,1] maps to UV [0,1]. The shadow pass flips Y
    // (proj[1][1] *= -1) and depth is already [0,1].
    vec2 uv = projCoord.xy * 0.5 + 0.5;
    float currentDepth = projCoord.z;

    // Out of range = fully lit (border color would do this too, but an early
    // return is safer)
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) return 1.0;

    float texel = dirShadow.invMapSize;
    float visibility = 0.0;
    // 4x4 PCF; offsets are in texels
    for (int x = -1; x <= 2; ++x) {
        for (int y = -1; y <= 2; ++y) {
            vec2 off = vec2(float(x) - 0.5, float(y) - 0.5) * texel;
            visibility += texture(uDirShadowMap, vec3(uv + off, currentDepth));
        }
    }
    return visibility / 16.0;
}

// Spot shadow PCF (4x4). Projects worldPos/N/L/slot via spotLightVP[slot]; the
// bias formula matches the directional path. sampler2DShadow array indices must
// be dynamically uniform in Vulkan GLSL, so the switch is unrolled (max 4 slots,
// negligible cost).
float sampleSpotShadow(vec3 worldPos, vec3 N, vec3 L, int slot) {
    if (slot < 0 || slot >= 4) return 1.0;
    if (slot >= spotShadow.validSlots.x) return 1.0;

    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float normalBias = 0.015 * (1.0 - NdotL);
    vec3 biasedPos = worldPos + N * normalBias;

    vec4 lightSpace = spotShadow.spotLightVP[slot] * vec4(biasedPos, 1.0);
    if (lightSpace.w <= 0.0) return 1.0;
    vec3 projCoord = lightSpace.xyz / lightSpace.w;
    vec2 uv = projCoord.xy * 0.5 + 0.5;
    float currentDepth = projCoord.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) return 1.0;

    float texel = spotShadow.params.x;
    float visibility = 0.0;
    // 4x4 PCF — same kernel as sampleDirShadow. sampler2DShadow array sampling
    // needs constant (or dynamically uniform) indices; slot comes from the SSBO
    // and every invocation takes one branch, so it is dyn-uniform.
    if (slot == 0) {
        for (int x = -1; x <= 2; ++x)
            for (int y = -1; y <= 2; ++y) {
                vec2 off = vec2(float(x) - 0.5, float(y) - 0.5) * texel;
                visibility += texture(uSpotShadowMap[0], vec3(uv + off, currentDepth));
            }
    } else if (slot == 1) {
        for (int x = -1; x <= 2; ++x)
            for (int y = -1; y <= 2; ++y) {
                vec2 off = vec2(float(x) - 0.5, float(y) - 0.5) * texel;
                visibility += texture(uSpotShadowMap[1], vec3(uv + off, currentDepth));
            }
    } else if (slot == 2) {
        for (int x = -1; x <= 2; ++x)
            for (int y = -1; y <= 2; ++y) {
                vec2 off = vec2(float(x) - 0.5, float(y) - 0.5) * texel;
                visibility += texture(uSpotShadowMap[2], vec3(uv + off, currentDepth));
            }
    } else {
        for (int x = -1; x <= 2; ++x)
            for (int y = -1; y <= 2; ++y) {
                vec2 off = vec2(float(x) - 0.5, float(y) - 0.5) * texel;
                visibility += texture(uSpotShadowMap[3], vec3(uv + off, currentDepth));
            }
    }
    return visibility / 16.0;
}

// Point cubemap shadow PCF: compareDepth = length(biasedPos - lightPos) / range
// with biasedPos = worldPos + N * normalOffset; samplerCubeShadow takes
// vec4(dir, compareDepth) with dir = biasedPos - lightPos. PCF taps use 20
// symmetric cube directions (LearnOpenGL) to avoid false shadows from
// XY-plane-only offsets on cube faces. Dual bias prevents Peter Panning: the
// normal offset moves the sample start off the caster surface, and the depth
// bias covers D32 quantization + face-derivative error; the shadow pass adds
// back-face culling + slope bias (0.8/1.5) so total bias stays on par with
// spot/dir shadows. Array indices must be dynamically uniform, so the switch is
// unrolled.
const vec3 kPointShadowPCFOffsets[20] = vec3[](
    vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
    vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
    vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
    vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
    vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

float samplePointShadowSlot(samplerCubeShadow tex, vec3 dir, float compareDepth, float diskRadius) {
    // Center tap + 20 cube-symmetric taps (~84 effective taps after hardware PCF)
    float sum = texture(tex, vec4(dir, compareDepth));
    for (int i = 0; i < 20; ++i) {
        sum += texture(tex, vec4(dir + kPointShadowPCFOffsets[i] * diskRadius, compareDepth));
    }
    return sum / 21.0;
}

float samplePointShadow(vec3 worldPos, vec3 N, vec3 L, int slot) {
    if (slot < 0 || slot >= 4) return 1.0;
    if (slot >= pointShadow.validSlots.x) return 1.0;

    vec3  lightPos = pointShadow.lightPosRange[slot].xyz;
    float range    = pointShadow.lightPosRange[slot].w;
    if (range <= 0.0001) return 1.0;

    // Unity-style per-light bias tuning: dual bias + per-slot multiplier.
    //   (1) Normal-offset bias: push the sample start along the receiver normal
    //       by normalOffset, so receivers flush against casters still receive
    //       shadows (same N-offset idea as directional/spot shadows).
    //   (2) Depth bias (shrinks compareDepth): covers D32 quantization and
    //       cubemap face-derivative error at edges; error grows as NdotL -> 0,
    //       so it scales with (1 - NdotL).
    // Base factors (normalOffset = 0.003, depthBiasWorld = 0.012) match Unity
    // URP "Additional Lights -> Shadow Bias" defaults; the per-slot multiplier
    // is exposed to artists/JSON via PointLightComponent.shadow{Normal,Depth}Bias.
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float normalBiasMul = pointShadow.params[slot].x;
    float depthBiasMul  = pointShadow.params[slot].y;

    float normalOffset = (0.003 * (1.0 - NdotL) + 0.0005) * normalBiasMul;
    vec3  biasedPos    = worldPos + N * normalOffset;

    vec3  dir  = biasedPos - lightPos;
    float dist = length(dir);
    if (dist >= range) return 1.0;

    float compareDepth = dist / range;

    float depthBiasWorld = max(0.012 * (1.0 - NdotL), 0.0015) * depthBiasMul;
    compareDepth -= depthBiasWorld / range;
    compareDepth = clamp(compareDepth, 0.0, 1.0);

    // PCF disk radius: too large softens shadow edges and hides acne but leaks
    // light; too small keeps acne. 0.02 is ~1 texel at 1K cubemap resolution.
    float diskRadius = 0.02;
    if (slot == 0) return samplePointShadowSlot(uPointShadowMap[0], dir, compareDepth, diskRadius);
    else if (slot == 1) return samplePointShadowSlot(uPointShadowMap[1], dir, compareDepth, diskRadius);
    else if (slot == 2) return samplePointShadowSlot(uPointShadowMap[2], dir, compareDepth, diskRadius);
    else                return samplePointShadowSlot(uPointShadowMap[3], dir, compareDepth, diskRadius);
}

const float PI = 3.14159265359;

// PBR functions
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float attenuation(float dist, float radius) {
    float x = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
    return x * x / (dist * dist + 1.0);
}

// Generic BRDF: given L direction and radiance, returns the lighting contribution
vec3 calcBRDF(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metallic, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);

    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular     = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    vec4 posSample = texture(gPosition, fragTexCoord);
    vec3 worldPos  = posSample.rgb;
    vec4 normalSample = texture(gNormal, fragTexCoord);
    vec3 normal    = normalSample.rgb;
    float roughness = normalSample.a;
    vec4 albedoSample = texture(gAlbedo, fragTexCoord);
    vec3 albedo    = albedoSample.rgb;
    float metallic = albedoSample.a;

    // Background pixels (debugMode == 5 must still show Hi-Z here)
    if (posSample.a == 0.0 && ubo.debugMode != 5) {
        outColor = vec4(0.02, 0.02, 0.04, 1.0);
        return;
    }

    // G-Buffer debug modes
    if (ubo.debugMode == 1) { outColor = vec4(worldPos * 0.5 + 0.5, 1.0); return; }
    if (ubo.debugMode == 2) { outColor = vec4(normal * 0.5 + 0.5, 1.0); return; }
    if (ubo.debugMode == 3) { outColor = vec4(albedo, 1.0); return; }
    if (ubo.debugMode == 4) {
        float d = length(worldPos - ubo.viewPos) / 10.0;
        outColor = vec4(vec3(1.0 - clamp(d, 0.0, 1.0)), 1.0);
        return;
    }
    // debugMode == 5: visualize Hi-Z mip 0. Hi-Z stores R32_SFLOAT depth
    // (0=near, 1=far); sampling mip 0 shows a depth copy from the compute
    // pyramid, verifying the build (similar to debugMode 4).
    if (ubo.debugMode == 5) {
        float h = textureLod(gHiZ, fragTexCoord, 0.0).r;
        // Same mapping as debugMode 4: near bright, far dark
        outColor = vec4(vec3(1.0 - clamp(h, 0.0, 1.0)), 1.0);
        return;
    }

    vec3 N = normalize(normal);
    vec3 V = normalize(ubo.viewPos - worldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 ambient = vec3(0.03) * albedo;
    vec3 Lo = vec3(0.0);

    // === Point lights ===
    for (int i = 0; i < ubo.lightCount; i++) {
        vec3  L    = pointLights[i].position - worldPos;
        float dist = length(L);
        if (dist > pointLights[i].radius) continue;
        L = normalize(L);
        float atten = attenuation(dist, pointLights[i].radius);
        vec3 radiance = pointLights[i].color * pointLights[i].intensity * atten;
        // If shadowSlot >= 0, apply cubemap PCF attenuation
        float pointShadowFactor = 1.0;
        if (pointLights[i].shadowSlot >= 0) {
            pointShadowFactor = samplePointShadow(worldPos, N, L, pointLights[i].shadowSlot);
        }
        Lo += calcBRDF(N, V, L, radiance, albedo, metallic, roughness, F0) * pointShadowFactor;
    }

    // === Directional lights (no attenuation) ===
    // Only the first directional light uses sampler2DShadow PCF; the others are
    // unaffected by the shadow map (matches Unity URP main light). The shadow
    // factor depends on the first light's L direction for normal-based bias.
    float dirShadowFactor = 1.0;
    if (ubo.dirLightCount > 0) {
        vec3 L0 = normalize(-dirLights[0].direction);
        dirShadowFactor = sampleDirShadow(worldPos, N, L0);
    }
    for (int i = 0; i < ubo.dirLightCount; i++) {
        vec3 L = normalize(-dirLights[i].direction);  // negated light direction = fragment-to-light direction
        vec3 radiance = dirLights[i].color * dirLights[i].intensity;
        float shadow = (i == 0) ? dirShadowFactor : 1.0;
        Lo += calcBRDF(N, V, L, radiance, albedo, metallic, roughness, F0) * shadow;
    }

    // === Spot lights (distance + cone attenuation) ===
    for (int i = 0; i < ubo.spotLightCount; i++) {
        vec3  L    = spotLights[i].position - worldPos;
        float dist = length(L);
        if (dist > spotLights[i].radius) continue;
        L = normalize(L);

        // Cone attenuation: angle between fragment direction and spot direction
        float theta   = dot(L, normalize(-spotLights[i].direction));
        float epsilon = spotLights[i].innerCos - spotLights[i].outerCos;
        float spotFactor = clamp((theta - spotLights[i].outerCos) / epsilon, 0.0, 1.0);

        float atten = attenuation(dist, spotLights[i].radius);
        vec3 radiance = spotLights[i].color * spotLights[i].intensity * atten * spotFactor;
        // If shadowSlot >= 0, apply spot shadow PCF attenuation
        float spotShadowFactor = 1.0;
        if (spotLights[i].shadowSlot >= 0) {
            spotShadowFactor = sampleSpotShadow(worldPos, N, L, spotLights[i].shadowSlot);
        }
        Lo += calcBRDF(N, V, L, radiance, albedo, metallic, roughness, F0) * spotShadowFactor;
    }

    vec3 color = ambient + Lo;
    // Lighting pass writes linear HDR (FP16). Exposure / tone mapping / gamma
    // are applied by the Composite Pass after Light Volume additive blending
    // (per-light correction in non-linear space was wrong). RT format is
    // R16G16B16A16_SFLOAT; downstream Composite consumes it.
    outColor = vec4(color, 1.0);
}
