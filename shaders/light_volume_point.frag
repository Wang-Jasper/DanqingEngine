// light_volume_point.frag — per-light point-light volume pass. One dispatch per
// proxy sphere: read the G-Buffer, discard background / out-of-radius pixels,
// then compute PBR (same equations as lighting.frag) and additively blend.
#version 450

// --- G-Buffer inputs (shares set 0 with the lighting pass) ---
layout(set = 0, binding = 0) uniform sampler2D gPosition;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;

layout(set = 0, binding = 3) uniform LightingUBO {
    vec3 viewPos;
    int  lightCount;
    int  debugMode;
    int  dirLightCount;
    int  spotLightCount;
    float exposure;
} ubo;

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

// Point cubemap shadows, kept strictly in sync with lighting.frag: if this
// light writes a shadow map (shadowSlot >= 0), attenuate this pass's Lo.
layout(set = 0, binding = 12) uniform samplerCubeShadow uPointShadowMap[4];
layout(set = 0, binding = 13) uniform PointShadowUBO {
    vec4  lightPosRange[4];
    ivec4 validSlots;
    // params[k].x = normalBiasMul, params[k].y = depthBiasMul.
    vec4  params[4];
} pointShadow;

layout(location = 0) in flat int vInstanceID;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

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

// PCF cubemap shadow sampling (logic aligned with lighting.frag)
const vec3 kPointShadowPCFOffsets[20] = vec3[](
    vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
    vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
    vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
    vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
    vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);
float samplePointShadowSlot(samplerCubeShadow tex, vec3 dir, float compareDepth, float diskRadius) {
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

    // Dual bias, strictly matching lighting.frag (per-slot multiplier).
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float normalBiasMul = pointShadow.params[slot].x;
    float depthBiasMul  = pointShadow.params[slot].y;

    float normalOffset = (0.003 * (1.0 - NdotL) + 0.0005) * normalBiasMul;
    vec3 biasedPos = worldPos + N * normalOffset;

    vec3 dir  = biasedPos - lightPos;
    float dist = length(dir);
    if (dist >= range) return 1.0;

    float compareDepth = dist / range;

    float biasWorld = max(0.012 * (1.0 - NdotL), 0.0015) * depthBiasMul;
    compareDepth -= biasWorld / range;
    compareDepth = clamp(compareDepth, 0.0, 1.0);

    float diskRadius = 0.02;
    if (slot == 0) return samplePointShadowSlot(uPointShadowMap[0], dir, compareDepth, diskRadius);
    else if (slot == 1) return samplePointShadowSlot(uPointShadowMap[1], dir, compareDepth, diskRadius);
    else if (slot == 2) return samplePointShadowSlot(uPointShadowMap[2], dir, compareDepth, diskRadius);
    else                return samplePointShadowSlot(uPointShadowMap[3], dir, compareDepth, diskRadius);
}

void main() {
    // Screen UV: gl_FragCoord.xy is in [0, viewport size]; divide by the
    // G-Buffer size obtained via textureSize
    vec2 texSize = vec2(textureSize(gPosition, 0));
    vec2 uv = gl_FragCoord.xy / texSize;

    vec4 posSample = texture(gPosition, uv);
    if (posSample.a == 0.0) discard; // background

    vec3 worldPos = posSample.rgb;
    PointLight pl = pointLights[vInstanceID];

    vec3 L = pl.position - worldPos;
    float dist = length(L);
    if (dist > pl.radius) discard; // outside the proxy sphere

    vec4 normalSample = texture(gNormal, uv);
    vec4 albedoSample = texture(gAlbedo, uv);
    vec3 normal    = normalSample.rgb;
    float roughness = normalSample.a;
    vec3 albedo    = albedoSample.rgb;
    float metallic = albedoSample.a;

    L = normalize(L);
    vec3 N = normalize(normal);
    vec3 V = normalize(ubo.viewPos - worldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float atten = attenuation(dist, pl.radius);
    vec3 radiance = pl.color * pl.intensity * atten;

    vec3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    float NdotL = max(dot(N, L), 0.0);
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Point shadow PCF: if this light wrote a shadow map, attenuate Lo
    float pointShadowFactor = 1.0;
    if (pl.shadowSlot >= 0) {
        pointShadowFactor = samplePointShadow(worldPos, N, L, pl.shadowSlot);
    }
    Lo *= pointShadowFactor;

    // Linear HDR additive output; exposure + tone mapping + gamma are applied
    // by the Composite Pass after all light blending. Additive blend on
    // R16G16B16A16_SFLOAT is the only mathematically correct way to combine
    // multiple lights. Alpha must be 0 so additive blend (srcA=ZERO, dstA=ONE)
    // preserves dst.a written by the lighting pass.
    outColor = vec4(Lo, 0.0);
}
