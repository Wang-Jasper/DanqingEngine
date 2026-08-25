// light_volume_spot.frag — spot-light volume pass. Same structure as the point
// volume; adds cone falloff (aligned with the lighting.frag spot branch) and
// discards fragments outside the cone (spotFactor == 0).
#version 450

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

// Spot shadow map array (max 4 D32_SFLOAT, sampler2DShadow)
layout(set = 0, binding = 10) uniform sampler2DShadow uSpotShadowMap[4];
// Spot light-space VP array + active slot count + invMapSize
layout(set = 0, binding = 11) uniform SpotShadowUBO {
    mat4  spotLightVP[4];
    ivec4 validSlots;   // x = active slot count (0..4)
    vec4  params;       // x = invMapSize
} spotShadow;

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

// Spot shadow PCF (logic aligned with lighting.frag). sampler2DShadow array
// indices must be dynamically uniform, so the switch is unrolled.
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

void main() {
    vec2 texSize = vec2(textureSize(gPosition, 0));
    vec2 uv = gl_FragCoord.xy / texSize;

    vec4 posSample = texture(gPosition, uv);
    if (posSample.a == 0.0) discard;

    vec3 worldPos = posSample.rgb;
    SpotLight sl = spotLights[vInstanceID];

    vec3 L = sl.position - worldPos;
    float dist = length(L);
    if (dist > sl.radius) discard;
    L = normalize(L);

    // Cone falloff: fragment-to-light direction vs spot direction
    float theta = dot(L, normalize(-sl.direction));
    float epsilon = sl.innerCos - sl.outerCos;
    float spotFactor = clamp((theta - sl.outerCos) / epsilon, 0.0, 1.0);
    if (spotFactor <= 0.0) discard; // outside the cone

    vec4 normalSample = texture(gNormal, uv);
    vec4 albedoSample = texture(gAlbedo, uv);
    vec3 normal    = normalSample.rgb;
    float roughness = normalSample.a;
    vec3 albedo    = albedoSample.rgb;
    float metallic = albedoSample.a;

    vec3 N = normalize(normal);
    vec3 V = normalize(ubo.viewPos - worldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float atten = attenuation(dist, sl.radius);
    vec3 radiance = sl.color * sl.intensity * atten * spotFactor;

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

    // Spot shadow PCF: if this light wrote a shadow map, attenuate Lo
    float spotShadowFactor = 1.0;
    if (sl.shadowSlot >= 0) {
        spotShadowFactor = sampleSpotShadow(worldPos, N, L, sl.shadowSlot);
    }
    Lo *= spotShadowFactor;


    // Linear HDR additive output; exposure + tone mapping + gamma are applied
    // by the Composite Pass after all light blending. Alpha must be 0 so
    // additive blend (srcA=ZERO, dstA=ONE) preserves dst.a written by the
    // lighting pass.
    outColor = vec4(Lo, 0.0);
}
