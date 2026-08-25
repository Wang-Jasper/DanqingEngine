// ============================================================================
// Light.h — light data structures for the deferred lighting pass. PointLight /
// DirectionalLight / SpotLight use std430 SSBO layout (vec3 16-byte aligned,
// struct size a multiple of 16); LightingUBO uses std140.
// ============================================================================
#pragma once

#include <glm/glm.hpp>
#include <vector>

// ============================================================================
// PointLight — single point light (std430 SSBO layout)
// ============================================================================
struct PointLight
{
    // World-space position (alignas(16) for std430 vec3 alignment)
    alignas(16) glm::vec3 position;
    // Attenuation radius; fragments beyond it are unaffected
    float radius;
    // Color (RGB, normalized [0,1])
    alignas(16) glm::vec3 color;
    // Intensity; multiplied with color for the final contribution (HDR > 1.0 allowed)
    float intensity;
    // Point shadow slot (-1 = none; 0..3 = index into uPointShadowMap[N]).
    // Written each frame by PointShadowSelectionSystem, sorted by length(color)*intensity.
    // Layout must match PointLight in lighting.frag / light_volume_point.{vert,frag}.
    int shadowSlot;
    // Pad so the std430 struct size stays a multiple of 16 (sizeof = 48).
    float _pad[3];
};

// ============================================================================
// DirectionalLight — directional light (std430 SSBO layout)
// ============================================================================
struct DirectionalLight
{
    alignas(16) glm::vec3 direction; // Light direction (from the light, normalized)
    float intensity;
    alignas(16) glm::vec3 color;
    float _pad;
};

// ============================================================================
// SpotLight — spot light (std430 SSBO layout)
// ============================================================================
struct SpotLight
{
    alignas(16) glm::vec3 position;
    float radius;
    alignas(16) glm::vec3 direction; // Spot direction (normalized)
    float intensity;
    alignas(16) glm::vec3 color;
    float innerCos; // cos(innerAngle); edge of the fully lit region
    float outerCos; // cos(outerAngle); edge where falloff reaches zero
    // Spot shadow slot (-1 = none; 0..3 = index into uSpotShadowMap[N]).
    // Written each frame by SpotShadowSelectionSystem, sorted by length(color)*intensity.
    // Layout must match SpotLight in lighting.frag / light_volume_spot.frag (replaces the old _pad[0]).
    int shadowSlot;
    float _pad[2];
};

// ============================================================================
// LightingUBO — per-frame uniform buffer for the lighting pass
// ============================================================================
struct LightingUBO
{
    alignas(16) glm::vec3 viewPos;
    int lightCount; // Point light count
    int debugMode;
    int dirLightCount;  // Directional light count
    int spotLightCount; // Spot light count
    float exposure;     // HDR exposure (applied to Lo/ambient before Reinhard)
};

// ============================================================================
// createDefaultLights() — the default scene's 4 point lights (warm key, cool
// fill, red underlight, green side). Returns a vector uploadable to the SSBO.
// ============================================================================
inline std::vector<PointLight> createDefaultLights()
{
    return {
        // {position, radius, color, intensity}
        {{2.0f, 2.0f, 2.0f}, 8.0f, {1.0f, 0.9f, 0.8f}, 3.0f},   // Warm key (front-right-top)
        {{-2.0f, 1.5f, -1.0f}, 6.0f, {0.3f, 0.5f, 1.0f}, 2.0f}, // Cool fill (back-left-top)
        {{0.0f, -1.0f, 2.0f}, 5.0f, {1.0f, 0.3f, 0.2f}, 1.5f},  // Red underlight (front-bottom)
        {{1.5f, 0.5f, -2.0f}, 5.0f, {0.2f, 1.0f, 0.3f}, 1.5f},  // Green side (back-right)
    };
}
