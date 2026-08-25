#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <memory>

#include <entt/entt.hpp>
#include "core/Allocator.h"

// Shared GPU mesh buffers (migrated from SceneNode.h).
struct MeshReference
{
    AllocatedBuffer vertexBuffer{};
    AllocatedBuffer indexBuffer{};
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0; // Actual vertex count; used by Scene Stats.

    // Mesh-local bounds computed from vertices at load; BoxCollider auto-fits to this.
    glm::vec3 boundsCenter = glm::vec3(0.0f);
    glm::vec3 boundsExtents = glm::vec3(0.5f);
};

// Editor display name.
struct NameComponent
{
    std::string name;
};

namespace TransformMath
{
    inline glm::quat quatFromEulerXYZDegrees(const glm::vec3 &eulerDegrees)
    {
        glm::vec3 radians = glm::radians(eulerDegrees);
        glm::quat qx = glm::angleAxis(radians.x, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::quat qy = glm::angleAxis(radians.y, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::quat qz = glm::angleAxis(radians.z, glm::vec3(0.0f, 0.0f, 1.0f));
        return glm::normalize(qx * qy * qz);
    }

    inline glm::mat3 normalizeRotationMatrix(const glm::mat3 &rotationScale)
    {
        // Expects R * diag(scale) with no shear: each column is a rotation axis
        // times its scale. Column-normalizing then yields the pure rotation; a
        // negative determinant (mirror) is fixed by flipping column 0.
        // Shear input (parent non-uniform scale x child rotation) is not handled;
        // that case needs polar decomposition.
        glm::mat3 rotation(1.0f);
        rotation[0] = glm::normalize(rotationScale[0]);
        rotation[1] = glm::normalize(rotationScale[1]);
        rotation[2] = glm::normalize(rotationScale[2]);

        if (glm::dot(glm::cross(rotation[0], rotation[1]), rotation[2]) < 0.0f)
        {
            rotation[0] = -rotation[0];
        }
        return rotation;
    }

    // Signed world-space scale from the 3x3 top-left of worldMatrix, following
    // normalizeRotationMatrix's determinant sign convention. Needed for world-space
    // sizes (e.g. world OBB from BoxCollider local halfExtents); local scale alone
    // misses scale inherited from parents.
    // If det(worldMatrix3x3) < 0 (mirror), scale.x is negated so that
    // worldMatrix[0..2] == normalizeRotationMatrix * diag(extractWorldScale).
    inline glm::vec3 extractWorldScale(const glm::mat4 &worldMatrix)
    {
        glm::mat3 rs(worldMatrix);
        glm::vec3 s(glm::length(rs[0]), glm::length(rs[1]), glm::length(rs[2]));
        // Clamp zero scale components to a tiny positive value to avoid divide-by-zero.
        const float minS = 1e-8f;
        s.x = (s.x < minS) ? minS : s.x;
        s.y = (s.y < minS) ? minS : s.y;
        s.z = (s.z < minS) ? minS : s.z;
        // Negative det mirrors the column-0 flip in normalizeRotationMatrix.
        if (glm::determinant(rs) < 0.0f)
            s.x = -s.x;
        return s;
    }

    inline glm::vec3 extractEulerXYZDegrees(const glm::mat3 &rotationMatrix)
    {
        float pitch, yaw, roll;
        glm::mat4 rotation4(1.0f);
        rotation4[0] = glm::vec4(rotationMatrix[0], 0.0f);
        rotation4[1] = glm::vec4(rotationMatrix[1], 0.0f);
        rotation4[2] = glm::vec4(rotationMatrix[2], 0.0f);
        glm::extractEulerAngleXYZ(rotation4, pitch, yaw, roll);
        return glm::degrees(glm::vec3(pitch, yaw, roll));
    }

    inline glm::vec3 extractEulerXYZDegrees(const glm::quat &orientation)
    {
        return extractEulerXYZDegrees(glm::mat3_cast(glm::normalize(orientation)));
    }

    inline glm::quat quatFromRotationMatrix(const glm::mat3 &rotationMatrix)
    {
        return glm::normalize(glm::quat_cast(normalizeRotationMatrix(rotationMatrix)));
    }
} // namespace TransformMath

// Position/rotation/scale plus a cached world matrix.
struct TransformComponent
{
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f}; // Euler angles in degrees, as used by Inspector / Scene JSON.
    glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};
    glm::mat4 worldMatrix = glm::mat4(1.0f); // Computed by TransformSystem.

    void syncOrientationFromEuler()
    {
        orientation = TransformMath::quatFromEulerXYZDegrees(rotation);
    }

    void syncEulerFromOrientation()
    {
        rotation = TransformMath::extractEulerXYZDegrees(orientation);
    }

    void setEulerRotation(const glm::vec3 &eulerDegrees)
    {
        rotation = eulerDegrees;
        syncOrientationFromEuler();
    }

    void setOrientation(const glm::quat &newOrientation)
    {
        orientation = glm::normalize(newOrientation);
        syncEulerFromOrientation();
    }

    void setRotationMatrix(const glm::mat3 &rotationMatrix)
    {
        setOrientation(TransformMath::quatFromRotationMatrix(rotationMatrix));
    }

    glm::mat4 getLocalMatrix() const
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        t *= glm::mat4_cast(glm::normalize(orientation));
        t = glm::scale(t, scale);
        return t;
    }
};

// Parent/child relations.
struct HierarchyComponent
{
    entt::entity parent = entt::null;
    std::vector<entt::entity> children;
};

// References shared GPU mesh buffers.
struct MeshComponent
{
    std::shared_ptr<MeshReference> mesh;
    std::string sourcePath; // "builtin:cube" or absolute OBJ path.
};

// Point light; position comes from TransformComponent.
struct PointLightComponent
{
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 2.0f;
    float radius = 5.0f;
    // Casts shadows (aligned with Unity URP Additional Light Shadows). Default
    // false keeps old scenes unchanged. When enabled and within the intensity-sorted
    // top-N (N=4), PointShadowPass renders it to a dedicated cubemap depth-only RT
    // with PCF in lighting.frag / light_volume_point.frag; lights outside top-N
    // have no shadows that frame but still light.
    bool castShadows = false;

    // Unity-style per-light shadow bias (aligned with URP "Additional Lights ->
    // Shadow Bias"): bias is an artist-tunable parameter, not an engine global.
    // Algorithmic fixes (RPDB, Contact Shadows) are listed in the project docs.
    //
    // Runtime bias = base x per-light multiplier:
    //   raster slope-scale  = 1.5f  x shadowSlopeBias
    //   raster constant     = 0.5f  x shadowConstantBias
    //   frag  normal-offset = 0.003 x shadowNormalBias x (1 - NdotL) + tiny
    //   frag  depth-bias    = 0.012 x shadowDepthBias  x (1 - NdotL) [clamped]
    //
    // Default 1.0 = engine default; 0 disables that component; 2.0 doubles it.
    float shadowSlopeBias = 1.0f;
    float shadowConstantBias = 1.0f;
    float shadowNormalBias = 1.0f;
    float shadowDepthBias = 1.0f;
};

// Directional light; direction derives from TransformComponent.rotation.
struct DirectionalLightComponent
{
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    // Casts shadows (aligned with Unity URP DirectionalLight.shadows). Default
    // false keeps old scenes unchanged; when enabled, DirectionalShadowPass renders
    // a depth-only shadow map with 4x4 PCF in lighting.frag.
    bool castShadows = false;
};

// Spot light; position and direction come from TransformComponent.
struct SpotLightComponent
{
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 3.0f;
    float radius = 10.0f;     // Falloff radius.
    float innerAngle = 15.0f; // Inner cone angle (degrees); fully lit inside.
    float outerAngle = 30.0f; // Outer cone angle (degrees); falls to zero at the edge.
    // Casts shadows (aligned with Unity URP Additional Light Shadows). Default
    // false keeps old scenes unchanged. When enabled and within the intensity-sorted
    // top-N (N=4), SpotShadowPass renders a dedicated 2K depth-only RT with 4x4 PCF
    // in lighting.frag; lights outside top-N have no shadows that frame but still light.
    bool castShadows = false;
};

// PBR material parameters (Cook-Torrance).
struct TextureResource;

struct MaterialComponent
{
    glm::vec3 albedo = {0.8f, 0.8f, 0.8f}; // Albedo color.
    float metallic = 0.0f;                 // Metallic [0,1].
    float roughness = 0.5f;                // Roughness [0,1].

    // Optional albedo texture; color x vertex color is used when null.
    std::shared_ptr<TextureResource> albedoTexture;
    VkDescriptorSet textureDescriptorSet = VK_NULL_HANDLE;
};

// Per-entity Python script slot (Unity MonoBehaviour analogue). Data layer only:
// holds scriptPath/enabled; ScriptEngine owns the runtime lifecycle
// (attach/on_start/on_update/on_stop/detach) via an entity-to-script map.
//
// Serialization vs. runtime fields:
//   - scriptPath / enabled  -> written to Scene JSON
//   - loaded / faulted / lastError -> runtime only (same convention as RigidBody.physicsIndex)
//
// scriptPath is relative to ${PROJECT_DIR}/scripts/ (e.g. "spin_cube.py");
// empty string = empty slot (Inspector shows "Empty", ScriptEngine does not attach).
struct ScriptComponent
{
    // --- Serialized fields ---
    std::string scriptPath; // Relative to scripts/ root; empty means unbound.
    bool enabled = true;    // When false, skips on_start/on_update/on_stop (Unity semantics).

    // --- Runtime cache (not serialized) ---
    bool loaded = false;   // Set true after ScriptEngine::attachScript succeeds.
    bool faulted = false;  // Per-entity exception flag; disables further callbacks.
    std::string lastError; // Last traceback summary (shown as an Inspector tooltip).
};
