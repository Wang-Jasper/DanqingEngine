// ============================================================================
// Vertex.h — Vertex layout + Vulkan binding/attribute descriptions,
// UniformBufferObject (MVP), GeometryPushConstants, InstanceData.
// ============================================================================

#pragma once

// Vulkan (VkVertexInputBindingDescription etc.)
#include <vulkan/vulkan.h>
// GLM (vec2, vec3, mat4)
#include <glm/glm.hpp>
// std::array for the fixed-size attribute description array
#include <array>

// ============================================================================
// Vertex — tightly packed, no padding:
// offset  0: position  (vec3, 12 bytes)
// offset 12: normal    (vec3, 12 bytes)
// offset 24: color     (vec3, 12 bytes)
// offset 36: texCoord  (vec2,  8 bytes)
// sizeof(Vertex) = 44 bytes
// ============================================================================
struct Vertex
{
    glm::vec3 position; // Position in model space
    glm::vec3 normal;   // Unit normal pointing outward (for lighting)
    glm::vec3 color;    // Color (RGB, [0,1])
    glm::vec2 texCoord; // UV texcoord (usually [0,1], for texture sampling)

    // ========================================================================
    // getBindingDescription() — how Vulkan reads vertex buffer data
    // ========================================================================
    // binding 0 (single buffer); stride = bytes per vertex; per-vertex rate
    // (per-instance only for instanced rendering).
    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    // ========================================================================
    // getAttributeDescriptions() — per-attribute binding/location/format/offset
    // ========================================================================
    // location matches layout(location = N) in shaders; format e.g.
    // R32G32B32_SFLOAT = three 32-bit floats = vec3; offset is the byte offset
    // of the member inside Vertex.
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attr{};

        // location 0: position (vec3), matches layout(location = 0) in vec3 inPosition
        attr[0].binding = 0;
        attr[0].location = 0;
        // R32G32B32_SFLOAT = three 32-bit floats = glm::vec3
        attr[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        // offsetof(Vertex, position) = 0
        attr[0].offset = offsetof(Vertex, position);

        // location 1: normal (vec3), matches layout(location = 1) in vec3 inNormal
        attr[1].binding = 0;
        attr[1].location = 1;
        attr[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        // offsetof(Vertex, normal) = 12 (after position)
        attr[1].offset = offsetof(Vertex, normal);

        // location 2: color (vec3), matches layout(location = 2) in vec3 inColor
        attr[2].binding = 0;
        attr[2].location = 2;
        attr[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        // offsetof(Vertex, color) = 24 (after position + normal)
        attr[2].offset = offsetof(Vertex, color);

        // location 3: texCoord (vec2), matches layout(location = 3) in vec2 inTexCoord
        attr[3].binding = 0;
        attr[3].location = 3;
        // R32G32_SFLOAT = two 32-bit floats = glm::vec2
        attr[3].format = VK_FORMAT_R32G32_SFLOAT;
        // offsetof(Vertex, texCoord) = 36 (after position + normal + color)
        attr[3].offset = offsetof(Vertex, texCoord);

        return attr;
    }
};

// ============================================================================
// UniformBufferObject — MVP matrices, uploaded per frame to the vertex shader
// ============================================================================
// alignas(16): std140 requires mat4 to be 16-byte aligned and glm::mat4 is not
// by default, so force it to match the GPU-side layout.
struct UniformBufferObject
{
    // View: world -> camera space (glm::lookAt)
    alignas(16) glm::mat4 view;
    // Projection: camera -> clip space (perspective). Vulkan's Y axis is
    // flipped vs OpenGL, so proj[1][1] is negated.
    alignas(16) glm::mat4 proj;
};

// ============================================================================
// GeometryPushConstants — Geometry Pass push constant (per-batch)
// ============================================================================
// Switched from per-object to per-batch: carries only the offset of this batch
// inside the per-frame instance SSBO plus a flag for whether the bound material
// has an albedo texture. Per-instance model matrix and material parameters live
// in the SSBO, indexed in the vertex shader via (instanceOffset + gl_InstanceIndex).
struct GeometryPushConstants
{
    uint32_t instanceOffset; // index into InstanceData[] for this batch (instance 0)
    float hasAlbedoTex;      // 1.0 if the bound texture descriptor set is a real texture, else 0.0
    uint32_t _pad0;
    uint32_t _pad1; // pad to 16 bytes (std430 / push_constant std140 friendly)
};

// ============================================================================
// InstanceData — Geometry Pass per-instance SSBO entry
// ============================================================================
// Stored in a per-frame host-mapped storage buffer, indexed in geometry.vert
// via (instanceOffset + gl_InstanceIndex). Layout matches std430 in geometry.vert.
struct InstanceData
{
    glm::mat4 model;             // 64 bytes
    glm::vec4 albedoAndMetallic; // 16 bytes: rgb=albedo, a=metallic
    glm::vec4 roughnessAndFlags; // 16 bytes: r=roughness, gba=reserved
}; // total 96 bytes
