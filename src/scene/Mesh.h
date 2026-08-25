// ============================================================================
// Mesh.h — Mesh class: OBJ loading (tinyobjloader), staging-buffer upload to
// GPU, buffer cleanup.
// ============================================================================

#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
// GLM (glm::vec3 used in SubMeshData)
#include <glm/glm.hpp>
#include "scene/Vertex.h"
// Allocator (createBufferWithStaging / destroyBuffer)
#include "core/Allocator.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

// Forward declaration to avoid circular includes
class VulkanContext;

// ============================================================================
// SubMeshData — per-material sub-mesh data (CPU side)
// ============================================================================
struct SubMeshData {
    std::string materialName;             // Material name from MTL
    glm::vec3   albedo    = {0.8f, 0.8f, 0.8f};  // From MTL Kd
    float       metallic  = 0.0f;
    float       roughness = 0.5f;
    glm::vec3   emissive  = {0.0f, 0.0f, 0.0f};  // From MTL Ke (emissive detection)
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
};

// ============================================================================
// Mesh — 3D mesh data management
// ============================================================================
class Mesh {
public:
    // Load an OBJ, merging all faces into a single mesh
    void loadFromOBJ(const std::string& filepath);

    // Load an OBJ split by material; returns one element if there is no MTL or a single material
    static std::vector<SubMeshData> loadSubMeshesFromOBJ(const std::string& filepath);

    // Upload vertex/index data to the GPU via a staging buffer
    // (CPU-visible scratch -> device-local GPU memory)
    // context: device/queue info. allocator: provides createBufferWithStaging.
    void upload(VulkanContext& context, Allocator& allocator);

    // Release the GPU vertex/index buffers
    void cleanup(Allocator& allocator);

    // GPU-side buffer handles and index count
    VkBuffer     getVertexBuffer() const { return vertexBuffer.buffer; }
    VkBuffer     getIndexBuffer()  const { return indexBuffer.buffer; }
    uint32_t     getIndexCount()   const { return static_cast<uint32_t>(indices.size()); }

    // CPU-side raw data (for debugging or further processing)
    const std::vector<Vertex>&   getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices()  const { return indices; }

private:
    // CPU-side vertex/index data (filled by loadFromOBJ, used by upload)
    std::vector<Vertex>   vertices;  // Deduplicated unique vertices
    std::vector<uint32_t> indices;   // Indices referencing vertices

    // GPU buffers (created by upload, destroyed by cleanup).
    // AllocatedBuffer = VkBuffer handle + VmaAllocation.
    AllocatedBuffer vertexBuffer{};  // Device-local, GPU-read
    AllocatedBuffer indexBuffer{};   // Device-local, GPU-read
};
