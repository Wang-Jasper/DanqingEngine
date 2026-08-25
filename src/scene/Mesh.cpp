// ============================================================================
// Mesh.cpp — Mesh implementation: OBJ loading (tinyobjloader), vertex dedup,
// staging-buffer GPU upload, cleanup.
// ============================================================================

#include "Mesh.h"
// VulkanContext needed by upload (device/queue)
#include "core/VulkanContext.h"

// ============================================================================
// TINYOBJLOADER_IMPLEMENTATION — header-only lib; define in exactly one .cpp
// (defining it in several .cpp files causes duplicate-definition link errors)
// ============================================================================
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// GLM_ENABLE_EXPERIMENTAL required for glm::hash
#define GLM_ENABLE_EXPERIMENTAL
// GLM hash functions (so glm::vec3 works as an std::unordered_map key)
#include <glm/gtx/hash.hpp>

// ============================================================================
// VertexHash — hash functor for vertex dedup (boost::hash_combine style;
// 0x9e3779b9 is the golden-ratio constant for good hash distribution)
// ============================================================================
struct VertexHash {
    size_t operator()(const Vertex& v) const {
        size_t h = 0;
        h ^= std::hash<float>{}(v.position.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.position.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.position.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.normal.x)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.normal.y)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.normal.z)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.texCoord.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(v.texCoord.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================================
// VertexEqual — equality functor for vertex dedup
// ============================================================================
// Equal iff position, normal, texCoord match; color is excluded because OBJ
// vertex colors are normally overridden by materials.
struct VertexEqual {
    bool operator()(const Vertex& a, const Vertex& b) const {
        return a.position == b.position &&
               a.normal   == b.normal &&
               a.texCoord == b.texCoord;
    }
};

// ============================================================================
// extractVertex — build a Vertex from tinyobj attributes and an index
// ============================================================================
static Vertex extractVertex(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index) {
    Vertex v{};
    v.position = {
        attrib.vertices[3 * index.vertex_index + 0],
        attrib.vertices[3 * index.vertex_index + 1],
        attrib.vertices[3 * index.vertex_index + 2]
    };
    if (index.normal_index >= 0) {
        v.normal = {
            attrib.normals[3 * index.normal_index + 0],
            attrib.normals[3 * index.normal_index + 1],
            attrib.normals[3 * index.normal_index + 2]
        };
    } else {
        v.normal = {0.0f, 1.0f, 0.0f};
    }
    if (index.texcoord_index >= 0) {
        v.texCoord = {
            attrib.texcoords[2 * index.texcoord_index + 0],
            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
        };
    } else {
        v.texCoord = {0.0f, 0.0f};
    }
    v.color = {1.0f, 1.0f, 1.0f};
    return v;
}

// ============================================================================
// computeAngleWeightedNormals — angle-weighted vertex normals
// ============================================================================
static void computeAngleWeightedNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    for (auto& v : vertices) v.normal = {0.0f, 0.0f, 0.0f};

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        auto& v0 = vertices[indices[i]];
        auto& v1 = vertices[indices[i + 1]];
        auto& v2 = vertices[indices[i + 2]];

        glm::vec3 e01 = v1.position - v0.position;
        glm::vec3 e02 = v2.position - v0.position;
        glm::vec3 e12 = v2.position - v1.position;

        glm::vec3 faceNormal = glm::cross(e01, e02);
        float faceLen = glm::length(faceNormal);
        if (faceLen < 1e-8f) continue;
        faceNormal /= faceLen;

        float angle0 = acosf(glm::clamp(glm::dot(glm::normalize(e01), glm::normalize(e02)), -1.0f, 1.0f));
        float angle1 = acosf(glm::clamp(glm::dot(glm::normalize(-e01), glm::normalize(e12)), -1.0f, 1.0f));
        float angle2 = acosf(glm::clamp(glm::dot(glm::normalize(-e02), glm::normalize(-e12)), -1.0f, 1.0f));

        v0.normal += faceNormal * angle0;
        v1.normal += faceNormal * angle1;
        v2.normal += faceNormal * angle2;
    }

    for (auto& v : vertices) {
        float len = glm::length(v.normal);
        if (len > 1e-6f) v.normal /= len;
    }
}

// ============================================================================
// loadFromOBJ() — load model data from an OBJ file
// ============================================================================
void Mesh::loadFromOBJ(const std::string& filepath) {
    tinyobj::attrib_t attrib;                     // Raw positions/normals/texcoords
    std::vector<tinyobj::shape_t> shapes;          // Geometry shapes (each holds face indices)
    std::vector<tinyobj::material_t> materials;    // Materials (unused for now)
    std::string warn, err;                         // Warnings and errors

    // Parse the OBJ; mtlbasepath lets tinyobj find referenced MTL files
    std::string mtlbasepath = filepath;
    auto lastSlash = mtlbasepath.find_last_of("/\\");
    mtlbasepath = (lastSlash != std::string::npos) ? mtlbasepath.substr(0, lastSlash + 1) : "";

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), mtlbasepath.c_str())) {
        throw std::runtime_error("Failed to load OBJ: " + warn + err);
    }

    // Print warnings (e.g. missing normals or texcoords)
    if (!warn.empty()) {
        std::cout << "[Mesh] OBJ warning: " << warn << "\n";
    }

    // Clear any previously loaded data
    vertices.clear();
    indices.clear();

    // Dedup map: Vertex -> index into vertices; VertexHash/VertexEqual provide hashing/equality
    std::unordered_map<Vertex, uint32_t, VertexHash, VertexEqual> uniqueVertices;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex = extractVertex(attrib, index);

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }
            indices.push_back(uniqueVertices[vertex]);
        }
    }

    // OBJ has no normals: compute angle-weighted vertex normals
    if (attrib.normals.empty()) {
        computeAngleWeightedNormals(vertices, indices);
        std::cout << "[Mesh] No normals in OBJ, computed angle-weighted vertex normals.\n";
    }

    std::cout << "[Mesh] Loaded " << filepath << ": "
              << vertices.size() << " unique vertices, "
              << indices.size() << " indices.\n";
}

// ============================================================================
// upload() — upload CPU data to the GPU via a staging buffer
// ============================================================================
// Staging: CPU-visible temp buffer, memcpy data, vkCmdCopyBuffer into a
// DEVICE_LOCAL buffer, destroy staging. Faster than rendering from CPU-visible
// memory because DEVICE_LOCAL bandwidth is far higher.
void Mesh::upload(VulkanContext& context, Allocator& allocator) {
    // Cannot upload an empty mesh
    if (vertices.empty() || indices.empty()) {
        throw std::runtime_error("Cannot upload empty mesh!");
    }

    VkDeviceSize vertexSize = sizeof(Vertex) * vertices.size();
    // Target buffer will be used as a vertex buffer
    vertexBuffer = allocator.createBufferWithStaging(
        context, vertices.data(), vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
    // Target buffer will be used as an index buffer
    indexBuffer = allocator.createBufferWithStaging(
        context, indices.data(), indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    std::cout << "[Mesh] Uploaded to GPU.\n";
}

// ============================================================================
// cleanup() — release GPU resources
// ============================================================================
// allocator.destroyBuffer calls vmaDestroyBuffer, releasing VkBuffer + VmaAllocation
void Mesh::cleanup(Allocator& allocator) {
    allocator.destroyBuffer(vertexBuffer);
    allocator.destroyBuffer(indexBuffer);
}

// ============================================================================
// loadSubMeshesFromOBJ() — load an OBJ split into per-material sub-meshes
// ============================================================================
std::vector<SubMeshData> Mesh::loadSubMeshesFromOBJ(const std::string& filepath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Directory path as the MTL search base
    std::string mtlbasepath = filepath;
    auto lastSlash = mtlbasepath.find_last_of("/\\");
    mtlbasepath = (lastSlash != std::string::npos) ? mtlbasepath.substr(0, lastSlash + 1) : "";

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), mtlbasepath.c_str())) {
        throw std::runtime_error("Failed to load OBJ: " + warn + err);
    }
    if (!warn.empty()) std::cout << "[Mesh] OBJ warning: " << warn << "\n";

    // No materials: fall back to a single SubMesh
    if (materials.empty()) {
        SubMeshData single;
        single.materialName = "default";

        std::unordered_map<Vertex, uint32_t, VertexHash, VertexEqual> uniqueVerts;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex v = extractVertex(attrib, index);
                if (uniqueVerts.count(v) == 0) { uniqueVerts[v] = (uint32_t)single.vertices.size(); single.vertices.push_back(v); }
                single.indices.push_back(uniqueVerts[v]);
            }
        }

        if (attrib.normals.empty()) {
            computeAngleWeightedNormals(single.vertices, single.indices);
            std::cout << "[Mesh] No normals in OBJ, computed angle-weighted vertex normals for single sub-mesh.\n";
        }

        return { single };
    }

    // Group faces by material_id (key = material_id, value = SubMeshData)
    std::unordered_map<int, SubMeshData> matGroups;
    // Persistent per-material dedup map (avoids rebuilding per face)
    std::unordered_map<int, std::unordered_map<Vertex, uint32_t, VertexHash, VertexEqual>> matUniqueVerts;

    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            int matId = shape.mesh.material_ids[f];

            if (matGroups.find(matId) == matGroups.end()) {
                SubMeshData sub;
                if (matId >= 0 && matId < (int)materials.size()) {
                    auto& m = materials[matId];
                    sub.materialName = m.name;
                    sub.albedo = { m.diffuse[0], m.diffuse[1], m.diffuse[2] };
                    sub.emissive = { m.emission[0], m.emission[1], m.emission[2] };
                    sub.roughness = 1.0f - std::min(m.shininess / 1000.0f, 1.0f);
                    sub.roughness = std::max(sub.roughness, 0.05f);
                } else {
                    sub.materialName = "unknown_" + std::to_string(matId);
                }
                matGroups[matId] = sub;
            }

            auto& sub = matGroups[matId];
            auto& uniqueVerts = matUniqueVerts[matId];

            for (int v = 1; v + 1 < fv; v++) {
                int triIndices[3] = { 0, v, v + 1 };
                for (int ti = 0; ti < 3; ti++) {
                    const auto& idx = shape.mesh.indices[indexOffset + triIndices[ti]];
                    Vertex vert = extractVertex(attrib, idx);

                    if (uniqueVerts.count(vert) == 0) {
                        uniqueVerts[vert] = (uint32_t)sub.vertices.size();
                        sub.vertices.push_back(vert);
                    }
                    sub.indices.push_back(uniqueVerts[vert]);
                }
            }
            indexOffset += fv;
        }
    }

    // Check normals once before the loop (avoid rescanning per sub-mesh)
    bool hasNormals = !attrib.normals.empty();

    // Collect into a vector, skipping empty sub-meshes
    std::vector<SubMeshData> result;
    for (auto& [id, sub] : matGroups) {
        if (sub.indices.empty()) continue;

        if (!hasNormals) {
            computeAngleWeightedNormals(sub.vertices, sub.indices);
        }

        std::cout << "[Mesh] SubMesh '" << sub.materialName << "': "
                  << sub.vertices.size() << " verts, " << sub.indices.size() / 3 << " tris, "
                  << "albedo=(" << sub.albedo.r << "," << sub.albedo.g << "," << sub.albedo.b << ")\n";
        result.push_back(std::move(sub));
    }

    return result;
}
