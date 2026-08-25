#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "scene/Vertex.h"
#include "scene/Mesh.h"
#include "scene/Light.h"
#include "core/Allocator.h"
#include "ecs/ECSScene.h"

class VulkanContext;

// ============================================================================
// ParsedMaterial — material data parsed from an MTL file
// ============================================================================
struct ParsedMaterial
{
    std::string name;
    glm::vec3 albedo = {0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
};

// ============================================================================
// SceneSetup — built-in mesh generation, default scene setup, model import,
// light syncing
// ============================================================================
namespace SceneSetup
{

    // Parse materials from an MTL file (all newmtl blocks)
    std::vector<ParsedMaterial> parseMTL(const std::string &filepath);

    // Built-in mesh data generation
    void generateCubeMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);
    void generatePlaneMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);

    // Compute a local-space AABB; writes MeshReference::boundsCenter/boundsExtents
    void computeMeshBounds(const std::vector<Vertex> &vertices, std::shared_ptr<MeshReference> &meshRef);

    // Create built-in meshes and upload them to the GPU
    void createBuiltinMeshes(VulkanContext &context, Allocator &allocator,
                             std::shared_ptr<MeshReference> &cubeMeshRef,
                             std::shared_ptr<MeshReference> &planeMeshRef);

    // Build the default scene from ECS entities (cube + ground + 4 lights)
    void setupDefaultScene(ECSScene &scene,
                           std::shared_ptr<MeshReference> &cubeMeshRef,
                           std::shared_ptr<MeshReference> &planeMeshRef);

    // ============================================================================
    // Benchmark scene presets (deterministic; scene.destroyAll() runs first).
    // Cubes are Static so frames stay stable for comparison. Empty / Stress100
    // (10x10 + 4 lights) / Stress1000 (10x10x10 + 16) / LightStress (100 + 256).
    // ============================================================================
    enum class BenchmarkPreset
    {
        Empty,
        Stress100,
        Stress1000,
        LightStress,
    };

    void loadBenchmarkScene(BenchmarkPreset preset,
                            ECSScene &scene,
                            std::shared_ptr<MeshReference> &cubeMeshRef,
                            std::shared_ptr<MeshReference> &planeMeshRef);

    const char *benchmarkPresetName(BenchmarkPreset preset);

    // Import an OBJ model into the scene
    void importModel(const std::string &filepath,
                     VulkanContext &context, Allocator &allocator,
                     ECSScene &scene,
                     std::vector<std::shared_ptr<MeshReference>> &importedMeshes);

    // Gather lights from the ECS and rebuild the SSBOs
    void syncLightsToGPU(ECSScene &scene, Allocator &allocator,
                         std::vector<PointLight> &lights,
                         AllocatedBuffer &lightSSBO,
                         std::vector<DirectionalLight> &dirLights,
                         AllocatedBuffer &dirLightSSBO,
                         std::vector<SpotLight> &spotLights,
                         AllocatedBuffer &spotLightSSBO,
                         VulkanContext &context);

} // namespace SceneSetup
