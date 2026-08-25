#include "scene/SceneSetup.h"
#include "core/VulkanContext.h"
#include "core/asset/AssetRegistry.h"
#include "ecs/Systems.h"
#include "ecs/Components.h"
#include "physics/PhysicsComponents.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <string>
#include <cstdint>

namespace SceneSetup
{

    // ============================================================================
    // computeMeshBounds — local-space AABB from the vertex array
    // ============================================================================
    void computeMeshBounds(const std::vector<Vertex> &vertices, std::shared_ptr<MeshReference> &meshRef)
    {
        if (vertices.empty())
            return;
        glm::vec3 bmin = vertices[0].position;
        glm::vec3 bmax = vertices[0].position;
        for (const auto &v : vertices)
        {
            bmin = glm::min(bmin, v.position);
            bmax = glm::max(bmax, v.position);
        }
        meshRef->boundsCenter = (bmin + bmax) * 0.5f;
        meshRef->boundsExtents = (bmax - bmin) * 0.5f;
    }

    // ============================================================================
    // parseMTL() — parse an MTL file, returning all materials
    // ============================================================================
    std::vector<ParsedMaterial> parseMTL(const std::string &filepath)
    {
        std::vector<ParsedMaterial> materials;
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "[SceneSetup] Failed to open MTL: " << filepath << "\n";
            return materials;
        }

        ParsedMaterial *current = nullptr;
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "newmtl")
            {
                materials.emplace_back();
                current = &materials.back();
                iss >> current->name;
            }
            else if (current && (token == "Kd" || token == "Ka"))
            {
                // Prefer Kd (diffuse); accept Ka only while albedo is still the default
                float r, g, b;
                iss >> r >> g >> b;
                if (token == "Kd")
                {
                    current->albedo = {r, g, b};
                }
                else if (token == "Ka" && current->albedo == glm::vec3(0.8f))
                {
                    current->albedo = {r, g, b};
                }
            }
            else if (current && token == "Ns")
            {
                // Ns (specular exponent) -> roughness approximation
                // Ns is usually 0-1000, higher = smoother
                float ns;
                iss >> ns;
                current->roughness = 1.0f - std::min(ns / 1000.0f, 1.0f);
                current->roughness = std::max(current->roughness, 0.05f);
            }
            else if (current && (token == "Pm" || token == "metallic"))
            {
                // Non-standard but common metallic field
                iss >> current->metallic;
            }
        }
        return materials;
    }

    void generateCubeMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
    {
        vertices = {
            // Front face (z = +0.5), normal (0,0,1), red
            {{-0.5f, -0.5f, 0.5f}, {0, 0, 1}, {1.0f, 0.3f, 0.3f}, {0, 0}},
            {{0.5f, -0.5f, 0.5f}, {0, 0, 1}, {1.0f, 0.3f, 0.3f}, {1, 0}},
            {{0.5f, 0.5f, 0.5f}, {0, 0, 1}, {1.0f, 0.3f, 0.3f}, {1, 1}},
            {{-0.5f, 0.5f, 0.5f}, {0, 0, 1}, {1.0f, 0.3f, 0.3f}, {0, 1}},
            // Back face (z = -0.5), normal (0,0,-1), green
            {{0.5f, -0.5f, -0.5f}, {0, 0, -1}, {0.3f, 1.0f, 0.3f}, {0, 0}},
            {{-0.5f, -0.5f, -0.5f}, {0, 0, -1}, {0.3f, 1.0f, 0.3f}, {1, 0}},
            {{-0.5f, 0.5f, -0.5f}, {0, 0, -1}, {0.3f, 1.0f, 0.3f}, {1, 1}},
            {{0.5f, 0.5f, -0.5f}, {0, 0, -1}, {0.3f, 1.0f, 0.3f}, {0, 1}},
            // Top face (y = +0.5), normal (0,1,0), blue
            {{-0.5f, 0.5f, 0.5f}, {0, 1, 0}, {0.3f, 0.3f, 1.0f}, {0, 0}},
            {{0.5f, 0.5f, 0.5f}, {0, 1, 0}, {0.3f, 0.3f, 1.0f}, {1, 0}},
            {{0.5f, 0.5f, -0.5f}, {0, 1, 0}, {0.3f, 0.3f, 1.0f}, {1, 1}},
            {{-0.5f, 0.5f, -0.5f}, {0, 1, 0}, {0.3f, 0.3f, 1.0f}, {0, 1}},
            // Bottom face (y = -0.5), normal (0,-1,0), yellow
            {{-0.5f, -0.5f, -0.5f}, {0, -1, 0}, {1.0f, 1.0f, 0.3f}, {0, 0}},
            {{0.5f, -0.5f, -0.5f}, {0, -1, 0}, {1.0f, 1.0f, 0.3f}, {1, 0}},
            {{0.5f, -0.5f, 0.5f}, {0, -1, 0}, {1.0f, 1.0f, 0.3f}, {1, 1}},
            {{-0.5f, -0.5f, 0.5f}, {0, -1, 0}, {1.0f, 1.0f, 0.3f}, {0, 1}},
            // Right face (x = +0.5), normal (1,0,0), magenta
            {{0.5f, -0.5f, 0.5f}, {1, 0, 0}, {1.0f, 0.3f, 1.0f}, {0, 0}},
            {{0.5f, -0.5f, -0.5f}, {1, 0, 0}, {1.0f, 0.3f, 1.0f}, {1, 0}},
            {{0.5f, 0.5f, -0.5f}, {1, 0, 0}, {1.0f, 0.3f, 1.0f}, {1, 1}},
            {{0.5f, 0.5f, 0.5f}, {1, 0, 0}, {1.0f, 0.3f, 1.0f}, {0, 1}},
            // Left face (x = -0.5), normal (-1,0,0), cyan
            {{-0.5f, -0.5f, -0.5f}, {-1, 0, 0}, {0.3f, 1.0f, 1.0f}, {0, 0}},
            {{-0.5f, -0.5f, 0.5f}, {-1, 0, 0}, {0.3f, 1.0f, 1.0f}, {1, 0}},
            {{-0.5f, 0.5f, 0.5f}, {-1, 0, 0}, {0.3f, 1.0f, 1.0f}, {1, 1}},
            {{-0.5f, 0.5f, -0.5f}, {-1, 0, 0}, {0.3f, 1.0f, 1.0f}, {0, 1}},
        };
        indices = {
            0,
            1,
            2,
            2,
            3,
            0,
            4,
            5,
            6,
            6,
            7,
            4,
            8,
            9,
            10,
            10,
            11,
            8,
            12,
            13,
            14,
            14,
            15,
            12,
            16,
            17,
            18,
            18,
            19,
            16,
            20,
            21,
            22,
            22,
            23,
            20,
        };
    }

    void generatePlaneMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
    {
        glm::vec3 n = {0, 1, 0};
        glm::vec3 c = {0.7f, 0.7f, 0.7f};
        vertices = {
            {{-0.5f, 0, -0.5f}, n, c, {0, 0}},
            {{0.5f, 0, -0.5f}, n, c, {1, 0}},
            {{0.5f, 0, 0.5f}, n, c, {1, 1}},
            {{-0.5f, 0, 0.5f}, n, c, {0, 1}},
        };
        indices = {0, 3, 2, 2, 1, 0};
    }

    void createBuiltinMeshes(VulkanContext &context, Allocator &allocator,
                             std::shared_ptr<MeshReference> &cubeMeshRef,
                             std::shared_ptr<MeshReference> &planeMeshRef)
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        // Cube
        generateCubeMesh(vertices, indices);
        cubeMeshRef = std::make_shared<MeshReference>();
        cubeMeshRef->vertexBuffer = allocator.createBufferWithStaging(
            context, vertices.data(), sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cubeMeshRef->indexBuffer = allocator.createBufferWithStaging(
            context, indices.data(), sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cubeMeshRef->indexCount = static_cast<uint32_t>(indices.size());
        cubeMeshRef->vertexCount = static_cast<uint32_t>(vertices.size());
        computeMeshBounds(vertices, cubeMeshRef);
        vertices.clear();
        indices.clear();
        generatePlaneMesh(vertices, indices);
        planeMeshRef = std::make_shared<MeshReference>();
        planeMeshRef->vertexBuffer = allocator.createBufferWithStaging(
            context, vertices.data(), sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        planeMeshRef->indexBuffer = allocator.createBufferWithStaging(
            context, indices.data(), sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        planeMeshRef->indexCount = static_cast<uint32_t>(indices.size());
        planeMeshRef->vertexCount = static_cast<uint32_t>(vertices.size());
        computeMeshBounds(vertices, planeMeshRef);
    }

    void setupDefaultScene(ECSScene &scene,
                           std::shared_ptr<MeshReference> &cubeMeshRef,
                           std::shared_ptr<MeshReference> &planeMeshRef)
    {
        auto &reg = scene.registry;

        // Cube with a dynamic rigid body (falls under gravity in Play mode)
        auto cube = scene.createEntity("Cube");
        reg.emplace<MeshComponent>(cube, MeshComponent{cubeMeshRef, "builtin:cube"});
        reg.emplace<MaterialComponent>(cube);
        auto &cubeTc = reg.get<TransformComponent>(cube);
        cubeTc.position = glm::vec3(0.0f, 3.0f, 0.0f);
        // CP-3.2: RigidBodyComponent no longer holds friction / restitution (moved to collider.material)
        reg.emplace<RigidBodyComponent>(cube, RigidBodyComponent{BodyType::Dynamic, 1.0f});
        {
            // Faithfully restoring the pre-CP-3.2 values: restitution=0.3, friction=0.5
            BoxColliderComponent bc;
            bc.center = glm::vec3(0.0f);
            bc.halfExtents = glm::vec3(0.5f);
            bc.material.restitution = 0.3f;
            bc.material.friction = 0.5f;
            reg.emplace<BoxColliderComponent>(cube, bc);
        }
        // Attach spin_cube.py to the Cube by default (per-entity ScriptComponent).
        // Only data fields are written; the module loads later in
        // ScriptEngine::syncFromScene (13.6), so the Inspector shows the
        // binding right after startup.
        {
            ScriptComponent sc;
            sc.scriptPath = "spin_cube.py"; // Relative to ${PROJECT_DIR}/scripts/
            sc.enabled = true;
            reg.emplace<ScriptComponent>(cube, sc);
        }

        // Ground (static rigid body, not affected by gravity)
        auto ground = scene.createEntity("Ground");
        reg.emplace<MeshComponent>(ground, MeshComponent{planeMeshRef, "builtin:plane"});
        auto &groundMat = reg.emplace<MaterialComponent>(ground);
        groundMat.albedo = glm::vec3(0.4f, 0.4f, 0.45f);
        groundMat.roughness = 0.9f;
        auto &groundTc = reg.get<TransformComponent>(ground);
        groundTc.position = glm::vec3(0.0f, -1.0f, 0.0f);
        groundTc.scale = glm::vec3(10.0f, 1.0f, 10.0f);
        // CP-3.2: as above, material moved to the collider
        reg.emplace<RigidBodyComponent>(ground, RigidBodyComponent{BodyType::Static, 0.0f});
        // Ground collider: top face aligned with the visual plane (y = -1),
        // thickness extends downward so objects don't look like they float
        // Faithfully restoring the pre-CP-3.2 values: restitution=0.3, friction=0.8
        {
            BoxColliderComponent bc;
            bc.center = glm::vec3(0.0f, -0.5f, 0.0f);
            bc.halfExtents = glm::vec3(0.5f, 0.5f, 0.5f);
            bc.material.restitution = 0.3f;
            bc.material.friction = 0.8f;
            reg.emplace<BoxColliderComponent>(ground, bc);
        }

        auto defaultLights = createDefaultLights();
        const char *lightNames[] = {"Warm White Light", "Cool Blue Light", "Red Light", "Green Light"};
        for (size_t i = 0; i < defaultLights.size(); i++)
        {
            auto light = scene.createEntity(lightNames[i]);
            auto &tc = reg.get<TransformComponent>(light);
            tc.position = defaultLights[i].position;
            reg.emplace<PointLightComponent>(light, PointLightComponent{
                                                        defaultLights[i].color, defaultLights[i].intensity, defaultLights[i].radius});
        }
    }
    void importModel(const std::string &filepath,
                     VulkanContext &context, Allocator &allocator,
                     ECSScene &scene,
                     std::vector<std::shared_ptr<MeshReference>> &importedMeshes)
    {
        vkDeviceWaitIdle(context.getDevice());

        // Try loading split by material
        auto subMeshes = Mesh::loadSubMeshesFromOBJ(filepath);

        // Extract the file name
        std::string name = filepath;
        auto lastSlash = name.find_last_of("/\\");
        if (lastSlash != std::string::npos)
            name = name.substr(lastSlash + 1);

        // Register the asset in the GUID system for a stable "guid:<uuid>"
        // reference. Assets outside the assets root (dragged in externally) get
        // an empty string; fall back to the absolute path and warn the user to
        // move the file into assets/.
        const std::string assetGuid = AssetRegistry::instance().registerNewAsset(filepath);
        auto makeSourceId = [&](const std::string &subName) -> std::string
        {
            if (!assetGuid.empty())
                return subName.empty() ? ("guid:" + assetGuid)
                                       : ("guid:" + assetGuid + "#" + subName);
            // Fallback: outside the assets root, cannot be GUID'd; keep the
            // absolute path so it is not resolved as a GUID on save (external
            // files shouldn't need to be part of the project).
            std::cerr << "[SceneSetup] Imported asset is outside assets root; reference will not survive moves: "
                      << filepath << "\n";
            return subName.empty() ? filepath : (filepath + "#" + subName);
        };

        if (subMeshes.size() <= 1)
        {
            // Single material or none -> legacy flow (single entity)
            auto &sub = subMeshes[0];
            VkDeviceSize vSize = sizeof(Vertex) * sub.vertices.size();
            VkDeviceSize iSize = sizeof(uint32_t) * sub.indices.size();

            auto meshRef = std::make_shared<MeshReference>();
            meshRef->vertexBuffer = allocator.createBufferWithStaging(context, sub.vertices.data(), vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            meshRef->indexBuffer = allocator.createBufferWithStaging(context, sub.indices.data(), iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            meshRef->indexCount = static_cast<uint32_t>(sub.indices.size());
            meshRef->vertexCount = static_cast<uint32_t>(sub.vertices.size());
            computeMeshBounds(sub.vertices, meshRef);
            importedMeshes.push_back(meshRef);

            auto entity = scene.createEntity(name);
            scene.registry.emplace<MeshComponent>(entity, MeshComponent{meshRef, makeSourceId("")});
            // Material info present: add a MaterialComponent
            if (sub.albedo != glm::vec3(0.8f, 0.8f, 0.8f) || sub.metallic != 0.0f || sub.roughness != 0.5f)
            {
                MaterialComponent mat;
                mat.albedo = sub.albedo;
                mat.metallic = sub.metallic;
                mat.roughness = sub.roughness;
                scene.registry.emplace<MaterialComponent>(entity, mat);
            }
            else
            {
                scene.registry.emplace<MaterialComponent>(entity);
            }
            scene.selectedEntity = entity;

            std::cout << "[SceneSetup] Imported: " << name
                      << " (" << sub.indices.size() / 3 << " triangles)\n";
        }
        else
        {
            // Multiple materials -> parent entity + one child entity per material
            auto parentEntity = scene.createEntity(name);

            for (auto &sub : subMeshes)
            {
                VkDeviceSize vSize = sizeof(Vertex) * sub.vertices.size();
                VkDeviceSize iSize = sizeof(uint32_t) * sub.indices.size();

                auto meshRef = std::make_shared<MeshReference>();
                meshRef->vertexBuffer = allocator.createBufferWithStaging(context, sub.vertices.data(), vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
                meshRef->indexBuffer = allocator.createBufferWithStaging(context, sub.indices.data(), iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
                meshRef->indexCount = static_cast<uint32_t>(sub.indices.size());
                meshRef->vertexCount = static_cast<uint32_t>(sub.vertices.size());
                computeMeshBounds(sub.vertices, meshRef);
                importedMeshes.push_back(meshRef);

                auto childEntity = scene.createEntity(sub.materialName);
                scene.setParent(childEntity, parentEntity);
                scene.registry.emplace<MeshComponent>(childEntity, MeshComponent{meshRef, makeSourceId(sub.materialName)});

                MaterialComponent mat;
                mat.albedo = sub.albedo;
                mat.metallic = sub.metallic;
                mat.roughness = sub.roughness;
                scene.registry.emplace<MaterialComponent>(childEntity, mat);

                // Emissive detection: Ke > 0 -> add a point light
                float emissiveStrength = sub.emissive.r + sub.emissive.g + sub.emissive.b;
                if (emissiveStrength > 0.1f)
                {
                    // Sub-mesh center as the light position
                    glm::vec3 center(0);
                    for (auto &v : sub.vertices)
                        center += v.position;
                    center /= (float)sub.vertices.size();

                    auto &tc = scene.registry.get<TransformComponent>(childEntity);
                    tc.position = center;

                    glm::vec3 lightColor = glm::normalize(sub.emissive);
                    float intensity = emissiveStrength / 3.0f;
                    scene.registry.emplace<PointLightComponent>(childEntity, PointLightComponent{lightColor, intensity, 5.0f});
                    std::cout << "[SceneSetup] Emissive material '" << sub.materialName << "' → PointLight at ("
                              << center.x << "," << center.y << "," << center.z << ") intensity=" << intensity << "\n";
                }
            }

            scene.selectedEntity = parentEntity;
            std::cout << "[SceneSetup] Imported: " << name << " (" << subMeshes.size() << " sub-meshes with materials)\n";
        }
    }

    void syncLightsToGPU(ECSScene &scene, Allocator &allocator,
                         std::vector<PointLight> &lights,
                         AllocatedBuffer &lightSSBO,
                         std::vector<DirectionalLight> &dirLights,
                         AllocatedBuffer &dirLightSSBO,
                         std::vector<SpotLight> &spotLights,
                         AllocatedBuffer &spotLightSSBO,
                         VulkanContext &context)
    {
        Systems::gatherLights(scene.registry, lights);
        if (lightSSBO.buffer != VK_NULL_HANDLE)
            allocator.destroyBuffer(lightSSBO);
        {
            VkDeviceSize ssboSize = lights.empty() ? sizeof(PointLight) : sizeof(PointLight) * lights.size();
            PointLight dummy{};
            const void *data = lights.empty() ? (const void *)&dummy : (const void *)lights.data();
            lightSSBO = allocator.createBufferWithStaging(
                context, data, ssboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        Systems::gatherDirLights(scene.registry, dirLights);
        if (dirLightSSBO.buffer != VK_NULL_HANDLE)
            allocator.destroyBuffer(dirLightSSBO);
        {
            VkDeviceSize size = dirLights.empty() ? sizeof(DirectionalLight) : sizeof(DirectionalLight) * dirLights.size();
            DirectionalLight dummy{};
            const void *data = dirLights.empty() ? (const void *)&dummy : (const void *)dirLights.data();
            dirLightSSBO = allocator.createBufferWithStaging(
                context, data, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        Systems::gatherSpotLights(scene.registry, spotLights);
        if (spotLightSSBO.buffer != VK_NULL_HANDLE)
            allocator.destroyBuffer(spotLightSSBO);
        {
            VkDeviceSize size = spotLights.empty() ? sizeof(SpotLight) : sizeof(SpotLight) * spotLights.size();
            SpotLight dummy{};
            const void *data = spotLights.empty() ? (const void *)&dummy : (const void *)spotLights.data();
            spotLightSSBO = allocator.createBufferWithStaging(
                context, data, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
    }

    // ========================================================================
    // Benchmark scene implementations
    // ========================================================================
    const char *benchmarkPresetName(BenchmarkPreset preset)
    {
        switch (preset)
        {
        case BenchmarkPreset::Empty:
            return "Empty";
        case BenchmarkPreset::Stress100:
            return "Stress100";
        case BenchmarkPreset::Stress1000:
            return "Stress1000";
        case BenchmarkPreset::LightStress:
            return "LightStress";
        }
        return "Unknown";
    }

    namespace
    {
        // Simple LCG (deterministic per seed)
        struct LCG
        {
            uint32_t state;
            float next01()
            {
                state = state * 1664525u + 1013904223u;
                return static_cast<float>(state >> 8) / 16777216.0f; // 24-bit mantissa
            }
            float range(float a, float b) { return a + (b - a) * next01(); }
        };

        // Cube entity with Mesh + Material only (no RigidBody, so physics is not
        // a noise factor at 1000 objects; the benchmark targets render throughput)
        entt::entity makeCubeEntity(ECSScene &scene,
                                    std::shared_ptr<MeshReference> &cubeMeshRef,
                                    const glm::vec3 &pos, const glm::vec3 &color,
                                    const std::string &name)
        {
            auto &reg = scene.registry;
            auto e = scene.createEntity(name);
            reg.emplace<MeshComponent>(e, MeshComponent{cubeMeshRef, "builtin:cube"});
            auto &mat = reg.emplace<MaterialComponent>(e);
            mat.albedo = color;
            mat.roughness = 0.5f;
            mat.metallic = 0.0f;
            auto &tc = reg.get<TransformComponent>(e);
            tc.position = pos;
            return e;
        }

        // Point light entity (Mesh-less; no components beyond Transform)
        void makePointLight(ECSScene &scene, const glm::vec3 &pos,
                            const glm::vec3 &color, float intensity, float radius,
                            const std::string &name)
        {
            auto &reg = scene.registry;
            auto e = scene.createEntity(name);
            auto &tc = reg.get<TransformComponent>(e);
            tc.position = pos;
            reg.emplace<PointLightComponent>(e, PointLightComponent{color, intensity, radius});
        }

        // Shared: arrange cubes in a grid. Positions spread evenly around the
        // origin with the given spacing.
        void layoutCubeGrid(ECSScene &scene,
                            std::shared_ptr<MeshReference> &cubeMeshRef,
                            int countX, int countY, int countZ,
                            float spacing, const glm::vec3 &origin,
                            uint32_t seed)
        {
            LCG rng{seed};
            int idx = 0;
            for (int z = 0; z < countZ; ++z)
                for (int y = 0; y < countY; ++y)
                    for (int x = 0; x < countX; ++x)
                    {
                        glm::vec3 pos = origin + glm::vec3(
                                                     (x - (countX - 1) * 0.5f) * spacing,
                                                     (y - (countY - 1) * 0.5f) * spacing,
                                                     (z - (countZ - 1) * 0.5f) * spacing);
                        glm::vec3 color(rng.range(0.3f, 0.9f),
                                        rng.range(0.3f, 0.9f),
                                        rng.range(0.3f, 0.9f));
                        makeCubeEntity(scene, cubeMeshRef, pos, color,
                                       "Cube_" + std::to_string(idx++));
                    }
        }
    } // anonymous namespace

    void loadBenchmarkScene(BenchmarkPreset preset,
                            ECSScene &scene,
                            std::shared_ptr<MeshReference> &cubeMeshRef,
                            std::shared_ptr<MeshReference> &planeMeshRef)
    {
        // Clear the scene (onBeforeDestroyEntity releases external resources like physics bodies)
        scene.destroyAll();
        (void)planeMeshRef; // No ground in benchmarks; a large ground G-Buffer region would skew the data

        switch (preset)
        {
        case BenchmarkPreset::Empty:
            // Fully empty scene, camera baseline only
            std::printf("[Bench] Loaded preset 'Empty' (0 entities, 0 lights)\n");
            return;

        case BenchmarkPreset::Stress100:
        {
            // 10 x 10 x 1 grid, spacing 2, origin (0, 1, 0)
            layoutCubeGrid(scene, cubeMeshRef, 10, 10, 1, 2.0f, glm::vec3(0, 1, 0), 0xC0FFEEu);
            // 4 default lights, matching setupDefaultScene's colors / intensities
            auto defaults = createDefaultLights();
            const char *names[] = {"BenchLight0", "BenchLight1", "BenchLight2", "BenchLight3"};
            for (size_t i = 0; i < defaults.size(); ++i)
                makePointLight(scene, defaults[i].position, defaults[i].color,
                               defaults[i].intensity, defaults[i].radius, names[i]);
            std::printf("[Bench] Loaded preset 'Stress100' (100 entities, 4 lights)\n");
            return;
        }

        case BenchmarkPreset::Stress1000:
        {
            // 10 x 10 x 10 grid, spacing 1.5
            layoutCubeGrid(scene, cubeMeshRef, 10, 10, 10, 1.5f, glm::vec3(0, 0, 0), 0xBADC0DEu);
            // 16 lights: 4x4 grid around the scene
            LCG rng{0xDEADBEEFu};
            int idx = 0;
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i)
                {
                    glm::vec3 pos((i - 1.5f) * 6.0f,
                                  rng.range(-3.0f, 6.0f),
                                  (j - 1.5f) * 6.0f);
                    glm::vec3 color(rng.range(0.4f, 1.0f),
                                    rng.range(0.4f, 1.0f),
                                    rng.range(0.4f, 1.0f));
                    makePointLight(scene, pos, color, 2.0f, 5.0f,
                                   "BenchLight_" + std::to_string(idx++));
                }
            std::printf("[Bench] Loaded preset 'Stress1000' (1000 entities, 16 lights)\n");
            return;
        }

        case BenchmarkPreset::LightStress:
        {
            // 100-cube grid + 256 lights (lighting throughput stress)
            layoutCubeGrid(scene, cubeMeshRef, 10, 10, 1, 2.0f, glm::vec3(0, 1, 0), 0xC0FFEEu);
            // 256 lights: 16x16 grid in the x-z plane, y random in [2,6]
            LCG rng{0x5EED5EEDu};
            int idx = 0;
            for (int j = 0; j < 16; ++j)
                for (int i = 0; i < 16; ++i)
                {
                    glm::vec3 pos((i - 7.5f) * 1.5f,
                                  rng.range(2.0f, 6.0f),
                                  (j - 7.5f) * 1.5f);
                    glm::vec3 color(rng.range(0.3f, 1.0f),
                                    rng.range(0.3f, 1.0f),
                                    rng.range(0.3f, 1.0f));
                    makePointLight(scene, pos, color, 1.5f, 3.0f,
                                   "BenchLight_" + std::to_string(idx++));
                }
            std::printf("[Bench] Loaded preset 'LightStress' (100 entities, 256 lights)\n");
            return;
        }
        }
    }

} // namespace SceneSetup
