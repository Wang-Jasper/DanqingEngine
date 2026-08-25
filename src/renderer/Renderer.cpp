// ============================================================================
// Renderer.cpp - core two-pass deferred renderer (after the renderer split)
// ============================================================================
// This file keeps only:
//   - init() / drawFrame() / recordCommandBuffer() / cleanup() skeleton dispatch
//   - Geometry Pass / Lighting Pass resource creation (tightly coupled to Vulkan pipelines)
//   - UBO updates, sync object creation, depth / sampler creation
// Moved to separate modules:
//   - EditorUI     -> editor/EditorUI.h/.cpp
//   - GizmoRenderer -> renderer/GizmoRenderer.h/.cpp
//   - SceneSetup   -> scene/SceneSetup.h/.cpp
// ============================================================================

#include "Renderer.h"
#include "renderer/PipelineBuilder.h"
#include "renderer/LightVolumeMesh.h"
#include "scene/SceneSetup.h"
#include "core/asset/AssetRegistry.h"
#include "scripting/ScriptEngine.h" // Embedded Python interpreter
#include "utils/Frustum.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm> // std::sort for instance batching
#include <chrono>
#include <thread>
#include <cstring>
#include <filesystem>

// ============================================================================
// resolveAssetsRoot() - locate the `assets/` directory at runtime.
// AssetRegistry needs an absolute directory path; we walk up a few
// parent directories from the current working directory until one contains
// an `assets/` folder. This avoids hard-coding build-config-dependent paths
// (e.g. running from build/Debug/ vs. from the project root).
// ============================================================================
static std::filesystem::path resolveAssetsRoot()
{
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();
    for (int depth = 0; depth < 5; ++depth)
    {
        fs::path candidate = cwd / "assets";
        std::error_code ec;
        if (fs::is_directory(candidate, ec))
            return fs::absolute(candidate, ec);
        if (!cwd.has_parent_path() || cwd.parent_path() == cwd)
            break;
        cwd = cwd.parent_path();
    }
    // Last-ditch fallback: return CWD/assets even if it doesn't exist yet;
    // AssetRegistry::initialize will log an error and no-op.
    return fs::absolute(fs::current_path() / "assets");
}

// ============================================================================
// resolveScriptsRoot() - locate the `scripts/` directory at runtime.
// ScriptEngine adds this directory to sys.path so user scripts
// can be loaded by module name and resolve sibling imports. Same walk-up
// strategy as resolveAssetsRoot() so behaviour stays consistent regardless
// of whether the executable is launched from project root or build/Debug/.
// ============================================================================
static std::filesystem::path resolveScriptsRoot()
{
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();
    for (int depth = 0; depth < 5; ++depth)
    {
        fs::path candidate = cwd / "scripts";
        std::error_code ec;
        if (fs::is_directory(candidate, ec))
            return fs::absolute(candidate, ec);
        if (!cwd.has_parent_path() || cwd.parent_path() == cwd)
            break;
        cwd = cwd.parent_path();
    }
    // Same fallback policy as resolveAssetsRoot: hand back a path even if
    // the directory doesn't exist; ScriptEngine will simply skip the
    // sys.path injection if the path is empty / missing.
    return fs::absolute(fs::current_path() / "scripts");
}

// ============================================================================
// init() - renderer initialization entry point
// ============================================================================
void Renderer::init(GLFWwindow *win)
{
    window = win;

    vulkanContext.init(window);
    swapchain.init(vulkanContext, window);
    commandManager.init(vulkanContext, MAX_FRAMES_IN_FLIGHT);
    allocator.init(vulkanContext, &commandManager);
    textureManager.init(&vulkanContext, &allocator, &commandManager);

    // Initialise the asset GUID registry as early as possible so
    // anything that imports assets later (including the default scene setup
    // that does not actually load external assets, but drag-drop / scene load
    // that happen after init() do) can resolve GUIDs.
    AssetRegistry::instance().initialize(resolveAssetsRoot());

    depthFormat = DepthUtils::findDepthFormat(vulkanContext.getPhysicalDevice());

    createDepthResources();
    createGBufferResources();
    createSampler();
    createGeometryPassResources();
    createTextureResources(); // Lifetime-stable, survives onViewportResize

    // Hi-Z pass needs the depth image view + sampler, so it
    // must initialize after those three
    hiZPass.init(vulkanContext.getDevice(), allocator.getVma(), viewportExtent,
                 depthImage.imageView, gbufferSampler);

    // ShadowPass init. Must run before
    // createLightingPassResources, which binds shadowPass.shadowMapView() into
    // its descriptors.
    shadowPass.init(vulkanContext.getDevice(), allocator, static_cast<uint32_t>(shadowMapResolution));

    // SpotShadowPass init (4 slots x 2K depth). Also must run
    // before createLightingPassResources, which binds spotShadowPass.shadowMapView(s).
    spotShadowPass.init(vulkanContext.getDevice(), allocator, static_cast<uint32_t>(spotShadowMapResolution));

    // PointShadowPass init (4 slots x 1K cubemap).
    // Also must run before createLightingPassResources.
    pointShadowPass.init(vulkanContext.getDevice(), allocator, static_cast<uint32_t>(pointShadowMapResolution));

    // Built-in meshes + default scene + light sync (delegated to SceneSetup)
    SceneSetup::createBuiltinMeshes(vulkanContext, allocator, cubeMeshRef, planeMeshRef);

    // ECS -> Physics destroy hook: unregister the entity's PhysicsWorld body
    // before it is destroyed. Must be set before setupDefaultScene so any
    // destroyEntity/destroyAll works. It also detaches the entity's
    // per-entity Python script so PyEntity handles captured inside EntityScript
    // don't dangle after the entity is destroyed. Order matters: detachScript
    // first (releases Python-side globals + self), then unregisterEntity
    // (releases the physics body).
    scene.onBeforeDestroyEntity = [this](entt::entity e)
    {
        if (scriptEngine)
            scriptEngine->detachScript(e);
        PhysicsSystem::unregisterEntity(scene.registry, e, physicsWorld);
    };

    SceneSetup::setupDefaultScene(scene, cubeMeshRef, planeMeshRef);
    doSyncLightsToGPU();

    createLightingPassResources();

    // The instance SSBO can be bound to shadowPass only after
    // the Geometry Pass resources exist.
    shadowPass.bindInstanceBuffers(vulkanContext.getDevice(), instanceSSBOs,
                                   sizeof(InstanceData) * MAX_INSTANCES);

    // Light volume proxy resources (icosphere VB/IB + two pipelines)
    createLightVolumeResources();

    // Gizmo pipeline (delegated to GizmoRenderer)
    gizmoRenderer.createLinePipeline(vulkanContext.getDevice(), vulkanContext, allocator,
                                     geomDescriptorSetLayout, gbuffer.getColorFormats(), depthFormat);

    createSyncObjects();

    // Initialize the GPU profiler
    gpuProfiler.init(vulkanContext, MAX_FRAMES_IN_FLIGHT);

    // ImGui + offscreen RT (delegated to EditorUI)
    editorUI.init(*this);
    editorUI.initImGui();
    editorUI.createOffscreenResources();

    // SSAO must run before Composite (composite descriptor
    // references ssaoImage.imageView). Both rebuilt on resize.
    createSsaoResources();
    // Bloom must run before Composite as well (composite
    // descriptor references bloomMips[0].imageView).
    createBloomResources();
    // Composite pass needs editorUI.hdrImage to exist, so create
    // it after createOffscreenResources(). Rebuilt on every viewport resize.
    createCompositeResources();
    // FXAA pass samples editorUI.ldrPreFxaaImage. Same lifetime.
    createFxaaResources();

    lastFrameTime = static_cast<float>(glfwGetTime());

    // Embedded Python interpreter goes last so every subsystem
    // it might surface (ECS, camera, physics) is fully constructed before
    // any user script runs. Smoke test: load `scripts/smoke.py` if present
    // and print "hello from python" via the engine.log embedded function.
    scriptEngine = std::make_unique<ScriptEngine>();
    scriptEngine->init(*this, resolveScriptsRoot());
    if (scriptEngine->isInitialized())
    {
        // Publish ECSScene + Camera to the embedded `engine`
        // module BEFORE any user script loads, so `from engine import scene`
        // works at the top of list_scene.py et al. The pointers stay valid
        // for the entire engine lifetime; cleanup() calls
        // scriptEngine->shutdown() which also clears these attributes.
        scriptEngine->setSceneContext(&scene, &camera);

        const std::filesystem::path scriptsDir = resolveScriptsRoot();
        std::error_code ec;

        // Smoke test: prove `import engine; engine.log(...)` works.
        std::filesystem::path smokePath = scriptsDir / "smoke.py";
        if (std::filesystem::exists(smokePath, ec))
        {
            scriptEngine->loadScript(smokePath);
        }
        else
        {
            std::cout << "[ScriptEngine] No smoke.py at " << smokePath.generic_string()
                      << " (skipping smoke test).\n";
        }

        // Binding verification: exercise glm + Component bindings
        // if the file is present. We deliberately load this AFTER smoke.py
        // so a binding regression doesn't suppress the simpler smoke output.
        // The script either logs "binding verification PASSED" or
        // raises - failure path goes through the standard scriptFaulted flow.
        std::filesystem::path verifyPath = scriptsDir / "verify_bindings.py";
        if (std::filesystem::exists(verifyPath, ec))
        {
            scriptEngine->loadScript(verifyPath);
        }

        // Verification: list_scene.py walks `engine.scene` and
        // logs every entity. Loaded last so we know glm + Component bindings
        // already passed before exercising the Entity / Scene handles.
        std::filesystem::path listPath = scriptsDir / "list_scene.py";
        if (std::filesystem::exists(listPath, ec))
        {
            scriptEngine->loadScript(listPath);
        }

        // Demo: spin_cube.py registers on_start / on_update /
        // on_stop. Loaded LAST so it occupies the active-script slot in
        // ScriptEngine::Impl::userModule by the time the user clicks Play.
        // The earlier scripts already finished their top-level self-tests.
        //
        // spin_cube.py is no longer hard-coded into the legacy
        // single-slot path. Instead, the default scene attaches a
        // ScriptComponent{scriptPath="spin_cube.py"} to the Cube entity
        // (see SceneSetup::setupDefaultScene), and ScriptEngine's
        // syncFromScene() picks it up on the next drawFrame() tick.
        // Removing the loadScript() call here means smoke / verify / list
        // remain the only top-level execs at start-up; per-entity user
        // scripts go through the ScriptComponent pipeline exclusively.
    }

    std::cout << "[Renderer] Initialized. Deferred Rendering (2-pass).\n";
}

// ============================================================================
// createGBufferResources()
// ============================================================================
void Renderer::createGBufferResources()
{
    gbuffer.init(vulkanContext.getDevice(), allocator.getVma(), viewportExtent);
}

// ============================================================================
// createDepthResources()
// ============================================================================
void Renderer::createDepthResources()
{
    depthImage = DepthUtils::createDepthImage(
        vulkanContext.getDevice(), allocator.getVma(), viewportExtent, depthFormat);
}

// ============================================================================
// createSampler()
// ============================================================================
void Renderer::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(vulkanContext.getDevice(), &samplerInfo, nullptr, &gbufferSampler) != VK_SUCCESS)
        throw std::runtime_error("Failed to create G-Buffer sampler!");
}

// ============================================================================
// createGeometryPassResources()
// ============================================================================
void Renderer::createGeometryPassResources()
{
    VkDevice device = vulkanContext.getDevice();

    // Geom set 0 now has 2 bindings
    //   binding 0: UBO  (view/proj)
    //   binding 2: SSBO (per-instance model + material), used by geometry.vert
    std::array<VkDescriptorSetLayoutBinding, 2> geomBindings{};
    geomBindings[0].binding = 0;
    geomBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    geomBindings[0].descriptorCount = 1;
    geomBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    geomBindings[1].binding = 2;
    geomBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    geomBindings[1].descriptorCount = 1;
    geomBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(geomBindings.size());
    layoutInfo.pBindings = geomBindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &geomDescriptorSetLayout);

    // texDescriptorSetLayout / texDescriptorPool / defaultTexDescriptorSet
    // are now created ONCE in createTextureResources() (called from init()).
    // We must not recreate them on resize, otherwise sets handed out by
    // allocateTextureDescriptorSet() (e.g. TextureStress preset) would be
    // silently invalidated.
    if (texDescriptorSetLayout == VK_NULL_HANDLE)
    {
        // First call (from init() before createTextureResources()) is allowed
        // to land here briefly because createGeometryPassResources() needs the
        // layout to build the pipeline layout below. We create just the layout
        // and let createTextureResources() do the rest.
        VkDescriptorSetLayoutBinding texBinding{};
        texBinding.binding = 0;
        texBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texBinding.descriptorCount = 1;
        texBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo texLayoutInfo{};
        texLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        texLayoutInfo.bindingCount = 1;
        texLayoutInfo.pBindings = &texBinding;
        vkCreateDescriptorSetLayout(device, &texLayoutInfo, nullptr, &texDescriptorSetLayout);
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GeometryPushConstants);

    std::array<VkDescriptorSetLayout, 2> setLayouts = {geomDescriptorSetLayout, texDescriptorSetLayout};
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    plInfo.pSetLayouts = setLayouts.data();
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushConstantRange;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &geomPipelineLayout);

    std::string shaderDir = SHADER_DIR;
    auto bindingDesc = Vertex::getBindingDescription();
    auto attributeDescs = Vertex::getAttributeDescriptions();

    PipelineBuilder builder;
    geomPipeline = builder
                       .setShaders(device, shaderDir + "/geometry.vert.spv", shaderDir + "/geometry.frag.spv")
                       .setVertexInput(bindingDesc, attributeDescs.data(), static_cast<uint32_t>(attributeDescs.size()))
                       .setInputAssembly()
                       .setViewportDynamic()
                       .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                       .setMultisampling()
                       .setDepthStencil(true, true, VK_COMPARE_OP_LESS)
                       .setColorBlending(false)
                       .setColorAttachmentFormats(gbuffer.getColorFormats())
                       .setDepthAttachmentFormat(depthFormat)
                       .build(device, geomPipelineLayout);
    builder.cleanupShaderModules(device);

    geomUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    geomUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    // Per-frame host-mapped instance SSBO (one slot per frame in flight).
    instanceSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        geomUniformBuffers[i] = allocator.createBuffer(
            sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), geomUniformBuffers[i].allocation, &geomUniformBuffersMapped[i]);

        instanceSSBOs[i] = allocator.createBuffer(
            sizeof(InstanceData) * MAX_INSTANCES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), instanceSSBOs[i].allocation, &instanceSSBOsMapped[i]);
    }

    // Pool sized for UBO + instance SSBO per frame.
    std::array<VkDescriptorPoolSize, 2> geomPoolSizes{};
    geomPoolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT};
    geomPoolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES_IN_FLIGHT};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(geomPoolSizes.size());
    poolInfo.pPoolSizes = geomPoolSizes.data();
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &geomDescriptorPool);

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, geomDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = geomDescriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    geomDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocInfo, geomDescriptorSets.data());

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = geomUniformBuffers[i].buffer;
        bufInfo.offset = 0;
        bufInfo.range = sizeof(UniformBufferObject);

        // Bind per-frame instance SSBO to set=0 binding=2.
        VkDescriptorBufferInfo instInfo{};
        instInfo.buffer = instanceSSBOs[i].buffer;
        instInfo.offset = 0;
        instInfo.range = sizeof(InstanceData) * MAX_INSTANCES;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = geomDescriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = geomDescriptorSets[i];
        writes[1].dstBinding = 2;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &instInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::cout << "[Renderer] Geometry Pass resources created.\n";
}

// ============================================================================
// createTextureResources()
// ============================================================================
// Lifetime-stable texture pool / default white set. Called ONCE from init();
// NOT touched by onViewportResize() so TextureStress and any future user-loaded
// material descriptor sets stay valid across resizes.
void Renderer::createTextureResources()
{
    VkDevice device = vulkanContext.getDevice();

    // texDescriptorSetLayout may have been created during the very first call
    // to createGeometryPassResources() (init order: createGeometryPassResources
    // runs before createTextureResources). Create it here only if nobody else
    // has yet.
    if (texDescriptorSetLayout == VK_NULL_HANDLE)
    {
        VkDescriptorSetLayoutBinding texBinding{};
        texBinding.binding = 0;
        texBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texBinding.descriptorCount = 1;
        texBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo texLayoutInfo{};
        texLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        texLayoutInfo.bindingCount = 1;
        texLayoutInfo.pBindings = &texBinding;
        vkCreateDescriptorSetLayout(device, &texLayoutInfo, nullptr, &texDescriptorSetLayout);
    }

    VkDescriptorPoolSize texPoolSize{};
    texPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texPoolSize.descriptorCount = 64;

    VkDescriptorPoolCreateInfo texPoolInfo{};
    texPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    texPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    texPoolInfo.poolSizeCount = 1;
    texPoolInfo.pPoolSizes = &texPoolSize;
    texPoolInfo.maxSets = 64;
    vkCreateDescriptorPool(device, &texPoolInfo, nullptr, &texDescriptorPool);

    auto whiteTex = textureManager.getDefaultWhiteTexture();

    VkDescriptorSetAllocateInfo texAllocInfo{};
    texAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    texAllocInfo.descriptorPool = texDescriptorPool;
    texAllocInfo.descriptorSetCount = 1;
    texAllocInfo.pSetLayouts = &texDescriptorSetLayout;
    vkAllocateDescriptorSets(device, &texAllocInfo, &defaultTexDescriptorSet);

    VkDescriptorImageInfo defaultImgInfo{};
    defaultImgInfo.sampler = whiteTex->sampler;
    defaultImgInfo.imageView = whiteTex->image.imageView;
    defaultImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet texWrite{};
    texWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    texWrite.dstSet = defaultTexDescriptorSet;
    texWrite.dstBinding = 0;
    texWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texWrite.descriptorCount = 1;
    texWrite.pImageInfo = &defaultImgInfo;
    vkUpdateDescriptorSets(device, 1, &texWrite, 0, nullptr);

    std::cout << "[Renderer] Texture pool resources created.\n";
}

// ============================================================================
// cleanupTextureResources()
// ============================================================================
void Renderer::cleanupTextureResources()
{
    VkDevice device = vulkanContext.getDevice();
    // Free any sets handed out by allocateTextureDescriptorSet() before
    // destroying the pool that owns them.
    freeAllAllocatedTextureDescriptorSets();
    if (texDescriptorPool)
    {
        vkDestroyDescriptorPool(device, texDescriptorPool, nullptr);
        texDescriptorPool = VK_NULL_HANDLE;
        defaultTexDescriptorSet = VK_NULL_HANDLE; // implicitly freed with pool
    }
    if (texDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, texDescriptorSetLayout, nullptr);
        texDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

// ============================================================================
// createLightingPassResources()
// ============================================================================
void Renderer::createLightingPassResources()
{
    VkDevice device = vulkanContext.getDevice();

    // Binding 7 = Hi-Z pyramid sampling (debugMode == 5
    // visualization & reserved for future SSR/occlusion)
    // Binding 8 = directional shadow map (sampler2DShadow)
    //                binding 9 = directional shadow UBO (lightVP + flags)
    // Binding 10 = spot shadow map array[4] (sampler2DShadow)
    //                binding 11 = spot shadow UBO (mat4 spotLightVP[4] + int validSlots)
    std::array<VkDescriptorSetLayoutBinding, 14> bindings{};
    for (uint32_t i = 0; i < 3; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    // The light volume vertex shader also reads the point SSBO (binding 4)
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    // The light volume vertex shader also reads the spot SSBO (binding 6)
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Directional shadow bindings
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Spot shadow map[4] + spot shadow UBO
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[10].descriptorCount = SpotShadowPass::MAX_SPOT_SHADOW_SLOTS;
    bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Point cubemap shadow map[4] + point shadow UBO
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = PointShadowPass::MAX_POINT_SHADOW_SLOTS;
    bindings[12].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &lightDescriptorSetLayout);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &lightDescriptorSetLayout;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &lightPipelineLayout);

    std::string shaderDir = SHADER_DIR;
    PipelineBuilder builder;
    lightPipeline = builder
                        .setShaders(device, shaderDir + "/lighting.vert.spv", shaderDir + "/lighting.frag.spv")
                        .setNoVertexInput()
                        .setInputAssembly()
                        .setViewportDynamic()
                        .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                        .setMultisampling()
                        .setDepthStencil(false, false)
                        .setColorBlending(false)
                        .setColorAttachmentFormat(VK_FORMAT_R16G16B16A16_SFLOAT) // HDR FP16 lighting RT
                        .build(device, lightPipelineLayout);
    builder.cleanupShaderModules(device);

    lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    lightUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        lightUniformBuffers[i] = allocator.createBuffer(
            sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), lightUniformBuffers[i].allocation, &lightUniformBuffersMapped[i]);
    }

    // Directional shadow UBO (lighting binding 9).
    // Layout must match DirShadowUBO in the shader: mat4 + int + float + 2*float
    // pad, totalling 80 bytes (mat4=64, int=4, float=4, pad=8 -> 80).
    struct DirShadowUBOLayout
    {
        alignas(16) glm::mat4 lightViewProj;
        int enabled;
        float invMapSize;
        float _pad0;
        float _pad1;
    };
    dirShadowUBOs.resize(MAX_FRAMES_IN_FLIGHT);
    dirShadowUBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        dirShadowUBOs[i] = allocator.createBuffer(
            sizeof(DirShadowUBOLayout), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), dirShadowUBOs[i].allocation, &dirShadowUBOsMapped[i]);
        // Initialize to "no shadow" so first frame before updateUniformBuffer() runs
        // doesn't read uninitialized memory in shader.
        DirShadowUBOLayout init{};
        init.lightViewProj = glm::mat4(1.0f);
        init.enabled = 0;
        init.invMapSize = 1.0f / static_cast<float>(shadowMapResolution);
        std::memcpy(dirShadowUBOsMapped[i], &init, sizeof(init));
    }

    // Spot shadow UBO (lighting binding 11).
    // Layout must match SpotShadowUBO in lighting.frag:
    //   mat4 spotLightVP[4] (256B) + ivec4 validSlots/_pad (16B) + float invMapSize (4B) + 12B pad = 288B
    // validSlots.x = active slot count this frame (0..4); the rest of the ivec3 is reserved.
    struct SpotShadowUBOLayout
    {
        alignas(16) glm::mat4 spotLightVP[SpotShadowPass::MAX_SPOT_SHADOW_SLOTS];
        alignas(16) glm::ivec4 validSlots; // x = slot count
        alignas(16) glm::vec4 params;      // x = invMapSize
    };
    spotShadowUBOs.resize(MAX_FRAMES_IN_FLIGHT);
    spotShadowUBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        spotShadowUBOs[i] = allocator.createBuffer(
            sizeof(SpotShadowUBOLayout), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), spotShadowUBOs[i].allocation, &spotShadowUBOsMapped[i]);
        SpotShadowUBOLayout init{};
        for (uint32_t k = 0; k < SpotShadowPass::MAX_SPOT_SHADOW_SLOTS; ++k)
            init.spotLightVP[k] = glm::mat4(1.0f);
        init.validSlots = glm::ivec4(0);
        init.params = glm::vec4(1.0f / static_cast<float>(spotShadowMapResolution), 0, 0, 0);
        std::memcpy(spotShadowUBOsMapped[i], &init, sizeof(init));
    }

    // Point shadow UBO (lighting binding 13).
    // Layout must match PointShadowUBO in lighting.frag:
    //   vec4 lightPosRange[4] (64B) + ivec4 validSlots (16B) + vec4 params[4] (64B) = 144B
    //   params[k].xy = (normalBiasMul, depthBiasMul), matching PointLightComponent.shadow*Bias.
    struct PointShadowUBOLayout
    {
        alignas(16) glm::vec4 lightPosRange[PointShadowPass::MAX_POINT_SHADOW_SLOTS];
        alignas(16) glm::ivec4 validSlots;                                     // x = slot count
        alignas(16) glm::vec4 params[PointShadowPass::MAX_POINT_SHADOW_SLOTS]; // per-slot bias multipliers
    };
    pointShadowUBOs.resize(MAX_FRAMES_IN_FLIGHT);
    pointShadowUBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        pointShadowUBOs[i] = allocator.createBuffer(
            sizeof(PointShadowUBOLayout), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), pointShadowUBOs[i].allocation, &pointShadowUBOsMapped[i]);
        PointShadowUBOLayout init{};
        for (uint32_t k = 0; k < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++k)
            init.lightPosRange[k] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        init.validSlots = glm::ivec4(0);
        for (uint32_t k = 0; k < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++k)
            init.params[k] = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f); // default multiplier 1.0
        std::memcpy(pointShadowUBOsMapped[i], &init, sizeof(init));
    }

    // The 3 light SSBOs are host-mapped per-frame buffers with fixed capacity
    lightSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    lightSSBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    dirLightSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    dirLightSSBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    spotLightSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    spotLightSSBOsMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        lightSSBOs[i] = allocator.createBuffer(
            sizeof(PointLight) * MAX_POINT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), lightSSBOs[i].allocation, &lightSSBOsMapped[i]);

        dirLightSSBOs[i] = allocator.createBuffer(
            sizeof(DirectionalLight) * MAX_DIR_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), dirLightSSBOs[i].allocation, &dirLightSSBOsMapped[i]);

        spotLightSSBOs[i] = allocator.createBuffer(
            sizeof(SpotLight) * MAX_SPOT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        vmaMapMemory(allocator.getVma(), spotLightSSBOs[i].allocation, &spotLightSSBOsMapped[i]);
    }

    // +1 sampler slot (Hi-Z)
    // +1 sampler (shadow map) + 1 UBO (shadow VP)
    // +4 samplers (spot shadow array[4]) + 1 UBO (spot shadow VP)
    // +4 samplers (point cubemap shadow array[4]) + 1 UBO (point shadow)
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    (5 + SpotShadowPass::MAX_SPOT_SHADOW_SLOTS + PointShadowPass::MAX_POINT_SHADOW_SLOTS) * MAX_FRAMES_IN_FLIGHT};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 * MAX_FRAMES_IN_FLIGHT};
    poolSizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * MAX_FRAMES_IN_FLIGHT};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &lightDescriptorPool);

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, lightDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocDescInfo{};
    allocDescInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocDescInfo.descriptorPool = lightDescriptorPool;
    allocDescInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocDescInfo.pSetLayouts = layouts.data();
    lightDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocDescInfo, lightDescriptorSets.data());

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        std::array<VkDescriptorImageInfo, 3> imageInfos{};
        for (uint32_t j = 0; j < GBuffer::COUNT; j++)
        {
            imageInfos[j].sampler = gbufferSampler;
            imageInfos[j].imageView = gbuffer.getImageView(j);
            imageInfos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = lightUniformBuffers[i].buffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(LightingUBO);

        // Per-frame host-mapped SSBO, binding the whole fixed capacity
        VkDescriptorBufferInfo ssboInfo{};
        ssboInfo.buffer = lightSSBOs[i].buffer;
        ssboInfo.offset = 0;
        ssboInfo.range = sizeof(PointLight) * MAX_POINT_LIGHTS;

        VkDescriptorBufferInfo dirSsboInfo{};
        dirSsboInfo.buffer = dirLightSSBOs[i].buffer;
        dirSsboInfo.offset = 0;
        dirSsboInfo.range = sizeof(DirectionalLight) * MAX_DIR_LIGHTS;

        VkDescriptorBufferInfo spotSsboInfo{};
        spotSsboInfo.buffer = spotLightSSBOs[i].buffer;
        spotSsboInfo.offset = 0;
        spotSsboInfo.range = sizeof(SpotLight) * MAX_SPOT_LIGHTS;

        // Binding 7 writes the Hi-Z sampling
        VkDescriptorImageInfo hizInfo{};
        hizInfo.sampler = hiZPass.sampler();
        hizInfo.imageView = hiZPass.fullView();
        hizInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // If HiZPass is not initialized (rare, e.g. resize ordering), fall back
        // to GBuffer albedo so the descriptor stays valid (no validation error);
        // the shader only reads it when debugMode == 5.
        if (hizInfo.imageView == VK_NULL_HANDLE)
        {
            hizInfo.sampler = gbufferSampler;
            hizInfo.imageView = gbuffer.getImageView(GBuffer::ALBEDO);
        }

        // Binding 8 = shadow map (sampler2DShadow)
        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler = shadowPass.shadowSampler();
        shadowInfo.imageView = shadowPass.shadowMapView();
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (shadowInfo.imageView == VK_NULL_HANDLE || shadowInfo.sampler == VK_NULL_HANDLE)
        {
            // Fallback: avoids validation errors from null handles while the
            // context is not fully initialized; only active during the brief
            // init window before shadowPass.init() rewrites it.
            shadowInfo.sampler = gbufferSampler;
            shadowInfo.imageView = gbuffer.getImageView(GBuffer::ALBEDO);
        }

        // Binding 9 = shadow UBO (mat4 + flags)
        VkDescriptorBufferInfo shadowUboInfo{};
        shadowUboInfo.buffer = dirShadowUBOs[i].buffer;
        shadowUboInfo.offset = 0;
        shadowUboInfo.range = VK_WHOLE_SIZE;

        // Binding 10 = spot shadow map array[4] (sampler2DShadow[4])
        std::array<VkDescriptorImageInfo, SpotShadowPass::MAX_SPOT_SHADOW_SLOTS> spotShadowInfos{};
        for (uint32_t s = 0; s < SpotShadowPass::MAX_SPOT_SHADOW_SLOTS; ++s)
        {
            spotShadowInfos[s].sampler = spotShadowPass.shadowSampler();
            spotShadowInfos[s].imageView = spotShadowPass.shadowMapView(s);
            spotShadowInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // Fallback: avoids null-handle validation errors if SpotShadowPass
            // is not initialized yet.
            if (spotShadowInfos[s].imageView == VK_NULL_HANDLE || spotShadowInfos[s].sampler == VK_NULL_HANDLE)
            {
                spotShadowInfos[s].sampler = gbufferSampler;
                spotShadowInfos[s].imageView = gbuffer.getImageView(GBuffer::ALBEDO);
            }
        }
        // Binding 11 = spot shadow UBO
        VkDescriptorBufferInfo spotShadowUboInfo{};
        spotShadowUboInfo.buffer = spotShadowUBOs[i].buffer;
        spotShadowUboInfo.offset = 0;
        spotShadowUboInfo.range = VK_WHOLE_SIZE;

        // Binding 12 = point cubemap shadow array[4].
        std::array<VkDescriptorImageInfo, PointShadowPass::MAX_POINT_SHADOW_SLOTS> pointShadowInfos{};
        for (uint32_t s = 0; s < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++s)
        {
            pointShadowInfos[s].sampler = pointShadowPass.shadowSampler();
            pointShadowInfos[s].imageView = pointShadowPass.shadowCubeView(s);
            pointShadowInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // Fallback: not initialized / null view -> degrade to albedo to
            // avoid validation errors (albedo is a 2D image and cannot bind to
            // samplerCubeShadow, but this fallback only applies during the
            // brief init window. If PointShadowPass.init() is ever skipped,
            // the resulting validation error is the signal.)
            if (pointShadowInfos[s].imageView == VK_NULL_HANDLE || pointShadowInfos[s].sampler == VK_NULL_HANDLE)
            {
                pointShadowInfos[s].sampler = gbufferSampler;
                pointShadowInfos[s].imageView = gbuffer.getImageView(GBuffer::ALBEDO);
            }
        }
        // Binding 13 = point shadow UBO
        VkDescriptorBufferInfo pointShadowUboInfo{};
        pointShadowUboInfo.buffer = pointShadowUBOs[i].buffer;
        pointShadowUboInfo.offset = 0;
        pointShadowUboInfo.range = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 14> writes{};
        for (uint32_t j = 0; j < 3; j++)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[j].dstSet = lightDescriptorSets[i];
            writes[j].dstBinding = j;
            writes[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[j].descriptorCount = 1;
            writes[j].pImageInfo = &imageInfos[j];
        }
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = lightDescriptorSets[i];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &uboInfo;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = lightDescriptorSets[i];
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &ssboInfo;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = lightDescriptorSets[i];
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &dirSsboInfo;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = lightDescriptorSets[i];
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &spotSsboInfo;
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet = lightDescriptorSets[i];
        writes[7].dstBinding = 7;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].descriptorCount = 1;
        writes[7].pImageInfo = &hizInfo;
        // Shadow map + shadow UBO
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[8].dstSet = lightDescriptorSets[i];
        writes[8].dstBinding = 8;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[8].descriptorCount = 1;
        writes[8].pImageInfo = &shadowInfo;
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[9].dstSet = lightDescriptorSets[i];
        writes[9].dstBinding = 9;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[9].descriptorCount = 1;
        writes[9].pBufferInfo = &shadowUboInfo;

        // Binding 10 = spot shadow map array[4], binding 11 = spot shadow UBO
        writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[10].dstSet = lightDescriptorSets[i];
        writes[10].dstBinding = 10;
        writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[10].descriptorCount = SpotShadowPass::MAX_SPOT_SHADOW_SLOTS;
        writes[10].pImageInfo = spotShadowInfos.data();
        writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[11].dstSet = lightDescriptorSets[i];
        writes[11].dstBinding = 11;
        writes[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[11].descriptorCount = 1;
        writes[11].pBufferInfo = &spotShadowUboInfo;

        // Binding 12 = point shadow cubemap array[4], binding 13 = point shadow UBO
        writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[12].dstSet = lightDescriptorSets[i];
        writes[12].dstBinding = 12;
        writes[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[12].descriptorCount = PointShadowPass::MAX_POINT_SHADOW_SLOTS;
        writes[12].pImageInfo = pointShadowInfos.data();
        writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[13].dstSet = lightDescriptorSets[i];
        writes[13].dstBinding = 13;
        writes[13].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[13].descriptorCount = 1;
        writes[13].pBufferInfo = &pointShadowUboInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::cout << "[Renderer] Lighting Pass resources created (capacity: "
              << MAX_POINT_LIGHTS << " point / " << MAX_DIR_LIGHTS << " dir / "
              << MAX_SPOT_LIGHTS << " spot, host-mapped per-frame SSBO).\n";
}

// ============================================================================
// updateLightingDescriptorSets() - fast-path descriptor rewrite
// ----------------------------------------------------------------------------
// Fast-path helper: inside onViewportResize_RT() the size-dependent images
// (GBuffer / Depth / HiZ / SSAO) have been rebuilt, but lightDescriptorPool /
// lightPipeline / lightDescriptorSetLayout / lightDescriptorSets are NOT
// destroyed (the layout is unchanged, so existing descriptor set handles stay
// valid). This function just re-writes the descriptors so all 14 bindings
// point at the new imageView / sampler / buffer.
//
// Its contents must stay in sync with the 14-write loop in
// createLightingPassResources() (buffer bindings are refreshed too; the
// UBO/SSBO handles don't change on the fast path, so re-writing them is legal
// and free).
// ============================================================================
void Renderer::updateLightingDescriptorSets()
{
    VkDevice device = vulkanContext.getDevice();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        std::array<VkDescriptorImageInfo, 3> imageInfos{};
        for (uint32_t j = 0; j < GBuffer::COUNT; j++)
        {
            imageInfos[j].sampler = gbufferSampler;
            imageInfos[j].imageView = gbuffer.getImageView(j);
            imageInfos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = lightUniformBuffers[i].buffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(LightingUBO);

        VkDescriptorBufferInfo ssboInfo{};
        ssboInfo.buffer = lightSSBOs[i].buffer;
        ssboInfo.offset = 0;
        ssboInfo.range = sizeof(PointLight) * MAX_POINT_LIGHTS;

        VkDescriptorBufferInfo dirSsboInfo{};
        dirSsboInfo.buffer = dirLightSSBOs[i].buffer;
        dirSsboInfo.offset = 0;
        dirSsboInfo.range = sizeof(DirectionalLight) * MAX_DIR_LIGHTS;

        VkDescriptorBufferInfo spotSsboInfo{};
        spotSsboInfo.buffer = spotLightSSBOs[i].buffer;
        spotSsboInfo.offset = 0;
        spotSsboInfo.range = sizeof(SpotLight) * MAX_SPOT_LIGHTS;

        VkDescriptorImageInfo hizInfo{};
        hizInfo.sampler = hiZPass.sampler();
        hizInfo.imageView = hiZPass.fullView();
        hizInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (hizInfo.imageView == VK_NULL_HANDLE)
        {
            hizInfo.sampler = gbufferSampler;
            hizInfo.imageView = gbuffer.getImageView(GBuffer::ALBEDO);
        }

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler = shadowPass.shadowSampler();
        shadowInfo.imageView = shadowPass.shadowMapView();
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (shadowInfo.imageView == VK_NULL_HANDLE || shadowInfo.sampler == VK_NULL_HANDLE)
        {
            shadowInfo.sampler = gbufferSampler;
            shadowInfo.imageView = gbuffer.getImageView(GBuffer::ALBEDO);
        }

        VkDescriptorBufferInfo shadowUboInfo{};
        shadowUboInfo.buffer = dirShadowUBOs[i].buffer;
        shadowUboInfo.offset = 0;
        shadowUboInfo.range = VK_WHOLE_SIZE;

        std::array<VkDescriptorImageInfo, SpotShadowPass::MAX_SPOT_SHADOW_SLOTS> spotShadowInfos{};
        for (uint32_t s = 0; s < SpotShadowPass::MAX_SPOT_SHADOW_SLOTS; ++s)
        {
            spotShadowInfos[s].sampler = spotShadowPass.shadowSampler();
            spotShadowInfos[s].imageView = spotShadowPass.shadowMapView(s);
            spotShadowInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (spotShadowInfos[s].imageView == VK_NULL_HANDLE || spotShadowInfos[s].sampler == VK_NULL_HANDLE)
            {
                spotShadowInfos[s].sampler = gbufferSampler;
                spotShadowInfos[s].imageView = gbuffer.getImageView(GBuffer::ALBEDO);
            }
        }
        VkDescriptorBufferInfo spotShadowUboInfo{};
        spotShadowUboInfo.buffer = spotShadowUBOs[i].buffer;
        spotShadowUboInfo.offset = 0;
        spotShadowUboInfo.range = VK_WHOLE_SIZE;

        std::array<VkDescriptorImageInfo, PointShadowPass::MAX_POINT_SHADOW_SLOTS> pointShadowInfos{};
        for (uint32_t s = 0; s < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++s)
        {
            pointShadowInfos[s].sampler = pointShadowPass.shadowSampler();
            pointShadowInfos[s].imageView = pointShadowPass.shadowCubeView(s);
            pointShadowInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (pointShadowInfos[s].imageView == VK_NULL_HANDLE || pointShadowInfos[s].sampler == VK_NULL_HANDLE)
            {
                pointShadowInfos[s].sampler = gbufferSampler;
                pointShadowInfos[s].imageView = gbuffer.getImageView(GBuffer::ALBEDO);
            }
        }
        VkDescriptorBufferInfo pointShadowUboInfo{};
        pointShadowUboInfo.buffer = pointShadowUBOs[i].buffer;
        pointShadowUboInfo.offset = 0;
        pointShadowUboInfo.range = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 14> writes{};
        for (uint32_t j = 0; j < 3; j++)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[j].dstSet = lightDescriptorSets[i];
            writes[j].dstBinding = j;
            writes[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[j].descriptorCount = 1;
            writes[j].pImageInfo = &imageInfos[j];
        }
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = lightDescriptorSets[i];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &uboInfo;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = lightDescriptorSets[i];
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &ssboInfo;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = lightDescriptorSets[i];
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &dirSsboInfo;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = lightDescriptorSets[i];
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &spotSsboInfo;
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet = lightDescriptorSets[i];
        writes[7].dstBinding = 7;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].descriptorCount = 1;
        writes[7].pImageInfo = &hizInfo;
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[8].dstSet = lightDescriptorSets[i];
        writes[8].dstBinding = 8;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[8].descriptorCount = 1;
        writes[8].pImageInfo = &shadowInfo;
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[9].dstSet = lightDescriptorSets[i];
        writes[9].dstBinding = 9;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[9].descriptorCount = 1;
        writes[9].pBufferInfo = &shadowUboInfo;
        writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[10].dstSet = lightDescriptorSets[i];
        writes[10].dstBinding = 10;
        writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[10].descriptorCount = SpotShadowPass::MAX_SPOT_SHADOW_SLOTS;
        writes[10].pImageInfo = spotShadowInfos.data();
        writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[11].dstSet = lightDescriptorSets[i];
        writes[11].dstBinding = 11;
        writes[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[11].descriptorCount = 1;
        writes[11].pBufferInfo = &spotShadowUboInfo;
        writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[12].dstSet = lightDescriptorSets[i];
        writes[12].dstBinding = 12;
        writes[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[12].descriptorCount = PointShadowPass::MAX_POINT_SHADOW_SLOTS;
        writes[12].pImageInfo = pointShadowInfos.data();
        writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[13].dstSet = lightDescriptorSets[i];
        writes[13].dstBinding = 13;
        writes[13].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[13].descriptorCount = 1;
        writes[13].pBufferInfo = &pointShadowUboInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

// ============================================================================
// createLightVolumeResources()
// ----------------------------------------------------------------------------
// Creates the light volume proxy mesh (icosphere) + 2 pipelines (point/spot).
//   - VB/IB are GPU-only, uploaded via staging
//   - Share light_volume.vert; a specialization constant LIGHT_TYPE derives the
//     two fragment pipelines (light_volume_point.frag / light_volume_spot.frag)
//   - Push constants: mat4 viewProj (64B), written every frame by recordCommandBuffer
//   - No depth attachment; the frag uses G-Buffer position to decide "pixel inside sphere"
//   - CullMode = FRONT (cull front faces, keep backs) - so fragments still fire
//     when the camera is inside the sphere
//   - DepthTest/DepthWrite both off
//   - Additive blending: multi-light contributions accumulate onto the offscreen RT
// ============================================================================
void Renderer::createLightVolumeResources()
{
    VkDevice device = vulkanContext.getDevice();

    // --- 1. Generate the icosphere ---
    auto mesh = LightVolumeMesh::generateIcosphere(1, 1.02f); // 42 vert, 80 tri
    lightVolumeIndexCount = static_cast<uint32_t>(mesh.indices.size());

    lightVolumeVB = allocator.createBufferWithStaging(
        vulkanContext, mesh.positions.data(),
        sizeof(glm::vec3) * mesh.positions.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    lightVolumeIB = allocator.createBufferWithStaging(
        vulkanContext, mesh.indices.data(),
        sizeof(uint32_t) * mesh.indices.size(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    // --- 2. Pipeline layout: shared lightDescriptorSetLayout + push constants ---
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset = 0;
    pc.size = sizeof(glm::mat4); // viewProj

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &lightDescriptorSetLayout; // reuse the lighting pass descriptor set
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &lightVolumePipelineLayout);

    // --- 3. Vertex input: vec3 position, binding 0 ---
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(glm::vec3);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.binding = 0;
    attr.location = 0;
    attr.format = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset = 0;

    // --- 4. Build the two pipelines (own vert + frag each) ---
    std::string shaderDir = SHADER_DIR;

    auto buildPipeline = [&](const std::string &vertPath, const std::string &fragPath) -> VkPipeline
    {
        PipelineBuilder builder;
        builder.setShaders(device, vertPath, fragPath)
            .setVertexInput(binding, &attr, 1)
            .setInputAssembly()
            .setViewportDynamic()
            .setRasterizer(VK_POLYGON_MODE_FILL,
                           VK_CULL_MODE_FRONT_BIT, // cull front faces, keep backs
                           VK_FRONT_FACE_COUNTER_CLOCKWISE)
            .setMultisampling()
            .setDepthStencil(false, false)                            // no depth test/write
            .setAdditiveBlending()                                    // additive blending
            .setColorAttachmentFormat(VK_FORMAT_R16G16B16A16_SFLOAT); // HDR FP16 lighting RT (matches lighting pipeline)
        VkPipeline pipe = builder.build(device, lightVolumePipelineLayout);
        builder.cleanupShaderModules(device);
        return pipe;
    };

    lightVolumePointPipeline = buildPipeline(shaderDir + "/light_volume_point.vert.spv",
                                             shaderDir + "/light_volume_point.frag.spv");
    lightVolumeSpotPipeline = buildPipeline(shaderDir + "/light_volume_spot.vert.spv",
                                            shaderDir + "/light_volume_spot.frag.spv");
    std::cout << "[Renderer] Light Volume resources created ("
              << mesh.positions.size() << " vert / " << (lightVolumeIndexCount / 3)
              << " tri icosphere, 2 pipelines).\n";
}

// ============================================================================
// doSyncLightsToGPU() - delegated to SceneSetup
// ============================================================================
void Renderer::doSyncLightsToGPU()
{
    // Lights are gathered + uploaded automatically every frame,
    // no manual sync needed. Kept as an empty function for external callers
    // (init / onViewportResize / EditorUI).
}

// ============================================================================
// updateLightBuffers()
//   Every frame: gather + frustum cull + memcpy into the current frame's
//   host-mapped SSBO
// ============================================================================
void Renderer::updateLightBuffers(uint32_t frameIndex)
{
    // Step 1: gather all lights from the ECS into temp buffers
    std::vector<PointLight> allPoints;
    std::vector<DirectionalLight> allDirs;
    std::vector<SpotLight> allSpots;
    Systems::gatherLights(scene.registry, allPoints);
    Systems::gatherDirLights(scene.registry, allDirs);
    Systems::gatherSpotLights(scene.registry, allSpots);

    lastLightCullStats.pointTotal = static_cast<uint32_t>(allPoints.size());
    lastLightCullStats.spotTotal = static_cast<uint32_t>(allSpots.size());

    // Step 2: build the camera frustum (using the current viewport aspect ratio)
    const float aspect = static_cast<float>(viewportExtent.width) /
                         static_cast<float>(viewportExtent.height);
    const glm::mat4 vp = camera.getProjectionMatrix(aspect) * camera.getViewMatrix();
    const FrustumCulling::Frustum frustum = FrustumCulling::extractFromViewProjection(vp);

    // Step 3: frustum cull (point / spot use bounding spheres; dir is not culled)
    lights.clear();
    spotLights.clear();
    if (lightFrustumCullEnabled)
    {
        for (const auto &pl : allPoints)
            if (FrustumCulling::testSphere(frustum, pl.position, pl.radius))
                lights.push_back(pl);
        for (const auto &sl : allSpots)
            if (FrustumCulling::testSphere(frustum, sl.position, sl.radius))
                spotLights.push_back(sl);
    }
    else
    {
        lights = std::move(allPoints);
        spotLights = std::move(allSpots);
    }
    dirLights = std::move(allDirs);

    // Step 4: capacity truncation + warning
    if (lights.size() > MAX_POINT_LIGHTS)
    {
        std::cerr << "[Lighting] PointLight count " << lights.size()
                  << " exceeds MAX_POINT_LIGHTS=" << MAX_POINT_LIGHTS << ", truncating.\n";
        lights.resize(MAX_POINT_LIGHTS);
    }
    if (dirLights.size() > MAX_DIR_LIGHTS)
    {
        std::cerr << "[Lighting] DirLight count " << dirLights.size()
                  << " exceeds MAX_DIR_LIGHTS=" << MAX_DIR_LIGHTS << ", truncating.\n";
        dirLights.resize(MAX_DIR_LIGHTS);
    }
    if (spotLights.size() > MAX_SPOT_LIGHTS)
    {
        std::cerr << "[Lighting] SpotLight count " << spotLights.size()
                  << " exceeds MAX_SPOT_LIGHTS=" << MAX_SPOT_LIGHTS << ", truncating.\n";
        spotLights.resize(MAX_SPOT_LIGHTS);
    }

    lastLightCullStats.pointVisible = static_cast<uint32_t>(lights.size());
    lastLightCullStats.spotVisible = static_cast<uint32_t>(spotLights.size());

    // Step 5: memcpy into the current frame's mapped SSBO (empty arrays need
    // no special handling: 0 bytes are written)
    if (!lights.empty())
        std::memcpy(lightSSBOsMapped[frameIndex], lights.data(),
                    sizeof(PointLight) * lights.size());
    if (!dirLights.empty())
        std::memcpy(dirLightSSBOsMapped[frameIndex], dirLights.data(),
                    sizeof(DirectionalLight) * dirLights.size());
    if (!spotLights.empty())
        std::memcpy(spotLightSSBOsMapped[frameIndex], spotLights.data(),
                    sizeof(SpotLight) * spotLights.size());
}

// ============================================================================
// doImportModel() - delegated to SceneSetup
// ============================================================================
void Renderer::doImportModel(const std::string &filepath)
{
    SceneSetup::importModel(filepath, vulkanContext, allocator, scene, importedMeshes);
    lightsDirty = true;
}

// ============================================================================
// allocateTextureDescriptorSet()
// ============================================================================
// Allocate a fresh COMBINED_IMAGE_SAMPLER descriptor set bound to (sampler,
// view) of the given texture, and remember the handle for later cleanup.
//
// The pool was created in createGeometryPassResources() with
//   maxSets = 64, descriptorCount = 64, FREE_DESCRIPTOR_SET_BIT
// so up to 64 - 1 (default white) = 63 textures can coexist. For the current
// benchmark presets that is more than enough; a future "many-textures" pass
// will require resizing or a per-set pool strategy.
VkDescriptorSet Renderer::allocateTextureDescriptorSet(const TextureResource &tex)
{
    VkDevice device = vulkanContext.getDevice();

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = texDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &texDescriptorSetLayout;

    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &allocInfo, &ds) != VK_SUCCESS || ds == VK_NULL_HANDLE)
    {
        std::cerr << "[Renderer] allocateTextureDescriptorSet: pool exhausted\n";
        return VK_NULL_HANDLE;
    }

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = tex.sampler;
    imgInfo.imageView = tex.image.imageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = ds;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    allocatedTextureDescriptorSets.push_back(ds);
    return ds;
}

// ============================================================================
// freeAllAllocatedTextureDescriptorSets()
// ============================================================================
// Returns every descriptor set previously handed out by
// allocateTextureDescriptorSet() back to the pool. Caller must guarantee no
// frame in flight is still referencing them (we ensure this in
// startBenchmarkByName() by calling vkDeviceWaitIdle before re-loading).
void Renderer::freeAllAllocatedTextureDescriptorSets()
{
    if (allocatedTextureDescriptorSets.empty())
        return;
    VkDevice device = vulkanContext.getDevice();
    vkFreeDescriptorSets(device, texDescriptorPool,
                         static_cast<uint32_t>(allocatedTextureDescriptorSets.size()),
                         allocatedTextureDescriptorSets.data());
    allocatedTextureDescriptorSets.clear();
}

// ============================================================================
// loadTextureStressScene()
// ============================================================================
// Programmatic 4-color preset for verifying (mesh, tex) batching.
// Implemented here (not in SceneSetup) because it needs textureManager +
// allocateTextureDescriptorSet, which SceneSetup does not own.
//
// Layout: 100 cubes in a 10x10 grid, 4 textures distributed round-robin so
// each color group has 25 entities sharing the same texDS. With instancing
// instancing this should produce dc.geometry == 4 (one batch per (cube, texK)).
void Renderer::loadTextureStressScene()
{
    // Make sure no in-flight frame still references previously bound texture
    // descriptor sets (we will free + re-allocate them below).
    vkDeviceWaitIdle(vulkanContext.getDevice());
    freeAllAllocatedTextureDescriptorSets();

    scene.destroyAll();

    // 1) Build 4 solid-color 16x16 RGBA8 textures (red / green / blue / yellow).
    const int W = 16, H = 16;
    const uint8_t palette[4][4] = {
        {220, 40, 40, 255},  // red
        {40, 200, 60, 255},  // green
        {40, 90, 230, 255},  // blue
        {230, 200, 40, 255}, // yellow
    };
    std::array<std::shared_ptr<TextureResource>, 4> texs;
    std::array<VkDescriptorSet, 4> texSets{};
    for (int t = 0; t < 4; ++t)
    {
        std::vector<unsigned char> pixels(W * H * 4);
        for (int i = 0; i < W * H; ++i)
        {
            pixels[i * 4 + 0] = palette[t][0];
            pixels[i * 4 + 1] = palette[t][1];
            pixels[i * 4 + 2] = palette[t][2];
            pixels[i * 4 + 3] = palette[t][3];
        }
        texs[t] = textureManager.createTextureFromPixels(
            pixels.data(), W, H,
            std::string("benchmark_color_") + std::to_string(t));
        texSets[t] = allocateTextureDescriptorSet(*texs[t]);
    }

    // 2) Spawn 100 cubes in a 10x10 grid; assign textures round-robin so
    //    each color group has 25 entities sharing the same texDS.
    auto &reg = scene.registry;
    const int gridN = 10;
    const float spacing = 2.0f;
    for (int y = 0; y < gridN; ++y)
        for (int x = 0; x < gridN; ++x)
        {
            int idx = y * gridN + x;
            int t = idx % 4;
            auto e = scene.createEntity("TexCube_" + std::to_string(idx));
            reg.emplace<MeshComponent>(e, MeshComponent{cubeMeshRef, "builtin:cube"});
            auto &mat = reg.emplace<MaterialComponent>(e);
            mat.albedo = glm::vec3(1.0f); // pure white so texture color shows through
            mat.metallic = 0.0f;
            mat.roughness = 0.6f;
            mat.albedoTexture = texs[t];
            mat.textureDescriptorSet = texSets[t];

            auto &tc = reg.get<TransformComponent>(e);
            tc.position = glm::vec3((x - 4.5f) * spacing, 1.0f, (y - 4.5f) * spacing);
        }

    // 3) Reuse the default 4-light setup (same as Stress100 for fair compare).
    auto defaults = createDefaultLights();
    const char *names[] = {"BenchLight0", "BenchLight1", "BenchLight2", "BenchLight3"};
    for (size_t i = 0; i < defaults.size(); ++i)
    {
        auto e = scene.createEntity(names[i]);
        auto &tc = reg.get<TransformComponent>(e);
        tc.position = defaults[i].position;
        reg.emplace<PointLightComponent>(
            e, PointLightComponent{defaults[i].color, defaults[i].intensity, defaults[i].radius});
    }
    std::printf("[Bench] Loaded preset 'TextureStress' (100 entities, 4 textures, 4 lights)\n");
}

// ============================================================================
// startBenchmarkByName() - CLI / programmatic batch entry point
// ============================================================================
bool Renderer::startBenchmarkByName(const std::string &presetName)
{
    if (presetName == "TextureStress")
    {
        loadTextureStressScene();
        benchmarkRunner.start(presetName);
        return true;
    }

    // Other presets: re-use scene loader + free any previously allocated
    // texture descriptor sets so reload-after-TextureStress is clean.
    vkDeviceWaitIdle(vulkanContext.getDevice());
    freeAllAllocatedTextureDescriptorSets();

    SceneSetup::BenchmarkPreset preset;
    if (presetName == "Empty")
        preset = SceneSetup::BenchmarkPreset::Empty;
    else if (presetName == "Stress100")
        preset = SceneSetup::BenchmarkPreset::Stress100;
    else if (presetName == "Stress1000")
        preset = SceneSetup::BenchmarkPreset::Stress1000;
    else if (presetName == "LightStress")
        preset = SceneSetup::BenchmarkPreset::LightStress;
    else
        return false;

    SceneSetup::loadBenchmarkScene(preset, scene, cubeMeshRef, planeMeshRef);
    benchmarkRunner.start(presetName);
    return true;
}

// ============================================================================
// setShadowMapResolution()
// ----------------------------------------------------------------------------
// Rebuilds ShadowPass's depth image (no other pass resources are touched) and
// refreshes dirShadowUBO's invMapSize field. Heavy (vkDeviceWaitIdle + rebuild),
// so it should only fire when the user changes it in Render Settings.
// ============================================================================
void Renderer::setShadowMapResolution(int r)
{
    if (r < 256)
        r = 256;
    if (r > 8192)
        r = 8192;
    if (r == shadowMapResolution)
        return;

    vkDeviceWaitIdle(vulkanContext.getDevice());
    shadowMapResolution = r;
    shadowPass.resize(vulkanContext.getDevice(), allocator,
                      static_cast<uint32_t>(shadowMapResolution));
    // ShadowPass.resize() rebuilds image / view / sampler / pipeline internally.
    // The lighting descriptor was bound to the OLD view + sampler in
    // createLightingPassResources, pointing at ShadowPass members; after the
    // rebuild both handles changed, so the lighting descriptor must be refreshed.
    //
    // This used to call onViewportResize() directly, rebuilding
    // the whole GBuffer / Depth / HiZ / SSAO / Bloom / Composite / FXAA /
    // lighting pipeline / light volume / gizmo chain (hundreds of ms). Only
    // binding 8 (shadow map) of the lighting descriptor actually needs to point
    // at ShadowPass's new view + sampler; the other bindings are refreshed too
    // (cheap).
    updateLightingDescriptorSets();
}
// ============================================================================
// onViewportResize() - rebuilds all render resources when the viewport size changes
// ============================================================================
void Renderer::onViewportResize()
{
    VkDevice device = vulkanContext.getDevice();
    vkDeviceWaitIdle(device);

    // Tear down all post-process passes BEFORE
    // recreating the resources they reference. Order doesn't matter among the
    // cleanup* calls, but they must all run before editorUI / gbuffer rebuild.
    cleanupCompositeResources();
    cleanupFxaaResources();
    cleanupSsaoResources();
    cleanupBloomResources();

    editorUI.cleanupOffscreenResources();
    editorUI.createOffscreenResources();

    // Hi-Z depends on the depth image view, so clean it up
    // before rebuilding depth/gbuffer
    hiZPass.cleanup(device, allocator.getVma());

    DepthUtils::destroyDepthImage(device, allocator.getVma(), depthImage);
    gbuffer.cleanup(device, allocator.getVma());
    gbuffer.init(device, allocator.getVma(), viewportExtent);
    createDepthResources();

    // Depth rebuilt -> rebuild Hi-Z (using the new depth view)
    hiZPass.init(device, allocator.getVma(), viewportExtent,
                 depthImage.imageView, gbufferSampler);

    // Rebuild the post-process chain in dependency order (SSAO -> Bloom ->
    // Composite -> FXAA): SSAO writes ssaoImage and Bloom writes bloomMips[0],
    // both sampled by Composite. FXAA samples the pre-FXAA LDR (owned by
    // EditorUI, already recreated above).
    createSsaoResources();
    createBloomResources();
    createCompositeResources();
    createFxaaResources();

    // Clean up the lighting pass uniform buffers
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vmaUnmapMemory(allocator.getVma(), lightUniformBuffers[i].allocation);
        allocator.destroyBuffer(lightUniformBuffers[i]);
    }
    // dirShadowUBOs are rebuilt together with the lighting pass
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < dirShadowUBOs.size() && dirShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), dirShadowUBOs[i].allocation);
            allocator.destroyBuffer(dirShadowUBOs[i]);
        }
    }
    dirShadowUBOs.clear();
    dirShadowUBOsMapped.clear();
    // spotShadowUBOs rebuilt in sync
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < spotShadowUBOs.size() && spotShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), spotShadowUBOs[i].allocation);
            allocator.destroyBuffer(spotShadowUBOs[i]);
        }
    }
    spotShadowUBOs.clear();
    spotShadowUBOsMapped.clear();
    // pointShadowUBOs rebuilt in sync
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < pointShadowUBOs.size() && pointShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), pointShadowUBOs[i].allocation);
            allocator.destroyBuffer(pointShadowUBOs[i]);
        }
    }
    pointShadowUBOs.clear();
    pointShadowUBOsMapped.clear();
    // Cleanup of the per-frame host-mapped light SSBOs (they are re-created on rebuild)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < lightSSBOs.size() && lightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), lightSSBOs[i].allocation);
            allocator.destroyBuffer(lightSSBOs[i]);
        }
        if (i < dirLightSSBOs.size() && dirLightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), dirLightSSBOs[i].allocation);
            allocator.destroyBuffer(dirLightSSBOs[i]);
        }
        if (i < spotLightSSBOs.size() && spotLightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), spotLightSSBOs[i].allocation);
            allocator.destroyBuffer(spotLightSSBOs[i]);
        }
    }
    vkDestroyDescriptorPool(device, lightDescriptorPool, nullptr);
    vkDestroyPipeline(device, lightPipeline, nullptr);
    vkDestroyPipelineLayout(device, lightPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, lightDescriptorSetLayout, nullptr);

    // Light volume resources (pipelines depend on
    // lightDescriptorSetLayout, so rebuild them together)
    if (lightVolumePointPipeline)
        vkDestroyPipeline(device, lightVolumePointPipeline, nullptr);
    if (lightVolumeSpotPipeline)
        vkDestroyPipeline(device, lightVolumeSpotPipeline, nullptr);
    if (lightVolumePipelineLayout)
        vkDestroyPipelineLayout(device, lightVolumePipelineLayout, nullptr);
    if (lightVolumeVB.buffer)
        allocator.destroyBuffer(lightVolumeVB);
    if (lightVolumeIB.buffer)
        allocator.destroyBuffer(lightVolumeIB);
    lightVolumePointPipeline = VK_NULL_HANDLE;
    lightVolumeSpotPipeline = VK_NULL_HANDLE;
    lightVolumePipelineLayout = VK_NULL_HANDLE;
    lightVolumeVB = {};
    lightVolumeIB = {};

    // Clean up the geometry pass uniform buffers
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vmaUnmapMemory(allocator.getVma(), geomUniformBuffers[i].allocation);
        allocator.destroyBuffer(geomUniformBuffers[i]);
    }
    // Per-frame instance SSBO
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < instanceSSBOs.size() && instanceSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), instanceSSBOs[i].allocation);
            allocator.destroyBuffer(instanceSSBOs[i]);
        }
    }
    instanceSSBOs.clear();
    instanceSSBOsMapped.clear();
    vkDestroyDescriptorPool(device, geomDescriptorPool, nullptr);
    vkDestroyPipeline(device, geomPipeline, nullptr);
    vkDestroyPipelineLayout(device, geomPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, geomDescriptorSetLayout, nullptr);
    // Do NOT destroy texDescriptorPool / texDescriptorSetLayout /
    // defaultTexDescriptorSet here. They're owned by createTextureResources()
    // and must survive every viewport resize, otherwise sets handed out by
    // allocateTextureDescriptorSet() (used by TextureStress preset and any
    // future per-material texture binding) would silently dangle.

    createGeometryPassResources();
    doSyncLightsToGPU();
    createLightingPassResources();
    createLightVolumeResources(); // Rebuild light volume pipelines (depend on lightDescriptorSetLayout)

    // Geometry Pass resources were rebuilt, so re-bind the
    // instance SSBO to shadowPass.
    shadowPass.bindInstanceBuffers(device, instanceSSBOs,
                                   sizeof(InstanceData) * MAX_INSTANCES);

    gizmoRenderer.cleanup(device, allocator);
    gizmoRenderer.createLinePipeline(device, vulkanContext, allocator,
                                     geomDescriptorSetLayout, gbuffer.getColorFormats(), depthFormat);
}

// ============================================================================
// onViewportResize_RT() - fast path (rebuilds only size-dependent resources)
// ----------------------------------------------------------------------------
// Rebuilds only the size-dependent render-target resources and refreshes the
// descriptors referencing them. Differences vs. the heavy onViewportResize():
//   x does NOT rebuild geomDescriptorSetLayout / geomPipelineLayout / geomPipeline
//   x does NOT rebuild lightDescriptorSetLayout / lightPipelineLayout /
//     lightPipeline / lightDescriptorPool / lightDescriptorSets (only re-writes
//     them via vkUpdateDescriptorSets)
//   x does NOT rebuild lightVolume* pipeline / VB / IB
//   x does NOT rebuild geomUniformBuffers / instanceSSBOs / lightUniformBuffers /
//     lightSSBOs / dirLightSSBOs / spotLightSSBOs / dir|spot|pointShadowUBOs
//   x does NOT rebuild the gizmoRenderer pipeline
//   + only rebuilds GBuffer / Depth / HiZ / EditorUI offscreen / Composite /
//     FXAA / SSAO / Bloom, and repoints lighting set bindings 0~2 (GBuffer) and
//     7 (HiZ) at the new views via updateLightingDescriptorSets().
//
// Precondition: every in-flight frame's fence must be complete (drawFrame()
// calls this after vkWaitForFences, so no extra vkDeviceWaitIdle is needed).
// ============================================================================
void Renderer::onViewportResize_RT()
{
    VkDevice device = vulkanContext.getDevice();

    // 1) Wait all in-flight fences first. drawFrame() only waits
    //    currentFrame; other in-flight slots' command buffers may still
    //    reference the old GBuffer / Depth views, so wait once more here to
    //    guarantee the GPU is not using them when we destroy the old
    //    resources. No vkDeviceWaitIdle, to avoid blocking until vsync.
    if (!inFlightFences.empty())
    {
        vkWaitForFences(device,
                        static_cast<uint32_t>(inFlightFences.size()),
                        inFlightFences.data(),
                        VK_TRUE,
                        UINT64_MAX);
    }

    // 2) tear down post-process passes that reference the size-dependent images.
    cleanupCompositeResources();
    cleanupFxaaResources();
    cleanupSsaoResources();
    cleanupBloomResources();

    // 3) EditorUI offscreen (hdrImage / ldrPreFxaaImage / offscreenImage)
    editorUI.cleanupOffscreenResources();
    editorUI.createOffscreenResources();

    // 4) HiZ (depends on depth view) -> tear down BEFORE depth/gbuffer rebuild.
    hiZPass.cleanup(device, allocator.getVma());

    // 5) Depth + GBuffer
    DepthUtils::destroyDepthImage(device, allocator.getVma(), depthImage);
    gbuffer.cleanup(device, allocator.getVma());
    gbuffer.init(device, allocator.getVma(), viewportExtent);
    createDepthResources();

    // 6) HiZ rebuild with new depth view
    hiZPass.init(device, allocator.getVma(), viewportExtent,
                 depthImage.imageView, gbufferSampler);

    // 7) post-process chain rebuild in dependency order
    createSsaoResources();
    createBloomResources();
    createCompositeResources();
    createFxaaResources();

    // 8) re-write the lighting descriptors (pool/layout/pipeline/sets are not
    //    destroyed; bindings 0~2/7 just repoint at the new views; the other
    //    bindings are refreshed too, at negligible cost).
    updateLightingDescriptorSets();

    // 9) ShadowPass's instance SSBO binding does not depend on the viewport, so
    //    no re-bind needed. Same for geomDescriptorSets: the geometry pass
    //    renders the GBuffer as color attachments and never samples it through
    //    descriptors, so the new GBuffer views don't need re-writing there.
}

// ============================================================================
// createCompositeResources()
// ============================================================================
// Pipeline that samples editorUI.hdrImage (FP16 linear HDR) and writes
// editorUI.offscreenImage (sRGB B8G8R8A8 LDR). Push constant carries exposure
// and tonemap mode; the descriptor set is per frame-in-flight and re-written
// every viewport resize because hdrImage.imageView is recreated.
//
// Rebuilt on every onViewportResize() - fully torn down by
// cleanupCompositeResources() first.
void Renderer::createCompositeResources()
{
    VkDevice device = vulkanContext.getDevice();

    // 1) descriptor set layout: hdr (binding 0) + ao (binding 1) + bloom (binding 2)
    std::array<VkDescriptorSetLayoutBinding, 3> compBindings{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        compBindings[i].binding = i;
        compBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        compBindings[i].descriptorCount = 1;
        compBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(compBindings.size());
    layoutInfo.pBindings = compBindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &compositeDescriptorSetLayout);

    // 2) pipeline layout - push constant: float exposure + int tonemapMode + 8 bytes pad
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset = 0;
    pc.size = 16; // matches composite.frag's CompositePush
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &compositeDescriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &compositePipelineLayout);

    // 3) pipeline (full-screen triangle, sRGB output)
    std::string shaderDir = SHADER_DIR;
    PipelineBuilder builder;
    compositePipeline = builder
                            .setShaders(device, shaderDir + "/composite.vert.spv", shaderDir + "/composite.frag.spv")
                            .setNoVertexInput()
                            .setInputAssembly()
                            .setViewportDynamic()
                            .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                            .setMultisampling()
                            .setDepthStencil(false, false)
                            .setColorBlending(false)
                            .setColorAttachmentFormat(swapchain.getImageFormat()) // sRGB B8G8R8A8 (matches editorUI.offscreenImage)
                            .build(device, compositePipelineLayout);
    builder.cleanupShaderModules(device);

    // 4) HDR sampler (linear, clamp-to-edge)
    if (hdrSampler == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &si, nullptr, &hdrSampler);
    }

    // 5) descriptor pool + per-frame descriptor sets
    // 3 samplers per set (hdr + ao + bloom).
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 3 * MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &compositeDescriptorPool);

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, compositeDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = compositeDescriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    compositeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocInfo, compositeDescriptorSets.data());

    // 6) bind hdrImage view to every per-frame set (single shared image - both
    //    frames-in-flight read the same hdrImage; the lighting pass barrier
    //    ensures it is in SHADER_READ_ONLY before composite samples it).
    //    Also bind ssaoImage at binding 1.
    //    Also bind bloomMips[0] at binding 2.
    auto &hdrImg = editorUI.getHdrImage();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        std::array<VkDescriptorImageInfo, 3> infos{};
        infos[0].sampler = hdrSampler;
        infos[0].imageView = hdrImg.imageView;
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].sampler = hdrSampler; // reuse linear clamp-to-edge sampler
        infos[1].imageView = ssaoImage.imageView;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[2].sampler = hdrSampler;
        infos[2].imageView = bloomMips[0].imageView;
        infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (uint32_t j = 0; j < 3; ++j)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[j].dstSet = compositeDescriptorSets[i];
            writes[j].dstBinding = j;
            writes[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[j].descriptorCount = 1;
            writes[j].pImageInfo = &infos[j];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::cout << "[Renderer] Composite Pass resources created.\n";
}

// ============================================================================
// cleanupCompositeResources()
// ============================================================================
void Renderer::cleanupCompositeResources()
{
    VkDevice device = vulkanContext.getDevice();
    if (compositeDescriptorPool)
    {
        vkDestroyDescriptorPool(device, compositeDescriptorPool, nullptr);
        compositeDescriptorPool = VK_NULL_HANDLE;
    }
    compositeDescriptorSets.clear();
    if (compositePipeline)
    {
        vkDestroyPipeline(device, compositePipeline, nullptr);
        compositePipeline = VK_NULL_HANDLE;
    }
    if (compositePipelineLayout)
    {
        vkDestroyPipelineLayout(device, compositePipelineLayout, nullptr);
        compositePipelineLayout = VK_NULL_HANDLE;
    }
    if (compositeDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, compositeDescriptorSetLayout, nullptr);
        compositeDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (hdrSampler)
    {
        vkDestroySampler(device, hdrSampler, nullptr);
        hdrSampler = VK_NULL_HANDLE;
    }
}

// ============================================================================
// createFxaaResources()
// ============================================================================
// Pipeline that samples editorUI.ldrPreFxaaImage (sRGB LDR, output of the
// composite pass) and writes editorUI.offscreenImage (final, ImGui-bound).
// Push constant carries (invResolution, enableFxaa); when disabled the shader
// passes the source through unchanged so the pipeline structure is stable.
void Renderer::createFxaaResources()
{
    VkDevice device = vulkanContext.getDevice();

    VkDescriptorSetLayoutBinding sampBinding{};
    sampBinding.binding = 0;
    sampBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampBinding.descriptorCount = 1;
    sampBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &sampBinding;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &fxaaDescriptorSetLayout);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset = 0;
    pc.size = 32; // vec2 invRes(8) + int enable(4) + int pad(4) + float threshold(4) + float subpixel(4) + int debugEdges(4) + int pad(4)
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &fxaaDescriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &fxaaPipelineLayout);

    std::string shaderDir = SHADER_DIR;
    PipelineBuilder builder;
    fxaaPipeline = builder
                       .setShaders(device, shaderDir + "/fxaa.vert.spv", shaderDir + "/fxaa.frag.spv")
                       .setNoVertexInput()
                       .setInputAssembly()
                       .setViewportDynamic()
                       .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                       .setMultisampling()
                       .setDepthStencil(false, false)
                       .setColorBlending(false)
                       .setColorAttachmentFormat(swapchain.getImageFormat()) // sRGB B8G8R8A8 (matches editorUI.offscreenImage)
                       .build(device, fxaaPipelineLayout);
    builder.cleanupShaderModules(device);

    if (ldrSampler == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &si, nullptr, &ldrSampler);
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &fxaaDescriptorPool);

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, fxaaDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = fxaaDescriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    fxaaDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocInfo, fxaaDescriptorSets.data());

    auto &preImg = editorUI.getLdrPreFxaaImage();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorImageInfo ii{};
        ii.sampler = ldrSampler;
        ii.imageView = preImg.imageView;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = fxaaDescriptorSets[i];
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    std::cout << "[Renderer] FXAA Pass resources created.\n";
}

// ============================================================================
// cleanupFxaaResources()
// ============================================================================
void Renderer::cleanupFxaaResources()
{
    VkDevice device = vulkanContext.getDevice();
    if (fxaaDescriptorPool)
    {
        vkDestroyDescriptorPool(device, fxaaDescriptorPool, nullptr);
        fxaaDescriptorPool = VK_NULL_HANDLE;
    }
    fxaaDescriptorSets.clear();
    if (fxaaPipeline)
    {
        vkDestroyPipeline(device, fxaaPipeline, nullptr);
        fxaaPipeline = VK_NULL_HANDLE;
    }
    if (fxaaPipelineLayout)
    {
        vkDestroyPipelineLayout(device, fxaaPipelineLayout, nullptr);
        fxaaPipelineLayout = VK_NULL_HANDLE;
    }
    if (fxaaDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, fxaaDescriptorSetLayout, nullptr);
        fxaaDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (ldrSampler)
    {
        vkDestroySampler(device, ldrSampler, nullptr);
        ldrSampler = VK_NULL_HANDLE;
    }
}

// ============================================================================
// createSsaoResources() - improved (view-space SSAO + bilateral
// blur + half-resolution upsample)
// ============================================================================
// Creates a 3-pass SSAO pipeline:
//   (1) raw SSAO          : half-res GBuffer sample -> ssaoRawImage     (R8)
//   (2) bilateral blur X  : half-res ssaoRaw       -> ssaoBlurXImage   (R8)
//   (3) bilateral blur Y  : half-res ssaoBlurX     -> ssaoImage         (R8, full-res, upsampled)
//
// The composite pass samples `ssaoImage` (full-res). SSAO always runs;
// `ssaoIntensity == 0` makes the raw output a constant 1.0 -> blur is a no-op
// -> composite multiplies by 1.0.
//
// Half-res reduces SSAO GPU cost ~4x; bilateral blur recovers smoothness with
// edge preservation. The SSAO pipeline reads a UBO carrying view+proj matrices
// (needed to project view-space hemisphere samples back to screen UV); the
// blur pipeline reads a UBO with `view` (depth-aware weight) + `invInputSize`
// (1.0 / half-res pixel size for kernel stride).
void Renderer::createSsaoResources()
{
    VkDevice device = vulkanContext.getDevice();
    VmaAllocator vma = allocator.getVma();

    const uint32_t halfW = std::max(1u, viewportExtent.width / 2);
    const uint32_t halfH = std::max(1u, viewportExtent.height / 2);

    auto createR8 = [&](AllocatedImage &img, uint32_t w, uint32_t h)
    {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8_UNORM;
        ii.extent.width = w;
        ii.extent.height = h;
        ii.extent.depth = 1;
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(vma, &ii, &aci, &img.image, &img.allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to create SSAO R8 image!");
        img.format = ii.format;
        img.extent = {w, h};
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = img.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = img.format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &vi, nullptr, &img.imageView);
    };

    // 1) AO images
    createR8(ssaoRawImage, halfW, halfH);                             // raw, half-res
    createR8(ssaoBlurXImage, halfW, halfH);                           // after H blur, half-res
    createR8(ssaoImage, viewportExtent.width, viewportExtent.height); // final, full-res

    // ------------------------------------------------------------------------
    // 2) Raw SSAO descriptor set layout: gPos + gNorm (sampled) + UBO (view/proj/screenSize)
    // ------------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings = b.data();
        vkCreateDescriptorSetLayout(device, &li, nullptr, &ssaoDescriptorSetLayout);
    }

    // 3) Raw SSAO pipeline layout - push constant (radius, intensity, bias, _pad)
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = 16;
        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &ssaoDescriptorSetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(device, &plInfo, nullptr, &ssaoPipelineLayout);
    }

    // 4) Raw SSAO pipeline (full-screen triangle, R8_UNORM target, dynamic viewport)
    {
        std::string shaderDir = SHADER_DIR;
        PipelineBuilder builder;
        ssaoPipeline = builder
                           .setShaders(device, shaderDir + "/ssao.vert.spv", shaderDir + "/ssao.frag.spv")
                           .setNoVertexInput()
                           .setInputAssembly()
                           .setViewportDynamic()
                           .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                           .setMultisampling()
                           .setDepthStencil(false, false)
                           .setColorBlending(false)
                           .setColorAttachmentFormat(VK_FORMAT_R8_UNORM)
                           .build(device, ssaoPipelineLayout);
        builder.cleanupShaderModules(device);
    }

    // 5) Raw SSAO UBO (view + proj + screenSize + pad) per-frame
    {
        struct SsaoUBO
        {
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec2 screenSize;
            glm::vec2 _pad;
        };
        ssaoUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        ssaoUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ssaoUniformBuffers[i] = allocator.createBuffer(
                sizeof(SsaoUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            vmaMapMemory(vma, ssaoUniformBuffers[i].allocation, &ssaoUniformBuffersMapped[i]);
        }
    }

    // 6) Raw SSAO descriptor pool + per-frame sets
    {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT;
        ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[1].descriptorCount = 1 * MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = static_cast<uint32_t>(ps.size());
        pi.pPoolSizes = ps.data();
        pi.maxSets = MAX_FRAMES_IN_FLIGHT;
        vkCreateDescriptorPool(device, &pi, nullptr, &ssaoDescriptorPool);

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, ssaoDescriptorSetLayout);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = ssaoDescriptorPool;
        ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        ai.pSetLayouts = layouts.data();
        ssaoDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        vkAllocateDescriptorSets(device, &ai, ssaoDescriptorSets.data());

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorImageInfo posInfo{};
            posInfo.sampler = gbufferSampler;
            posInfo.imageView = gbuffer.getImageView(GBuffer::POSITION);
            posInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo normInfo{};
            normInfo.sampler = gbufferSampler;
            normInfo.imageView = gbuffer.getImageView(GBuffer::NORMAL);
            normInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = ssaoUniformBuffers[i].buffer;
            uboInfo.offset = 0;
            uboInfo.range = VK_WHOLE_SIZE;

            std::array<VkWriteDescriptorSet, 3> w{};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = ssaoDescriptorSets[i];
            w[0].dstBinding = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].descriptorCount = 1;
            w[0].pImageInfo = &posInfo;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = ssaoDescriptorSets[i];
            w[1].dstBinding = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].descriptorCount = 1;
            w[1].pImageInfo = &normInfo;
            w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[2].dstSet = ssaoDescriptorSets[i];
            w[2].dstBinding = 2;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[2].descriptorCount = 1;
            w[2].pBufferInfo = &uboInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        }
    }

    // ------------------------------------------------------------------------
    // 7) Blur descriptor set layout: aoIn + gPos (sampled) + UBO (view, invInputSize)
    // ------------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings = b.data();
        vkCreateDescriptorSetLayout(device, &li, nullptr, &ssaoBlurDescriptorSetLayout);
    }

    // 8) Blur pipeline layout - push constant (dir, _, _, _)
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = 16;
        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &ssaoBlurDescriptorSetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(device, &plInfo, nullptr, &ssaoBlurPipelineLayout);
    }

    // 9) Blur pipeline (reused for X and Y; viewport differs)
    {
        std::string shaderDir = SHADER_DIR;
        PipelineBuilder builder;
        ssaoBlurPipeline = builder
                               .setShaders(device, shaderDir + "/ssao.vert.spv", shaderDir + "/ssao_blur.frag.spv")
                               .setNoVertexInput()
                               .setInputAssembly()
                               .setViewportDynamic()
                               .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                               .setMultisampling()
                               .setDepthStencil(false, false)
                               .setColorBlending(false)
                               .setColorAttachmentFormat(VK_FORMAT_R8_UNORM)
                               .build(device, ssaoBlurPipelineLayout);
        builder.cleanupShaderModules(device);
    }

    // 10) Blur UBO (view + invInputSize + pad) per-frame
    {
        struct BlurUBO
        {
            glm::mat4 view;
            glm::vec2 invInputSize;
            glm::vec2 _pad;
        };
        ssaoBlurUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        ssaoBlurUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ssaoBlurUniformBuffers[i] = allocator.createBuffer(
                sizeof(BlurUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            vmaMapMemory(vma, ssaoBlurUniformBuffers[i].allocation, &ssaoBlurUniformBuffersMapped[i]);
        }
    }

    // 11) Blur descriptor pool + 2 per-frame set arrays (X: ssaoRaw -> ssaoBlurX, Y: ssaoBlurX -> ssaoImage)
    {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 4 * MAX_FRAMES_IN_FLIGHT; // 2 sets/frame x 2 sampled bindings
        ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[1].descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT; // 2 sets/frame x 1 UBO
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = static_cast<uint32_t>(ps.size());
        pi.pPoolSizes = ps.data();
        pi.maxSets = 2 * MAX_FRAMES_IN_FLIGHT;
        vkCreateDescriptorPool(device, &pi, nullptr, &ssaoBlurDescriptorPool);

        auto allocSets = [&](std::vector<VkDescriptorSet> &out)
        {
            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, ssaoBlurDescriptorSetLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = ssaoBlurDescriptorPool;
            ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
            ai.pSetLayouts = layouts.data();
            out.resize(MAX_FRAMES_IN_FLIGHT);
            vkAllocateDescriptorSets(device, &ai, out.data());
        };
        allocSets(ssaoBlurXDescriptorSets);
        allocSets(ssaoBlurYDescriptorSets);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            // Common: gPosition (set binding=1) + UBO (binding=2)
            VkDescriptorImageInfo posInfo{};
            posInfo.sampler = gbufferSampler;
            posInfo.imageView = gbuffer.getImageView(GBuffer::POSITION);
            posInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = ssaoBlurUniformBuffers[i].buffer;
            uboInfo.offset = 0;
            uboInfo.range = VK_WHOLE_SIZE;

            // X set: aoIn = ssaoRawImage
            VkDescriptorImageInfo aoXInfo{};
            aoXInfo.sampler = gbufferSampler;
            aoXInfo.imageView = ssaoRawImage.imageView;
            aoXInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // Y set: aoIn = ssaoBlurXImage
            VkDescriptorImageInfo aoYInfo{};
            aoYInfo.sampler = gbufferSampler;
            aoYInfo.imageView = ssaoBlurXImage.imageView;
            aoYInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::array<VkWriteDescriptorSet, 6> w{};
            // X: bind 0 (aoRaw), 1 (gPos), 2 (ubo)
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = ssaoBlurXDescriptorSets[i];
            w[0].dstBinding = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].descriptorCount = 1;
            w[0].pImageInfo = &aoXInfo;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = ssaoBlurXDescriptorSets[i];
            w[1].dstBinding = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].descriptorCount = 1;
            w[1].pImageInfo = &posInfo;
            w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[2].dstSet = ssaoBlurXDescriptorSets[i];
            w[2].dstBinding = 2;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[2].descriptorCount = 1;
            w[2].pBufferInfo = &uboInfo;
            // Y: bind 0 (aoBlurX), 1 (gPos), 2 (ubo)
            w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[3].dstSet = ssaoBlurYDescriptorSets[i];
            w[3].dstBinding = 0;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[3].descriptorCount = 1;
            w[3].pImageInfo = &aoYInfo;
            w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[4].dstSet = ssaoBlurYDescriptorSets[i];
            w[4].dstBinding = 1;
            w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[4].descriptorCount = 1;
            w[4].pImageInfo = &posInfo;
            w[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[5].dstSet = ssaoBlurYDescriptorSets[i];
            w[5].dstBinding = 2;
            w[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[5].descriptorCount = 1;
            w[5].pBufferInfo = &uboInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        }
    }

    std::cout << "[Renderer] SSAO Pass resources created (half-res view-space + bilateral blur).\n";
}

// ============================================================================
// cleanupSsaoResources()
// ============================================================================
void Renderer::cleanupSsaoResources()
{
    VkDevice device = vulkanContext.getDevice();
    VmaAllocator vma = allocator.getVma();

    // Blur UBOs
    for (uint32_t i = 0; i < ssaoBlurUniformBuffers.size(); ++i)
    {
        if (ssaoBlurUniformBuffers[i].buffer)
        {
            vmaUnmapMemory(vma, ssaoBlurUniformBuffers[i].allocation);
            allocator.destroyBuffer(ssaoBlurUniformBuffers[i]);
        }
    }
    ssaoBlurUniformBuffers.clear();
    ssaoBlurUniformBuffersMapped.clear();

    // Raw SSAO UBOs
    for (uint32_t i = 0; i < ssaoUniformBuffers.size(); ++i)
    {
        if (ssaoUniformBuffers[i].buffer)
        {
            vmaUnmapMemory(vma, ssaoUniformBuffers[i].allocation);
            allocator.destroyBuffer(ssaoUniformBuffers[i]);
        }
    }
    ssaoUniformBuffers.clear();
    ssaoUniformBuffersMapped.clear();

    // Blur descriptor pool (frees both X and Y sets together)
    if (ssaoBlurDescriptorPool)
    {
        vkDestroyDescriptorPool(device, ssaoBlurDescriptorPool, nullptr);
        ssaoBlurDescriptorPool = VK_NULL_HANDLE;
    }
    ssaoBlurXDescriptorSets.clear();
    ssaoBlurYDescriptorSets.clear();

    if (ssaoBlurPipeline)
    {
        vkDestroyPipeline(device, ssaoBlurPipeline, nullptr);
        ssaoBlurPipeline = VK_NULL_HANDLE;
    }
    if (ssaoBlurPipelineLayout)
    {
        vkDestroyPipelineLayout(device, ssaoBlurPipelineLayout, nullptr);
        ssaoBlurPipelineLayout = VK_NULL_HANDLE;
    }
    if (ssaoBlurDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, ssaoBlurDescriptorSetLayout, nullptr);
        ssaoBlurDescriptorSetLayout = VK_NULL_HANDLE;
    }

    // Raw SSAO descriptor pool
    if (ssaoDescriptorPool)
    {
        vkDestroyDescriptorPool(device, ssaoDescriptorPool, nullptr);
        ssaoDescriptorPool = VK_NULL_HANDLE;
    }
    ssaoDescriptorSets.clear();
    if (ssaoPipeline)
    {
        vkDestroyPipeline(device, ssaoPipeline, nullptr);
        ssaoPipeline = VK_NULL_HANDLE;
    }
    if (ssaoPipelineLayout)
    {
        vkDestroyPipelineLayout(device, ssaoPipelineLayout, nullptr);
        ssaoPipelineLayout = VK_NULL_HANDLE;
    }
    if (ssaoDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, ssaoDescriptorSetLayout, nullptr);
        ssaoDescriptorSetLayout = VK_NULL_HANDLE;
    }

    auto destroyImg = [&](AllocatedImage &img)
    {
        if (img.imageView)
        {
            vkDestroyImageView(device, img.imageView, nullptr);
            img.imageView = VK_NULL_HANDLE;
        }
        if (img.image)
        {
            vmaDestroyImage(vma, img.image, img.allocation);
            img.image = VK_NULL_HANDLE;
            img.allocation = VK_NULL_HANDLE;
        }
    };
    destroyImg(ssaoImage);
    destroyImg(ssaoBlurXImage);
    destroyImg(ssaoRawImage);
}

// ============================================================================
// createBloomResources()
// ============================================================================
// Builds the 7-mip bloom pyramid (R16G16B16A16_SFLOAT) and the 3 pipelines:
//   * threshold  : samples editorUI.hdrImage  -> bloomMips[0]
//   * downsample : samples bloomMips[i]       -> bloomMips[i+1]    for i = 0..5
//   * upsample   : samples bloomMips[i+1]     -> bloomMips[i] (ADD) for i = 5..0
//
// After the upsample chain, bloomMips[0] holds the final blurred bright pass.
// composite.frag samples it and adds `bloom * intensity` onto the lit HDR.
//
// All three pipelines share a 1-sampler descriptor SET LAYOUT and a 16-byte
// push constant; only blend state and shaders differ.
//
// Lifetime: rebuilt on every onViewportResize() because every mip's image
// extent depends on viewportExtent.
void Renderer::createBloomResources()
{
    VkDevice device = vulkanContext.getDevice();
    VmaAllocator vma = allocator.getVma();

    // ------------------------------------------------------------------------
    // 1) Bloom pyramid - 7 mip images, each its own VkImage + view.
    //    mip 0 = full-res, mip i = max(1, viewportExtent >> i).
    // ------------------------------------------------------------------------
    auto createBloomMip = [&](AllocatedImage &img, uint32_t w, uint32_t h)
    {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ii.extent.width = w;
        ii.extent.height = h;
        ii.extent.depth = 1;
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(vma, &ii, &aci, &img.image, &img.allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to create bloom mip image!");
        img.format = ii.format;
        img.extent = {w, h};

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = img.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = img.format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &vi, nullptr, &img.imageView);
    };
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i)
    {
        uint32_t w = std::max(1u, viewportExtent.width >> i);
        uint32_t h = std::max(1u, viewportExtent.height >> i);
        createBloomMip(bloomMips[i], w, h);
    }

    // ------------------------------------------------------------------------
    // 2) Sampler shared by all bloom passes (linear, clamp-to-edge so kernel
    //    samples near the edge don't wrap around).
    // ------------------------------------------------------------------------
    if (bloomSampler == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &si, nullptr, &bloomSampler);
    }

    // ------------------------------------------------------------------------
    // 3) Threshold pipeline (single sampled image at binding 0).
    // ------------------------------------------------------------------------
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(device, &li, nullptr, &bloomThresholdDSLayout);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = 16; // threshold + softKnee + 2 pad floats
        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &bloomThresholdDSLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(device, &plInfo, nullptr, &bloomThresholdPLayout);

        std::string shaderDir = SHADER_DIR;
        PipelineBuilder builder;
        bloomThresholdPipeline = builder
                                     .setShaders(device,
                                                 shaderDir + "/composite.vert.spv", // reuse fullscreen triangle
                                                 shaderDir + "/bloom_threshold.frag.spv")
                                     .setNoVertexInput()
                                     .setInputAssembly()
                                     .setViewportDynamic()
                                     .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                    VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                     .setMultisampling()
                                     .setDepthStencil(false, false)
                                     .setColorBlending(false)
                                     .setColorAttachmentFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
                                     .build(device, bloomThresholdPLayout);
        builder.cleanupShaderModules(device);
    }

    // ------------------------------------------------------------------------
    // 4) Downsample + Upsample share descriptor set layout AND pipeline layout.
    //    Push constant is 16 bytes (vec2 srcTexel + 1 float scatter + 1 pad).
    // ------------------------------------------------------------------------
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(device, &li, nullptr, &bloomChainDSLayout);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = 16;
        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &bloomChainDSLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(device, &plInfo, nullptr, &bloomChainPLayout);

        std::string shaderDir = SHADER_DIR;
        // Downsample (no blend - overwrites destination)
        {
            PipelineBuilder builder;
            bloomDownsamplePipeline = builder
                                          .setShaders(device,
                                                      shaderDir + "/composite.vert.spv",
                                                      shaderDir + "/bloom_downsample.frag.spv")
                                          .setNoVertexInput()
                                          .setInputAssembly()
                                          .setViewportDynamic()
                                          .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                         VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                          .setMultisampling()
                                          .setDepthStencil(false, false)
                                          .setColorBlending(false)
                                          .setColorAttachmentFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
                                          .build(device, bloomChainPLayout);
            builder.cleanupShaderModules(device);
        }
        // Upsample (additive blend - accumulates onto next-finer mip)
        {
            PipelineBuilder builder;
            bloomUpsamplePipeline = builder
                                        .setShaders(device,
                                                    shaderDir + "/composite.vert.spv",
                                                    shaderDir + "/bloom_upsample.frag.spv")
                                        .setNoVertexInput()
                                        .setInputAssembly()
                                        .setViewportDynamic()
                                        .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                       VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                        .setMultisampling()
                                        .setDepthStencil(false, false)
                                        .setAdditiveBlending() // src*1 + dst*1
                                        .setColorAttachmentFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
                                        .build(device, bloomChainPLayout);
            builder.cleanupShaderModules(device);
        }
    }

    // ------------------------------------------------------------------------
    // 5) Descriptor pool + per-frame sets:
    //   threshold[i]      : samples editorUI.hdrImage
    //   downsample[m][i]  : samples bloomMips[m]   (m = 0..5)
    //   upsample[m][i]    : samples bloomMips[m+1] (m = 0..5)
    // ------------------------------------------------------------------------
    const uint32_t setsPerFrame = 1 + (BLOOM_MIP_COUNT - 1) + (BLOOM_MIP_COUNT - 1); // 1 + 6 + 6
    {
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = setsPerFrame * MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        pi.maxSets = setsPerFrame * MAX_FRAMES_IN_FLIGHT;
        vkCreateDescriptorPool(device, &pi, nullptr, &bloomDescriptorPool);
    }

    auto allocOneSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = bloomDescriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet ds;
        vkAllocateDescriptorSets(device, &ai, &ds);
        return ds;
    };
    auto writeImage = [&](VkDescriptorSet ds, VkImageView view)
    {
        VkDescriptorImageInfo ii{};
        ii.sampler = bloomSampler;
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds;
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    };

    bloomThresholdSets.resize(MAX_FRAMES_IN_FLIGHT);
    bloomDownsampleSets.assign(BLOOM_MIP_COUNT - 1, std::vector<VkDescriptorSet>(MAX_FRAMES_IN_FLIGHT));
    bloomUpsampleSets.assign(BLOOM_MIP_COUNT - 1, std::vector<VkDescriptorSet>(MAX_FRAMES_IN_FLIGHT));

    auto &hdrImg = editorUI.getHdrImage();
    for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
    {
        bloomThresholdSets[f] = allocOneSet(bloomThresholdDSLayout);
        writeImage(bloomThresholdSets[f], hdrImg.imageView);

        for (uint32_t m = 0; m < BLOOM_MIP_COUNT - 1; ++m)
        {
            bloomDownsampleSets[m][f] = allocOneSet(bloomChainDSLayout);
            writeImage(bloomDownsampleSets[m][f], bloomMips[m].imageView);

            bloomUpsampleSets[m][f] = allocOneSet(bloomChainDSLayout);
            writeImage(bloomUpsampleSets[m][f], bloomMips[m + 1].imageView);
        }
    }

    std::cout << "[Renderer] Bloom Pass resources created (7-mip pyramid, threshold + downsample + additive upsample).\n";
}

// ============================================================================
// cleanupBloomResources()
// ============================================================================
void Renderer::cleanupBloomResources()
{
    VkDevice device = vulkanContext.getDevice();
    VmaAllocator vma = allocator.getVma();

    if (bloomDescriptorPool)
    {
        vkDestroyDescriptorPool(device, bloomDescriptorPool, nullptr);
        bloomDescriptorPool = VK_NULL_HANDLE;
    }
    bloomThresholdSets.clear();
    bloomDownsampleSets.clear();
    bloomUpsampleSets.clear();

    if (bloomThresholdPipeline)
    {
        vkDestroyPipeline(device, bloomThresholdPipeline, nullptr);
        bloomThresholdPipeline = VK_NULL_HANDLE;
    }
    if (bloomThresholdPLayout)
    {
        vkDestroyPipelineLayout(device, bloomThresholdPLayout, nullptr);
        bloomThresholdPLayout = VK_NULL_HANDLE;
    }
    if (bloomThresholdDSLayout)
    {
        vkDestroyDescriptorSetLayout(device, bloomThresholdDSLayout, nullptr);
        bloomThresholdDSLayout = VK_NULL_HANDLE;
    }
    if (bloomDownsamplePipeline)
    {
        vkDestroyPipeline(device, bloomDownsamplePipeline, nullptr);
        bloomDownsamplePipeline = VK_NULL_HANDLE;
    }
    if (bloomUpsamplePipeline)
    {
        vkDestroyPipeline(device, bloomUpsamplePipeline, nullptr);
        bloomUpsamplePipeline = VK_NULL_HANDLE;
    }
    if (bloomChainPLayout)
    {
        vkDestroyPipelineLayout(device, bloomChainPLayout, nullptr);
        bloomChainPLayout = VK_NULL_HANDLE;
    }
    if (bloomChainDSLayout)
    {
        vkDestroyDescriptorSetLayout(device, bloomChainDSLayout, nullptr);
        bloomChainDSLayout = VK_NULL_HANDLE;
    }
    if (bloomSampler)
    {
        vkDestroySampler(device, bloomSampler, nullptr);
        bloomSampler = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i)
    {
        if (bloomMips[i].imageView)
        {
            vkDestroyImageView(device, bloomMips[i].imageView, nullptr);
            bloomMips[i].imageView = VK_NULL_HANDLE;
        }
        if (bloomMips[i].image)
        {
            vmaDestroyImage(vma, bloomMips[i].image, bloomMips[i].allocation);
            bloomMips[i].image = VK_NULL_HANDLE;
            bloomMips[i].allocation = VK_NULL_HANDLE;
        }
    }
}

// ============================================================================
// createSyncObjects()
// ============================================================================
void Renderer::createSyncObjects()
{
    VkDevice device = vulkanContext.getDevice();
    uint32_t imageCount = swapchain.getImageCount();

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(imageCount);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i]);
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);
    }
    for (uint32_t i = 0; i < imageCount; i++)
    {
        vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]);
    }
}

// ============================================================================
// updateUniformBuffer()
// ============================================================================
void Renderer::updateUniformBuffer(uint32_t frameIndex)
{
    float aspect = static_cast<float>(viewportExtent.width) / static_cast<float>(viewportExtent.height);

    UniformBufferObject ubo{};
    ubo.view = camera.getViewMatrix();
    ubo.proj = camera.getProjectionMatrix(aspect);

    memcpy(geomUniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));

    LightingUBO lightUbo{};
    lightUbo.viewPos = camera.getPosition();
    lightUbo.debugMode = debugMode;
    lightUbo.dirLightCount = static_cast<int>(dirLights.size());
    lightUbo.exposure = exposure;

    // With useLightVolume=true the fullscreen lighting.frag skips
    // the point/spot loops (handled by the light volume proxy pass instead).
    // Directional light is always drawn by the fullscreen pass.
    // useLightVolume=false uses the original all-lights loop path (for comparison).
    if (useLightVolume)
    {
        lightUbo.lightCount = 0;
        lightUbo.spotLightCount = 0;
    }
    else
    {
        lightUbo.lightCount = static_cast<int>(lights.size());
        lightUbo.spotLightCount = static_cast<int>(spotLights.size());
    }
    memcpy(lightUniformBuffersMapped[frameIndex], &lightUbo, sizeof(lightUbo));

    // ===========================================================
    // Fill the directional shadow UBO + update ShadowPass VP
    // -----------------------------------------------------------
    // Finds the first directional light with castShadows = true in the ECS;
    // if none, shadow lookup is disabled. Matches Systems::gatherDirLights:
    // the direction comes from the TransformComponent.rotation forward axis.
    // ===========================================================
    struct DirShadowUBOLayout
    {
        alignas(16) glm::mat4 lightViewProj;
        int enabled;
        float invMapSize;
        float _pad0;
        float _pad1;
    };
    DirShadowUBOLayout shUbo{};
    shUbo.lightViewProj = glm::mat4(1.0f);
    shUbo.enabled = 0;
    shUbo.invMapSize = 1.0f / static_cast<float>(shadowMapResolution);

    bool foundCaster = false;
    if (shadowsEnabled)
    {
        auto view = scene.registry.view<DirectionalLightComponent, TransformComponent>();
        for (auto e : view)
        {
            const auto &dlc = view.get<DirectionalLightComponent>(e);
            if (!dlc.castShadows)
                continue;
            const auto &tc = view.get<TransformComponent>(e);
            // Same convention as Systems::gatherDirLights: forward = rot * (0,0,-1)
            // Use orientation quaternion to be consistent with the ECS data path.
            glm::vec3 fwd = glm::normalize(tc.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
            shadowPass.updateLightSpaceVP(frameIndex, fwd, camera.getPosition(), shadowDistance);
            shUbo.lightViewProj = shadowPass.lastLightViewProj();
            shUbo.enabled = 1;
            foundCaster = true;
            break; // Only the first directional light casts (Unity URP Main Light)
        }
    }
    if (frameIndex < dirShadowUBOsMapped.size() && dirShadowUBOsMapped[frameIndex])
    {
        std::memcpy(dirShadowUBOsMapped[frameIndex], &shUbo, sizeof(shUbo));
    }
    // record reads this flag to decide whether to actually record ShadowPass
    // (skipped when there is no caster).
    currentFrameHasShadowCaster = foundCaster;

    // ===========================================================
    // Spot shadow Top-N selection + UBO writes
    // -----------------------------------------------------------
    // 1) scan all SpotLightComponents with castShadows = true
    // 2) sort by length(color) * intensity descending, take top N (=4)
    // 3) write shadowSlot = k into each selected spot's spotLights[] entry
    // 4) call spotShadowPass.updateLightSpaceVP(frameIndex, k, ...)
    // 5) copy the N lightVPs into spotShadowUBO[frame]
    // ===========================================================
    struct SpotShadowUBOLayout
    {
        alignas(16) glm::mat4 spotLightVP[SpotShadowPass::MAX_SPOT_SHADOW_SLOTS];
        alignas(16) glm::ivec4 validSlots;
        alignas(16) glm::vec4 params;
    };
    SpotShadowUBOLayout spUbo{};
    for (uint32_t k = 0; k < SpotShadowPass::MAX_SPOT_SHADOW_SLOTS; ++k)
        spUbo.spotLightVP[k] = glm::mat4(1.0f);
    spUbo.validSlots = glm::ivec4(0);
    spUbo.params = glm::vec4(1.0f / static_cast<float>(spotShadowMapResolution), 0, 0, 0);
    currentFrameSpotShadowSlots = 0;
    for (auto &S : currentFrameSpotShadows)
        S = SelectedSpotShadow{};

    if (shadowsEnabled)
    {
        // Fix: selection must be based on spotLights[] (the
        // frustum-culled GPU list), not the ECS view - otherwise, once the
        // camera frustum culls some spots, the ECS view indices drift from the
        // real spotLights[] subscripts and shadowSlot gets written to the wrong
        // light. Also, does the SpotLight SSBO carry a castShadows flag? No -
        // there is no castShadows field in the SSBO. So to look up lights by
        // worldPos/worldDir/score we must detour: collect the matching ECS
        // entities in parallel (keeping access to the ECS-level castShadows).
        //
        // Implementation: walk the view in the same order as
        // Systems::gatherSpotLights, but only collect lights that made it into
        // spotLights[] (i.e. not frustum-culled). We re-run the view while
        // tracking the spotLights[] write pointer.
        struct Candidate
        {
            int spotIndex; // subscript into spotLights[]
            float score;
            glm::vec3 worldPos;
            glm::vec3 worldDir;
            float outerAngle;
            float range;
        };
        std::vector<Candidate> cands;
        cands.reserve(16);

        // Rebuild the frustum to replicate updateLightBuffers' culling behavior.
        const float aspect = static_cast<float>(viewportExtent.width) /
                             static_cast<float>(viewportExtent.height);
        const glm::mat4 vp = camera.getProjectionMatrix(aspect) * camera.getViewMatrix();
        const FrustumCulling::Frustum camFrustum =
            FrustumCulling::extractFromViewProjection(vp);

        int spotIdx = 0; // actual subscript into spotLights[]
        auto svw = scene.registry.view<SpotLightComponent, TransformComponent>();
        for (auto e : svw)
        {
            const auto &slc = svw.get<SpotLightComponent>(e);
            const auto &tc = svw.get<TransformComponent>(e);
            glm::vec3 wpos = glm::vec3(tc.worldMatrix[3]);
            // Culling rules identical to updateLightBuffers
            bool inSpotLights = true;
            if (lightFrustumCullEnabled)
                inSpotLights = FrustumCulling::testSphere(camFrustum, wpos, slc.radius);
            if (!inSpotLights)
                continue; // this light did not make it into spotLights[]
            if (spotIdx >= static_cast<int>(MAX_SPOT_LIGHTS))
                break; // lights truncated in updateLightBuffers don't participate either
            if (slc.castShadows)
            {
                Candidate c{};
                c.spotIndex = spotIdx;
                c.score = glm::length(slc.color) * slc.intensity;
                c.worldPos = wpos;
                c.worldDir = glm::normalize(glm::vec3(tc.worldMatrix * glm::vec4(0, 0, -1, 0)));
                c.outerAngle = slc.outerAngle;
                c.range = slc.radius;
                cands.push_back(c);
            }
            ++spotIdx;
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Candidate &a, const Candidate &b)
                  { return a.score > b.score; });
        const uint32_t take = std::min<uint32_t>(static_cast<uint32_t>(cands.size()),
                                                 SpotShadowPass::MAX_SPOT_SHADOW_SLOTS);
        for (uint32_t k = 0; k < take; ++k)
        {
            const auto &c = cands[k];
            spotShadowPass.updateLightSpaceVP(frameIndex, k,
                                              c.worldPos, c.worldDir,
                                              c.outerAngle, c.range);
            spUbo.spotLightVP[k] = spotShadowPass.lastLightViewProj(k);
            // Write the shadowSlot back into spotLights[] (filled earlier by gatherSpotLights)
            if (c.spotIndex >= 0 && c.spotIndex < static_cast<int>(spotLights.size()))
            {
                spotLights[c.spotIndex].shadowSlot = static_cast<int>(k);
            }
            // Save the selection result; record needs the frustum + caster collection
            if (k < currentFrameSpotShadows.size())
            {
                currentFrameSpotShadows[k].lightViewProj = spUbo.spotLightVP[k];
                currentFrameSpotShadows[k].lightPos = c.worldPos;
                currentFrameSpotShadows[k].lightDir = c.worldDir;
                currentFrameSpotShadows[k].outerAngle = c.outerAngle;
                currentFrameSpotShadows[k].range = c.range;
            }
        }
        spUbo.validSlots.x = static_cast<int>(take);
        currentFrameSpotShadowSlots = take;
        // spotLights[] changed (shadowSlot), so re-memcpy it into the GPU SSBO
        if (frameIndex < spotLightSSBOsMapped.size() && spotLightSSBOsMapped[frameIndex] && !spotLights.empty())
        {
            std::memcpy(spotLightSSBOsMapped[frameIndex], spotLights.data(),
                        sizeof(SpotLight) * std::min<size_t>(spotLights.size(), MAX_SPOT_LIGHTS));
        }
    }
    if (frameIndex < spotShadowUBOsMapped.size() && spotShadowUBOsMapped[frameIndex])
    {
        std::memcpy(spotShadowUBOsMapped[frameIndex], &spUbo, sizeof(spUbo));
    }

    // ===========================================================
    // Point shadow Top-N selection + UBO writes
    // -----------------------------------------------------------
    // 1) scan all PointLightComponents with castShadows = true
    // 2) sort by length(color) * intensity descending, take top N (=4)
    // 3) write shadowSlot = k into each selected point's lights[] entry
    // 4) call pointShadowPass.updateLightSpaceVP(frameIndex, k, ...)
    // 5) copy the N lightPosRange values into pointShadowUBO[frame]
    //
    // Same alignment strategy as spot selection: candidate.lightIndex must
    // exactly match the lights[] subscript after updateLightBuffers, or
    // shadowSlot would be written to the wrong light.
    // ===========================================================
    struct PointShadowUBOLayout
    {
        alignas(16) glm::vec4 lightPosRange[PointShadowPass::MAX_POINT_SHADOW_SLOTS];
        alignas(16) glm::ivec4 validSlots;
        alignas(16) glm::vec4 params[PointShadowPass::MAX_POINT_SHADOW_SLOTS];
    };
    PointShadowUBOLayout pUbo{};
    for (uint32_t k = 0; k < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++k)
    {
        pUbo.lightPosRange[k] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        pUbo.params[k] = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f); // default multiplier 1.0
    }
    pUbo.validSlots = glm::ivec4(0);
    currentFramePointShadowSlots = 0;
    for (auto &P : currentFramePointShadows)
        P = SelectedPointShadow{};

    if (shadowsEnabled)
    {
        struct PointCandidate
        {
            int lightIndex; // subscript into lights[]
            float score;
            glm::vec3 worldPos;
            float range;
            // Per-light shadow bias (copied from PointLightComponent)
            float slopeBiasMul;
            float constantBiasMul;
            float normalBiasMul;
            float depthBiasMul;
        };
        std::vector<PointCandidate> pcands;
        pcands.reserve(16);

        // Same camera frustum as spot selection, to replicate updateLightBuffers' culling rules
        const float aspectP = static_cast<float>(viewportExtent.width) /
                              static_cast<float>(viewportExtent.height);
        const glm::mat4 vpP = camera.getProjectionMatrix(aspectP) * camera.getViewMatrix();
        const FrustumCulling::Frustum camFrustumP =
            FrustumCulling::extractFromViewProjection(vpP);

        int pointIdx = 0; // actual subscript into lights[]
        auto pvw = scene.registry.view<PointLightComponent, TransformComponent>();
        for (auto e : pvw)
        {
            const auto &plc = pvw.get<PointLightComponent>(e);
            const auto &tc = pvw.get<TransformComponent>(e);
            glm::vec3 wpos = glm::vec3(tc.worldMatrix[3]);
            // Culling rules identical to updateLightBuffers
            bool inLights = true;
            if (lightFrustumCullEnabled)
                inLights = FrustumCulling::testSphere(camFrustumP, wpos, plc.radius);
            if (!inLights)
                continue;
            if (pointIdx >= static_cast<int>(MAX_POINT_LIGHTS))
                break;
            if (plc.castShadows)
            {
                PointCandidate c{};
                c.lightIndex = pointIdx;
                c.score = glm::length(plc.color) * plc.intensity;
                c.worldPos = wpos;
                c.range = plc.radius;
                c.slopeBiasMul = plc.shadowSlopeBias;
                c.constantBiasMul = plc.shadowConstantBias;
                c.normalBiasMul = plc.shadowNormalBias;
                c.depthBiasMul = plc.shadowDepthBias;
                pcands.push_back(c);
            }
            ++pointIdx;
        }
        std::sort(pcands.begin(), pcands.end(),
                  [](const PointCandidate &a, const PointCandidate &b)
                  { return a.score > b.score; });
        const uint32_t takeP = std::min<uint32_t>(static_cast<uint32_t>(pcands.size()),
                                                  PointShadowPass::MAX_POINT_SHADOW_SLOTS);
        for (uint32_t k = 0; k < takeP; ++k)
        {
            const auto &c = pcands[k];
            pointShadowPass.updateLightSpaceVP(frameIndex, k, c.worldPos, c.range);
            pUbo.lightPosRange[k] = glm::vec4(c.worldPos, glm::max(c.range, 0.2f));
            // Write the per-slot frag-side bias multipliers
            // (raster bias is passed at record time)
            pUbo.params[k] = glm::vec4(c.normalBiasMul, c.depthBiasMul, 0.0f, 0.0f);
            // Write the shadowSlot back into lights[]
            if (c.lightIndex >= 0 && c.lightIndex < static_cast<int>(lights.size()))
            {
                lights[c.lightIndex].shadowSlot = static_cast<int>(k);
            }
            // Save the selection result; record needs caster distance-sphere
            // culling + the raster bias
            if (k < currentFramePointShadows.size())
            {
                // Base bias (aligned with Unity URP defaults)
                //   constant base = 0.5f: guards against D32 quantization noise
                //   slope base    = 1.5f: covers grazing-angle face acne
                // Peter Panning is mainly controlled by the frag-side dual-track
                // bias; raster only handles quantization noise and slope acne, to
                // avoid double over-biasing with the frag side.
                constexpr float kRasterConstantBase = 0.5f;
                constexpr float kRasterSlopeBase = 1.5f;
                currentFramePointShadows[k].lightPos = c.worldPos;
                currentFramePointShadows[k].range = c.range;
                currentFramePointShadows[k].rasterConstantBias = kRasterConstantBase * c.constantBiasMul;
                currentFramePointShadows[k].rasterSlopeBias = kRasterSlopeBase * c.slopeBiasMul;
                currentFramePointShadows[k].fragNormalBiasMul = c.normalBiasMul;
                currentFramePointShadows[k].fragDepthBiasMul = c.depthBiasMul;
            }
        }
        pUbo.validSlots.x = static_cast<int>(takeP);
        currentFramePointShadowSlots = takeP;
        // lights[] changed (shadowSlot), so re-memcpy it into the GPU SSBO
        if (frameIndex < lightSSBOsMapped.size() && lightSSBOsMapped[frameIndex] && !lights.empty())
        {
            std::memcpy(lightSSBOsMapped[frameIndex], lights.data(),
                        sizeof(PointLight) * std::min<size_t>(lights.size(), MAX_POINT_LIGHTS));
        }
    }
    if (frameIndex < pointShadowUBOsMapped.size() && pointShadowUBOsMapped[frameIndex])
    {
        std::memcpy(pointShadowUBOsMapped[frameIndex], &pUbo, sizeof(pUbo));
    }
}

// ============================================================================
// drawFrame()
// ============================================================================
void Renderer::drawFrame()
{
    VkDevice device = vulkanContext.getDevice();

    float currentTime = static_cast<float>(glfwGetTime());
    float deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;
    if (cursorCaptured)
    {
        camera.processKeyboard(window, deltaTime);
    }

    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // The fence is waited, meaning this slot's GPU commands from
    // the last frame finished, so query results can be read back
    gpuProfiler.resolve(currentFrame);

    // Deferred viewport resize. EditorUI only sets the
    // pending flag when it detects a panel size change mid-ImGui-frame; here,
    // with currentFrame's fence already waited, is the safe point to actually
    // rebuild. onViewportResize_RT() waits the other in-flight slots' fences
    // internally (usually already done or nearly done), which is far cheaper
    // than vkDeviceWaitIdle.
    if (viewportResizePending)
    {
        if (pendingViewportExtent.width > 0 && pendingViewportExtent.height > 0 &&
            (pendingViewportExtent.width != viewportExtent.width ||
             pendingViewportExtent.height != viewportExtent.height))
        {
            viewportExtent = pendingViewportExtent;
            onViewportResize_RT();
        }
        viewportResizePending = false;
    }

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain.getSwapchain(), UINT64_MAX,
                                            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchainResources();
        return;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swap chain image!");

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    // CPU profiler frame start
    cpuProfiler.beginFrame();
    cpuProfiler.beginPass("Frame Total");

    cpuProfiler.beginPass("Physics");
    Systems::updateTransforms(scene.registry);

    // Physics update (runs after TransformSystem; worldMatrix is refreshed
    // again once the new positions are written back)
    PhysicsSystem::update(scene.registry, physicsWorld, deltaTime);
    if (PhysicsSystem::simulationRunning)
    {
        Systems::updateTransforms(scene.registry);
    }
    cpuProfiler.endPass("Physics");

    // Drive user-script `on_update(dt)` while in Play mode.
    // Placed AFTER the physics step (and the post-physics transform sync)
    // so scripts observe the same world-space state the renderer is about
    // to draw - and so script-side Transform writes land on top of the
    // physics result instead of being overwritten by it.
    //
    // Stop/Pause both skip the call: editor-time poking via the Inspector
    // must not be polluted by per-frame script logic. ScriptEngine itself
    // also short-circuits when scriptFaulted_ is set, so a runaway exception
    // doesn't spam the console every frame.
    //
    // Drive callOnUpdateAll() over per-entity ScriptComponents.
    // syncFromScene() runs every frame (cheap when nothing changed) so
    // adding/removing/changing ScriptComponents in the Inspector takes
    // effect on the very next tick - attach happens immediately, on_start
    // still waits for Play (intentional: matches Unity MonoBehaviour).
    if (scriptEngine && scriptEngine->isInitialized())
    {
        cpuProfiler.beginPass("Script");
        scriptEngine->syncFromScene(scene);
        if (PhysicsSystem::simulationRunning && !PhysicsSystem::simulationPaused)
        {
            scriptEngine->callOnUpdateAll(scene, deltaTime);
            // Re-run TransformSystem so any TRS edits made by per-entity
            // scripts are turned into worldMatrix updates BEFORE rendering /
            // BVH rebuild. (Same rationale as the legacy call below.)
            Systems::updateTransforms(scene.registry);
        }
        cpuProfiler.endPass("Script");
    }

    // worldMatrix is now final, so rebuild the render BVH.
    // Full rebuild every frame (N is small; simpler and equivalent to
    // maintaining incremental proxies).
    if (bvhCullingEnabled && frustumCullingEnabled)
    {
        cpuProfiler.beginPass("BVH Rebuild");
        renderBVH.rebuildFromScene(scene.registry);
        cpuProfiler.endPass("BVH Rebuild");
    }

    // Lights are gathered + frustum-culled + uploaded to the
    // current frame's host-mapped SSBO every frame (replaces the old
    // lightsDirty rebuild path; editor light edits take effect next frame)
    cpuProfiler.beginPass("Light Gather + Cull");
    updateLightBuffers(currentFrame);
    cpuProfiler.endPass("Light Gather + Cull");
    lightsDirty = false; // kept for legacy code that writes this field

    cpuProfiler.beginPass("Update Buffers");
    updateUniformBuffer(currentFrame);
    cpuProfiler.endPass("Update Buffers");

    cpuProfiler.beginPass("EditorUI Build");
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    editorUI.drawEditorUI();
    ImGui::Render();
    cpuProfiler.endPass("EditorUI Build");

    cpuProfiler.beginPass("Record Cmd");
    VkCommandBuffer cmd = commandManager.getCommandBuffer(currentFrame);
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex);
    cpuProfiler.endPass("Record Cmd");

    VkSemaphore waitSem[] = {imageAvailableSemaphores[currentFrame]};
    VkSemaphore signalSem[] = {renderFinishedSemaphores[imageIndex]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSem;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSem;
    vkQueueSubmit(vulkanContext.getGraphicsQueue(), 1, &submitInfo, inFlightFences[currentFrame]);

    VkSwapchainKHR swapChains[] = {swapchain.getSwapchain()};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSem;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    result = vkQueuePresentKHR(vulkanContext.getPresentQueue(), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapchainResources();
    }

    // CPU profiler frame end (Frame Total is recorded after
    // present, including the GPU wait; symmetric with GpuProfiler's Frame
    // Total timing: that is actual GPU work time, while the CPU Frame Total is
    // the actual drawFrame duration on the CPU side)
    cpuProfiler.endPass("Frame Total");
    cpuProfiler.endFrame();

    // Benchmark advance (warmup -> sampling -> CSV write).
    // CPU Frame Total is used as frame_ms (includes the GPU wait); the GPU Pass
    // Frame Total is gpu_ms; both are already averaged in endFrame.
    {
        const auto &cEntries = cpuProfiler.getEntries();
        auto cIt = cEntries.find("Frame Total");
        double cpuFrameMs = (cIt != cEntries.end()) ? cIt->second.lastMs : 0.0;

        double gpuFrameMs = 0.0;
        const auto &gStats = gpuProfiler.getPassStats();
        auto gIt = gStats.find("Frame Total");
        if (gIt != gStats.end())
            gpuFrameMs = gIt->second.lastMs;

        // During a benchmark, use the CPU frame as the "real frame time"
        // (more stable, includes present wait).
        // Pass the five-bucket DrawCallStats breakdown so the CSV
        // writes geometry / lightVolume / lighting / gizmo / imgui columns
        // separately, for cross-engine comparison alignment.
        benchmarkRunner.tick(cpuFrameMs, gpuFrameMs, cpuFrameMs,
                             lastDrawCallStats.total(),
                             lastDrawCallStats.geometry,
                             lastDrawCallStats.lightVolume,
                             lastDrawCallStats.lighting,
                             lastDrawCallStats.gizmo,
                             lastDrawCallStats.imgui);
    }

    // With targetFps > 0, sleep on the CPU until the target
    // frame time. Timing note: endFrame already recorded this frame's Frame
    // Total (without the sleep); the sleep only affects the next frame's
    // deltaTime, so it cannot pollute the profiler data.
    if (targetFps > 0)
    {
        const double targetFrameMs = 1000.0 / static_cast<double>(targetFps);
        // frameTimer records the real moment of "previous frame end"
        // (drawFrame call interval)
        static auto lastEnd = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(now - lastEnd).count();
        double remainMs = targetFrameMs - elapsedMs;
        if (remainMs > 0.5)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int64_t>(remainMs * 1000.0)));
        }
        lastEnd = std::chrono::high_resolution_clock::now();
    }

    // Frame counter (for headless benchmarks)
    ++frameCount;

    // Benchmark done + autoExitOnBenchmarkDone -> close window
    // Uses prevBenchmarkRunning edge detection so the Idle state is not
    // mistaken for "finished"
    {
        bool runningNow = benchmarkRunner.isRunning();
        if (autoExitOnBenchmarkDone && prevBenchmarkRunning && !runningNow)
        {
            std::printf("[Bench] benchmark done, requesting window close\n");
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        prevBenchmarkRunning = runningNow;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ============================================================================
// recordCommandBuffer()
// ============================================================================
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Zero the draw call counters at the start of each frame
    // (gizmo / imgui are accumulated via external add calls, also within this
    // frame's cycle)
    curDrawCallStats = DrawCallStats{};

    // Profiler frame start (reset query pool + overall timing)
    gpuProfiler.beginFrame(cmd, currentFrame);
    gpuProfiler.beginPass(cmd, "Frame Total");

    VkExtent2D sceneExtent = viewportExtent;
    VkExtent2D swapExtent = swapchain.getExtent();
    VkImage swapImg = swapchain.getImages()[imageIndex];
    VkImageView swapView = swapchain.getImageViews()[imageIndex];

    // ========================================================================
    // Pre-pass - collect visible draw items + upload instance SSBO.
    // ------------------------------------------------------------------------
    // Culling/sort/upload is hoisted before the Geometry Pass so Geometry Pass
    // and Directional Shadow Pass share one drawItems / instance SSBO, avoiding
    // a second BVH + ECS walk. drawItems is only valid for this function's
    // lifetime.
    // ========================================================================
    struct DrawItem
    {
        MeshReference *mesh;
        VkDescriptorSet texDS;
        float hasAlbedoTex;
        InstanceData inst;
    };
    std::vector<DrawItem> drawItems;
    drawItems.reserve(256);
    lastCullStats = {0, 0};
    {
        const float aspect = static_cast<float>(viewportExtent.width) /
                             static_cast<float>(viewportExtent.height);
        const glm::mat4 vp = camera.getProjectionMatrix(aspect) * camera.getViewMatrix();
        const FrustumCulling::Frustum frustum = FrustumCulling::extractFromViewProjection(vp);

        auto collectEntity = [&](entt::entity e, MeshComponent &mc, TransformComponent &tc)
        {
            DrawItem item{};
            item.mesh = mc.mesh.get();
            item.inst.model = tc.worldMatrix;

            VkDescriptorSet texDS = defaultTexDescriptorSet;
            float hasAlbedoTex = 0.0f;

            if (scene.registry.all_of<MaterialComponent>(e))
            {
                auto &mat = scene.registry.get<MaterialComponent>(e);
                item.inst.albedoAndMetallic = glm::vec4(mat.albedo, mat.metallic);
                item.inst.roughnessAndFlags = glm::vec4(mat.roughness, 0.0f, 0.0f, 0.0f);
                if (mat.albedoTexture && mat.textureDescriptorSet != VK_NULL_HANDLE)
                {
                    texDS = mat.textureDescriptorSet;
                    hasAlbedoTex = 1.0f;
                }
            }
            else
            {
                item.inst.albedoAndMetallic = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
                item.inst.roughnessAndFlags = glm::vec4(0.5f, 0.0f, 0.0f, 0.0f);
            }
            item.inst.roughnessAndFlags.g = hasAlbedoTex;
            item.texDS = texDS;
            item.hasAlbedoTex = hasAlbedoTex;
            drawItems.push_back(item);
        };

        auto view = scene.registry.view<MeshComponent, TransformComponent>();

        if (frustumCullingEnabled && bvhCullingEnabled)
        {
            for (auto e : view)
            {
                auto &mc = view.get<MeshComponent>(e);
                if (!mc.mesh)
                    continue;
                lastCullStats.total++;
            }
            renderBVH.queryFrustum(frustum, [&](entt::entity e)
                                   {
                if (!scene.registry.valid(e)) return;
                if (!scene.registry.all_of<MeshComponent, TransformComponent>(e)) return;
                auto &mc = scene.registry.get<MeshComponent>(e);
                auto &tc = scene.registry.get<TransformComponent>(e);
                if (!mc.mesh) return;
                glm::vec3 wmin, wmax;
                FrustumCulling::transformLocalAABBToWorld(
                    tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
                    wmin, wmax);
                if (!FrustumCulling::testAABB(frustum, wmin, wmax)) return;
                lastCullStats.visible++;
                collectEntity(e, mc, tc); });
        }
        else
        {
            for (auto e : view)
            {
                auto &mc = view.get<MeshComponent>(e);
                auto &tc = view.get<TransformComponent>(e);
                if (!mc.mesh)
                    continue;
                lastCullStats.total++;
                if (frustumCullingEnabled)
                {
                    glm::vec3 wmin, wmax;
                    FrustumCulling::transformLocalAABBToWorld(
                        tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
                        wmin, wmax);
                    if (!FrustumCulling::testAABB(frustum, wmin, wmax))
                        continue;
                }
                lastCullStats.visible++;
                collectEntity(e, mc, tc);
            }
        }

        // Sort by (mesh, texDS) for instance batching.
        if (!drawItems.empty())
        {
            std::sort(drawItems.begin(), drawItems.end(),
                      [](const DrawItem &a, const DrawItem &b)
                      {
                          if (a.mesh != b.mesh)
                              return a.mesh < b.mesh;
                          return a.texDS < b.texDS;
                      });
        }

        // Truncate to MAX_INSTANCES (warn-once).
        if (drawItems.size() > MAX_INSTANCES)
        {
            static bool warned = false;
            if (!warned)
            {
                std::cerr << "[Renderer] InstanceData buffer overflow ("
                          << drawItems.size() << " > " << MAX_INSTANCES << "), truncating.\n";
                warned = true;
            }
            drawItems.resize(MAX_INSTANCES);
        }

        // Upload to per-frame host-mapped SSBO in a single memcpy. Both
        // Geometry Pass and ShadowPass read from this buffer in the same frame.
        if (!drawItems.empty())
        {
            InstanceData *dst = reinterpret_cast<InstanceData *>(instanceSSBOsMapped[currentFrame]);
            for (uint32_t i = 0; i < drawItems.size(); ++i)
                dst[i] = drawItems[i].inst;
        }
    }

    // Helper: walk drawItems and emit one instanced draw per (mesh, texDS) run.
    // Used by Geometry Pass (camera-frustum culled drawItems, with texDS bind).
    auto issueBatchedDraws = [&](VkPipelineLayout layout, bool bindTex)
    {
        const uint32_t count = static_cast<uint32_t>(drawItems.size());
        uint32_t i = 0;
        uint32_t draws = 0;
        while (i < count)
        {
            MeshReference *batchMesh = drawItems[i].mesh;
            VkDescriptorSet batchTex = drawItems[i].texDS;
            float batchHasTex = drawItems[i].hasAlbedoTex;
            uint32_t batchStart = i;
            uint32_t batchEnd = i + 1;
            while (batchEnd < count &&
                   drawItems[batchEnd].mesh == batchMesh &&
                   drawItems[batchEnd].texDS == batchTex)
            {
                ++batchEnd;
            }

            GeometryPushConstants pc{};
            pc.instanceOffset = batchStart;
            pc.hasAlbedoTex = batchHasTex;
            vkCmdPushConstants(cmd, layout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GeometryPushConstants), &pc);

            if (bindTex)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        layout, 1, 1, &batchTex, 0, nullptr);
            }

            VkBuffer vbufs[] = {batchMesh->vertexBuffer.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, batchMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, batchMesh->indexCount, batchEnd - batchStart, 0, 0, 0);
            ++draws;
            i = batchEnd;
        }
        return draws;
    };

    // ========================================================================
    // Fix: dedicated shadow drawItems - culled with the LIGHT frustum
    // ------------------------------------------------------------------------
    // Problem: ShadowPass used to reuse the camera-frustum-culled drawItems,
    // so shadows vanished for objects culled by the camera but still visible
    // in the shadow-relevant region.
    // Fix: collect + sort + upload independently for ShadowPass, using the
    // light's orthographic frustum from shadowPass.lastLightViewProj().
    // ========================================================================
    std::vector<DrawItem> shadowDrawItems;
    if (shadowsEnabled && currentFrameHasShadowCaster)
    {
        shadowDrawItems.reserve(drawItems.size() + 32);
        const FrustumCulling::Frustum lightFrustum =
            FrustumCulling::extractFromViewProjection(shadowPass.lastLightViewProj());

        auto collectForShadow = [&](entt::entity e, MeshComponent &mc, TransformComponent &tc)
        {
            DrawItem item{};
            item.mesh = mc.mesh.get();
            item.inst.model = tc.worldMatrix;

            // ShadowPass needs no texture / material parameters, only model.
            // But the SSBO layout is the shared InstanceData 96B layout, so
            // albedo/metallic etc. are filled with zeros (shadow.vert doesn't
            // read those fields).
            item.inst.albedoAndMetallic = glm::vec4(0.0f);
            item.inst.roughnessAndFlags = glm::vec4(0.0f);
            item.texDS = VK_NULL_HANDLE;
            item.hasAlbedoTex = 0.0f;
            shadowDrawItems.push_back(item);
        };

        auto svcView = scene.registry.view<MeshComponent, TransformComponent>();
        for (auto e : svcView)
        {
            auto &mc = svcView.get<MeshComponent>(e);
            auto &tc = svcView.get<TransformComponent>(e);
            if (!mc.mesh)
                continue;

            if (frustumCullingEnabled)
            {
                glm::vec3 wmin, wmax;
                FrustumCulling::transformLocalAABBToWorld(
                    tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
                    wmin, wmax);
                if (!FrustumCulling::testAABB(lightFrustum, wmin, wmax))
                    continue; // fully outside the light's orthographic frustum -> can't reach the shadow map
            }
            collectForShadow(e, mc, tc);
        }

        // Sort by mesh for instance batching (ShadowPass ignores textures, only groups by mesh).
        if (!shadowDrawItems.empty())
        {
            std::sort(shadowDrawItems.begin(), shadowDrawItems.end(),
                      [](const DrawItem &a, const DrawItem &b)
                      { return a.mesh < b.mesh; });
        }

        const uint32_t maxShadow = ShadowPass::MAX_SHADOW_INSTANCES;
        if (shadowDrawItems.size() > maxShadow)
        {
            static bool warnedShadow = false;
            if (!warnedShadow)
            {
                std::cerr << "[Renderer] Shadow InstanceData buffer overflow ("
                          << shadowDrawItems.size() << " > " << maxShadow << "), truncating.\n";
                warnedShadow = true;
            }
            shadowDrawItems.resize(maxShadow);
        }

        // Upload into ShadowPass's own instance SSBO.
        if (!shadowDrawItems.empty())
        {
            void *dst = shadowPass.getInstanceSSBOMapped(currentFrame);
            if (dst)
            {
                InstanceData *typed = reinterpret_cast<InstanceData *>(dst);
                for (uint32_t i = 0; i < shadowDrawItems.size(); ++i)
                    typed[i] = shadowDrawItems[i].inst;
            }
        }
    }

    // Helper: walk shadowDrawItems and emit instanced draws (no texture bind).
    auto issueShadowDraws = [&](VkPipelineLayout layout)
    {
        const uint32_t count = static_cast<uint32_t>(shadowDrawItems.size());
        uint32_t i = 0;
        uint32_t draws = 0;
        while (i < count)
        {
            MeshReference *batchMesh = shadowDrawItems[i].mesh;
            uint32_t batchStart = i;
            uint32_t batchEnd = i + 1;
            while (batchEnd < count && shadowDrawItems[batchEnd].mesh == batchMesh)
                ++batchEnd;

            GeometryPushConstants pc{};
            pc.instanceOffset = batchStart;
            pc.hasAlbedoTex = 0.0f;
            vkCmdPushConstants(cmd, layout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GeometryPushConstants), &pc);

            VkBuffer vbufs[] = {batchMesh->vertexBuffer.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, batchMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, batchMesh->indexCount, batchEnd - batchStart, 0, 0, 0);
            ++draws;
            i = batchEnd;
        }
        return draws;
    };

    // ========================================================================
    // Directional Shadow Pass (pre-Geometry, depth-only)
    // ------------------------------------------------------------------------
    // Runs BEFORE Geometry Pass so the shadow map is already in
    // SHADER_READ_ONLY layout when Lighting Pass starts. ShadowPass uses its
    // OWN instance SSBO + LIGHT-frustum-culled shadowDrawItems (independent
    // from the camera-frustum-culled drawItems used by Geometry Pass).
    // ========================================================================
    if (shadowsEnabled && currentFrameHasShadowCaster && !shadowDrawItems.empty())
    {
        gpuProfiler.beginPass(cmd, "DirShadow");
        shadowPass.record(cmd, currentFrame,
                          [&](VkCommandBuffer c, VkPipelineLayout layout)
                          {
                              (void)c;
                              issueShadowDraws(layout);
                          });
        gpuProfiler.endPass(cmd, "DirShadow");
    }
    else
    {
        // No caster / empty shadow draw list: still need shadow map in
        // SHADER_READ_ONLY layout for the lighting descriptor.
        shadowPass.transitionToReadOnly(cmd, /*firstUse=*/false);
    }

    // ========================================================================
    // Spot Shadow Pass (pre-Geometry, depth-only, 4 slots)
    // ------------------------------------------------------------------------
    // For each active slot:
    //   1) extract the frustum from the lightViewProj saved during selection
    //   2) walk all ECS meshes, culling against the spot frustum
    //   3) upload into that slot's own instance SSBO
    //   4) call spotShadowPass.recordSlot(cmd, currentFrame, slot, drawFn)
    // Inactive slots go through transitionSlotToReadOnly so the lighting
    // descriptor stays readable.
    // ========================================================================
    if (shadowsEnabled && currentFrameSpotShadowSlots > 0)
    {
        gpuProfiler.beginPass(cmd, "SpotShadow");
        // Temp cache: avoids lambda closures capturing short-lived objects
        std::array<std::vector<DrawItem>, SpotShadowPass::MAX_SPOT_SHADOW_SLOTS> slotDrawItems;
        for (uint32_t s = 0; s < currentFrameSpotShadowSlots; ++s)
        {
            const auto &sel = currentFrameSpotShadows[s];

            auto &items = slotDrawItems[s];
            items.reserve(64);
            auto svcView = scene.registry.view<MeshComponent, TransformComponent>();
            for (auto e : svcView)
            {
                auto &mc = svcView.get<MeshComponent>(e);
                auto &tc = svcView.get<TransformComponent>(e);
                if (!mc.mesh)
                    continue;
                // Sphere culling with "distance <= range + AABB
                // radius". Gribb-Hartmann frustum is not used because with the
                // Y-flipped perspective VP matrix its boundary planes can
                // invert, causing over-culling (measured: keep counts varied
                // wildly across slots in the same scene). Spot range is
                // usually small (12-20 in the demo), so sphere culling is
                // conservative enough and very cheap.
                {
                    glm::vec3 wmin, wmax;
                    FrustumCulling::transformLocalAABBToWorld(
                        tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
                        wmin, wmax);
                    glm::vec3 center = (wmin + wmax) * 0.5f;
                    float radius = glm::length((wmax - wmin) * 0.5f);
                    float dist = glm::length(center - sel.lightPos);
                    if (dist > sel.range + radius)
                        continue;
                }
                DrawItem item{};
                item.mesh = mc.mesh.get();
                item.inst.model = tc.worldMatrix;
                item.inst.albedoAndMetallic = glm::vec4(0.0f);
                item.inst.roughnessAndFlags = glm::vec4(0.0f);
                item.texDS = VK_NULL_HANDLE;
                item.hasAlbedoTex = 0.0f;
                items.push_back(item);
            }

            if (items.empty())
            {
                // This slot has no casters this frame: keep the old shadow map
                // content (whatever the last frame left), but the layout must
                // still go to SHADER_READ_ONLY.
                spotShadowPass.transitionSlotToReadOnly(cmd, s);
                continue;
            }

            std::sort(items.begin(), items.end(),
                      [](const DrawItem &a, const DrawItem &b)
                      { return a.mesh < b.mesh; });

            const uint32_t maxInst = SpotShadowPass::MAX_INSTANCES_PER_SLOT;
            if (items.size() > maxInst)
            {
                static bool warnedSpot = false;
                if (!warnedSpot)
                {
                    std::cerr << "[Renderer] Spot shadow slot " << s
                              << " InstanceData buffer overflow ("
                              << items.size() << " > " << maxInst << "), truncating.\n";
                    warnedSpot = true;
                }
                items.resize(maxInst);
            }

            // Upload to slot's instance SSBO
            void *dst = spotShadowPass.getInstanceSSBOMapped(currentFrame, s);
            if (dst)
            {
                InstanceData *typed = reinterpret_cast<InstanceData *>(dst);
                for (uint32_t i = 0; i < items.size(); ++i)
                    typed[i] = items[i].inst;
            }

            // Record this slot
            const std::vector<DrawItem> &itemsRef = items;
            spotShadowPass.recordSlot(cmd, currentFrame, s,
                                      [&itemsRef, cmd](VkCommandBuffer c, VkPipelineLayout layout)
                                      {
                                          (void)c;
                                          const uint32_t count = static_cast<uint32_t>(itemsRef.size());
                                          uint32_t i = 0;
                                          while (i < count)
                                          {
                                              MeshReference *batchMesh = itemsRef[i].mesh;
                                              uint32_t batchStart = i;
                                              uint32_t batchEnd = i + 1;
                                              while (batchEnd < count && itemsRef[batchEnd].mesh == batchMesh)
                                                  ++batchEnd;

                                              GeometryPushConstants pc{};
                                              pc.instanceOffset = batchStart;
                                              pc.hasAlbedoTex = 0.0f;
                                              vkCmdPushConstants(cmd, layout,
                                                                 VK_SHADER_STAGE_VERTEX_BIT, 0,
                                                                 sizeof(GeometryPushConstants), &pc);

                                              VkBuffer vbufs[] = {batchMesh->vertexBuffer.buffer};
                                              VkDeviceSize offsets[] = {0};
                                              vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
                                              vkCmdBindIndexBuffer(cmd, batchMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                                              vkCmdDrawIndexed(cmd, batchMesh->indexCount, batchEnd - batchStart, 0, 0, 0);
                                              i = batchEnd;
                                          }
                                      });
        }
        // Transition the inactive slots to SHADER_READ_ONLY so the descriptor stays readable
        for (uint32_t s = currentFrameSpotShadowSlots; s < SpotShadowPass::MAX_SPOT_SHADOW_SLOTS; ++s)
            spotShadowPass.transitionSlotToReadOnly(cmd, s);
        gpuProfiler.endPass(cmd, "SpotShadow");
    }
    else
    {
        // No spot casters at all: transition every slot
        spotShadowPass.transitionAllToReadOnly(cmd);
    }

    // ========================================================================
    // Point shadow per-slot record (6 faces x N slots)
    // ------------------------------------------------------------------------
    // For each active slot:
    //   1) distance-sphere culling from the lightPos / range saved during
    //      selection (same as spot)
    //   2) upload into that slot's own instance SSBO (all 6 faces share one
    //      caster list)
    //   3) call pointShadowPass.recordSlot(cmd, currentFrame, slot, drawFn),
    //      which loops 6 faces with one begin/end + draw each
    // Inactive slots go through transitionSlotToReadOnly so the lighting
    // descriptor stays readable.
    // ========================================================================
    if (shadowsEnabled && currentFramePointShadowSlots > 0)
    {
        gpuProfiler.beginPass(cmd, "PointShadow");
        std::array<std::vector<DrawItem>, PointShadowPass::MAX_POINT_SHADOW_SLOTS> pointSlotDrawItems;
        for (uint32_t s = 0; s < currentFramePointShadowSlots; ++s)
        {
            const auto &sel = currentFramePointShadows[s];

            auto &items = pointSlotDrawItems[s];
            items.reserve(64);
            auto pvcView = scene.registry.view<MeshComponent, TransformComponent>();
            for (auto e : pvcView)
            {
                auto &mc = pvcView.get<MeshComponent>(e);
                auto &tc = pvcView.get<TransformComponent>(e);
                if (!mc.mesh)
                    continue;
                // Same distance-sphere culling as spot (a point light's
                // natural spherical range is simpler than a frustum)
                {
                    glm::vec3 wmin, wmax;
                    FrustumCulling::transformLocalAABBToWorld(
                        tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
                        wmin, wmax);
                    glm::vec3 center = (wmin + wmax) * 0.5f;
                    float radius = glm::length((wmax - wmin) * 0.5f);
                    float dist = glm::length(center - sel.lightPos);
                    if (dist > sel.range + radius)
                        continue;
                }
                DrawItem item{};
                item.mesh = mc.mesh.get();
                item.inst.model = tc.worldMatrix;
                item.inst.albedoAndMetallic = glm::vec4(0.0f);
                item.inst.roughnessAndFlags = glm::vec4(0.0f);
                item.texDS = VK_NULL_HANDLE;
                item.hasAlbedoTex = 0.0f;
                items.push_back(item);
            }

            if (items.empty())
            {
                pointShadowPass.transitionSlotToReadOnly(cmd, s);
                continue;
            }

            std::sort(items.begin(), items.end(),
                      [](const DrawItem &a, const DrawItem &b)
                      { return a.mesh < b.mesh; });

            const uint32_t maxInstP = PointShadowPass::MAX_INSTANCES_PER_SLOT;
            if (items.size() > maxInstP)
            {
                static bool warnedPoint = false;
                if (!warnedPoint)
                {
                    std::cerr << "[Renderer] Point shadow slot " << s
                              << " InstanceData buffer overflow ("
                              << items.size() << " > " << maxInstP << "), truncating.\n";
                    warnedPoint = true;
                }
                items.resize(maxInstP);
            }

            void *dst = pointShadowPass.getInstanceSSBOMapped(currentFrame, s);
            if (dst)
            {
                InstanceData *typed = reinterpret_cast<InstanceData *>(dst);
                for (uint32_t i = 0; i < items.size(); ++i)
                    typed[i] = items[i].inst;
            }

            // Note: drawFn pushes instanceOffset into push range offset=0
            // size=4; it must NOT push the whole GeometryPushConstants (the
            // point shadow pipeline layout's push range size is 8: the first
            // 4 bytes are instanceOffset, the last 4 are faceIndex, pushed by
            // recordSlot before each face).
            const std::vector<DrawItem> &itemsRef = items;
            pointShadowPass.recordSlot(cmd, currentFrame, s,
                                       sel.rasterConstantBias, sel.rasterSlopeBias,
                                       [&itemsRef, cmd](VkCommandBuffer c, VkPipelineLayout layout)
                                       {
                                           (void)c;
                                           const uint32_t count = static_cast<uint32_t>(itemsRef.size());
                                           uint32_t i = 0;
                                           while (i < count)
                                           {
                                               MeshReference *batchMesh = itemsRef[i].mesh;
                                               uint32_t batchStart = i;
                                               uint32_t batchEnd = i + 1;
                                               while (batchEnd < count && itemsRef[batchEnd].mesh == batchMesh)
                                                   ++batchEnd;

                                               // push only instanceOffset (4 bytes) at offset 0
                                               uint32_t inst = batchStart;
                                               vkCmdPushConstants(cmd, layout,
                                                                  VK_SHADER_STAGE_VERTEX_BIT,
                                                                  0, sizeof(uint32_t), &inst);

                                               VkBuffer vbufs[] = {batchMesh->vertexBuffer.buffer};
                                               VkDeviceSize offsets[] = {0};
                                               vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
                                               vkCmdBindIndexBuffer(cmd, batchMesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                                               vkCmdDrawIndexed(cmd, batchMesh->indexCount, batchEnd - batchStart, 0, 0, 0);
                                               i = batchEnd;
                                           }
                                       });
        }
        for (uint32_t s = currentFramePointShadowSlots; s < PointShadowPass::MAX_POINT_SHADOW_SLOTS; ++s)
            pointShadowPass.transitionSlotToReadOnly(cmd, s);
        gpuProfiler.endPass(cmd, "PointShadow");
    }
    else
    {
        pointShadowPass.transitionAllToReadOnly(cmd);
    }

    // === PASS 1: Geometry Pass ===
    gpuProfiler.beginPass(cmd, "Geometry");
    auto gbImages = gbuffer.getImages();
    for (auto &img : gbImages)
    {
        VulkanUtils::transitionImageLayout(cmd, img,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    VulkanUtils::transitionImageLayout(cmd, depthImage.image,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);

    std::array<VkRenderingAttachmentInfo, GBuffer::COUNT> gbAttachments{};
    for (uint32_t i = 0; i < GBuffer::COUNT; i++)
    {
        gbAttachments[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gbAttachments[i].imageView = gbuffer.getImageView(i);
        gbAttachments[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        gbAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbAttachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        gbAttachments[i].clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    }

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = depthImage.imageView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Depth must be sampled by the Hi-Z compute shader after
    // the Geometry Pass, so it must be STORED
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo geomRenderInfo{};
    geomRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    geomRenderInfo.renderArea = {{0, 0}, sceneExtent};
    geomRenderInfo.layerCount = 1;
    geomRenderInfo.colorAttachmentCount = GBuffer::COUNT;
    geomRenderInfo.pColorAttachments = gbAttachments.data();
    geomRenderInfo.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &geomRenderInfo);
    gpuProfiler.beginPipelineStats(cmd, "Geometry");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(sceneExtent.width);
    viewport.height = static_cast<float>(sceneExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, sceneExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            geomPipelineLayout, 0, 1, &geomDescriptorSets[currentFrame], 0, nullptr);

    // drawItems / instance SSBO were prepared in the Pre-pass;
    // here we just walk drawItems and issue instanced draws. The frustum cull
    // stats were also written to lastCullStats in the Pre-pass, so no re-walk
    // is needed in this stage.
    curDrawCallStats.geometry += issueBatchedDraws(geomPipelineLayout, /*bindTex=*/true);

    // Gizmo drawing (delegated to GizmoRenderer)
    gpuProfiler.endPipelineStats(cmd);    // Geometry stats exclude gizmo
    gpuProfiler.endPass(cmd, "Geometry"); // Geometry timing excludes gizmo / transition
    gpuProfiler.beginPass(cmd, "Gizmo");
    curDrawCallStats.gizmo += gizmoRenderer.recordGizmoCommands(cmd,
                                                                geomDescriptorSets[currentFrame], viewport, scissor, scene,
                                                                PhysicsSystem::simulationRunning);
    gpuProfiler.endPass(cmd, "Gizmo");

    vkCmdEndRendering(cmd);

    // === BARRIER ===
    for (auto &img : gbImages)
    {
        VulkanUtils::transitionImageLayout(cmd, img,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // === Hi-Z Build Pass (depth -> Hi-Z pyramid) ===
    // depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depthImage.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        b.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    gpuProfiler.beginPass(cmd, "HiZ Build");
    hiZPass.dispatch(cmd); // internally transitions all Hi-Z mips to SHADER_READ_ONLY_OPTIMAL
    gpuProfiler.endPass(cmd, "HiZ Build");

    // === PASS 1.5: SSAO Pass (view-space + bilateral blur + half-res) ===
    // Pipeline:
    //   (a) raw   : GBuffer (full-res Pos/Norm) -> ssaoRawImage (HALF-res, R8)
    //   (b) blurX : ssaoRawImage              -> ssaoBlurXImage (HALF-res, R8)
    //   (c) blurY : ssaoBlurXImage            -> ssaoImage     (FULL-res, R8, upsampled)
    // Composite samples ssaoImage. ssaoIntensity == 0 -> raw outputs constant
    // 1.0 -> blurs are no-ops -> composite multiplies by 1.0.
    gpuProfiler.beginPass(cmd, "SSAO");
    {
        // -- Upload SSAO UBO (view + proj + screenSize at half-res) --
        struct SsaoUBO
        {
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec2 screenSize;
            glm::vec2 _pad;
        };
        const float aspect = static_cast<float>(viewportExtent.width) /
                             static_cast<float>(viewportExtent.height);
        SsaoUBO sUbo{};
        sUbo.view = camera.getViewMatrix();
        sUbo.proj = camera.getProjectionMatrix(aspect);
        const uint32_t halfW = std::max(1u, viewportExtent.width / 2);
        const uint32_t halfH = std::max(1u, viewportExtent.height / 2);
        sUbo.screenSize = glm::vec2(static_cast<float>(halfW), static_cast<float>(halfH));
        sUbo._pad = glm::vec2(0.0f);
        std::memcpy(ssaoUniformBuffersMapped[currentFrame], &sUbo, sizeof(sUbo));

        // -- Upload SSAO Blur UBO (view + invInputSize at half-res) --
        struct BlurUBO
        {
            glm::mat4 view;
            glm::vec2 invInputSize;
            glm::vec2 _pad;
        };
        BlurUBO bUbo{};
        bUbo.view = sUbo.view;
        bUbo.invInputSize = glm::vec2(1.0f / static_cast<float>(halfW),
                                      1.0f / static_cast<float>(halfH));
        bUbo._pad = glm::vec2(0.0f);
        std::memcpy(ssaoBlurUniformBuffersMapped[currentFrame], &bUbo, sizeof(bUbo));

        // -- (a) Raw SSAO @ half-res -> ssaoRawImage --
        VulkanUtils::transitionImageLayout(cmd, ssaoRawImage.image,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        {
            VkExtent2D halfExtent{halfW, halfH};
            VkRenderingAttachmentInfo aoAtt{};
            aoAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            aoAtt.imageView = ssaoRawImage.imageView;
            aoAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            aoAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            aoAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            aoAtt.clearValue.color = {{1.0f, 1.0f, 1.0f, 1.0f}};

            VkRenderingInfo aoRenderInfo{};
            aoRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            aoRenderInfo.renderArea = {{0, 0}, halfExtent};
            aoRenderInfo.layerCount = 1;
            aoRenderInfo.colorAttachmentCount = 1;
            aoRenderInfo.pColorAttachments = &aoAtt;

            VkViewport halfVp{};
            halfVp.width = static_cast<float>(halfW);
            halfVp.height = static_cast<float>(halfH);
            halfVp.minDepth = 0.0f;
            halfVp.maxDepth = 1.0f;
            VkRect2D halfSc{{0, 0}, halfExtent};

            vkCmdBeginRendering(cmd, &aoRenderInfo);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoPipeline);
            vkCmdSetViewport(cmd, 0, 1, &halfVp);
            vkCmdSetScissor(cmd, 0, 1, &halfSc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    ssaoPipelineLayout, 0, 1,
                                    &ssaoDescriptorSets[currentFrame], 0, nullptr);
            struct SsaoPush
            {
                float radius;
                float intensity;
                float bias;
                int _pad;
            };
            SsaoPush sp{ssaoRadius, ssaoIntensity, ssaoBias, 0};
            vkCmdPushConstants(cmd, ssaoPipelineLayout,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(sp), &sp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ++curDrawCallStats.lighting;
            vkCmdEndRendering(cmd);
        }
        VulkanUtils::transitionImageLayout(cmd, ssaoRawImage.image,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // -- (b) Bilateral Blur X @ half-res -> ssaoBlurXImage --
        VulkanUtils::transitionImageLayout(cmd, ssaoBlurXImage.image,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        {
            VkExtent2D halfExtent{halfW, halfH};
            VkRenderingAttachmentInfo att{};
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = ssaoBlurXImage.imageView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = {{1.0f, 1.0f, 1.0f, 1.0f}};
            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea = {{0, 0}, halfExtent};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;

            VkViewport halfVp{};
            halfVp.width = static_cast<float>(halfW);
            halfVp.height = static_cast<float>(halfH);
            halfVp.minDepth = 0.0f;
            halfVp.maxDepth = 1.0f;
            VkRect2D halfSc{{0, 0}, halfExtent};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoBlurPipeline);
            vkCmdSetViewport(cmd, 0, 1, &halfVp);
            vkCmdSetScissor(cmd, 0, 1, &halfSc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    ssaoBlurPipelineLayout, 0, 1,
                                    &ssaoBlurXDescriptorSets[currentFrame], 0, nullptr);
            struct BlurPush
            {
                int dir;
                int p0, p1, p2;
            };
            BlurPush bp{0, 0, 0, 0};
            vkCmdPushConstants(cmd, ssaoBlurPipelineLayout,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(bp), &bp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ++curDrawCallStats.lighting;
            vkCmdEndRendering(cmd);
        }
        VulkanUtils::transitionImageLayout(cmd, ssaoBlurXImage.image,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // -- (c) Bilateral Blur Y + Upsample @ FULL-res -> ssaoImage --
        VulkanUtils::transitionImageLayout(cmd, ssaoImage.image,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        {
            VkRenderingAttachmentInfo att{};
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = ssaoImage.imageView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = {{1.0f, 1.0f, 1.0f, 1.0f}};
            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea = {{0, 0}, sceneExtent};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoBlurPipeline);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    ssaoBlurPipelineLayout, 0, 1,
                                    &ssaoBlurYDescriptorSets[currentFrame], 0, nullptr);
            struct BlurPush
            {
                int dir;
                int p0, p1, p2;
            };
            BlurPush bp{1, 0, 0, 0};
            vkCmdPushConstants(cmd, ssaoBlurPipelineLayout,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(bp), &bp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ++curDrawCallStats.lighting;
            vkCmdEndRendering(cmd);
        }
        VulkanUtils::transitionImageLayout(cmd, ssaoImage.image,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    gpuProfiler.endPass(cmd, "SSAO");

    // === PASS 2: Lighting Pass -> HDR FP16 RT ===
    gpuProfiler.beginPass(cmd, "Lighting");
    auto &hdrImage = editorUI.getHdrImage();
    auto &offscreenImage = editorUI.getOffscreenImage();
    VulkanUtils::transitionImageLayout(cmd, hdrImage.image,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = hdrImage.imageView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingInfo lightRenderInfo{};
    lightRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    lightRenderInfo.renderArea = {{0, 0}, sceneExtent};
    lightRenderInfo.layerCount = 1;
    lightRenderInfo.colorAttachmentCount = 1;
    lightRenderInfo.pColorAttachments = &colorAtt;

    vkCmdBeginRendering(cmd, &lightRenderInfo);
    gpuProfiler.beginPipelineStats(cmd, "Lighting");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeline);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            lightPipelineLayout, 0, 1, &lightDescriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    ++curDrawCallStats.lighting; // Draw call stats bucket

    // Light volume - additive-blends point/spot contributions
    // inside the same RenderingInfo
    lastLightVolumeStats = {0, 0};
    if (useLightVolume && (!lights.empty() || !spotLights.empty()))
    {
        const float aspect = static_cast<float>(viewportExtent.width) /
                             static_cast<float>(viewportExtent.height);
        glm::mat4 vp = camera.getProjectionMatrix(aspect) * camera.getViewMatrix();

        VkBuffer vbufs[] = {lightVolumeVB.buffer};
        VkDeviceSize offsets[] = {0};

        // --- Point Lights ---
        if (!lights.empty())
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightVolumePointPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    lightVolumePipelineLayout, 0, 1,
                                    &lightDescriptorSets[currentFrame], 0, nullptr);
            vkCmdPushConstants(cmd, lightVolumePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &vp);
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, lightVolumeIB.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, lightVolumeIndexCount,
                             static_cast<uint32_t>(lights.size()), 0, 0, 0);
            lastLightVolumeStats.pointDraws = static_cast<uint32_t>(lights.size());
            ++curDrawCallStats.lightVolume; // Draw call stats bucket (1 instanced draw counts as 1)
        }

        // --- Spot Lights ---
        if (!spotLights.empty())
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightVolumeSpotPipeline);
            // After the pipeline switch the descriptor set / push constants are
            // on the same layout and need no rebind; but to be safe against
            // future reordering, rebind the descriptor set explicitly once
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    lightVolumePipelineLayout, 0, 1,
                                    &lightDescriptorSets[currentFrame], 0, nullptr);
            vkCmdPushConstants(cmd, lightVolumePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &vp);
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, lightVolumeIB.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, lightVolumeIndexCount,
                             static_cast<uint32_t>(spotLights.size()), 0, 0, 0);
            lastLightVolumeStats.spotDraws = static_cast<uint32_t>(spotLights.size());
            ++curDrawCallStats.lightVolume; // Draw call stats bucket
        }
    }

    gpuProfiler.endPipelineStats(cmd);
    vkCmdEndRendering(cmd);

    // hdrImage: COLOR_ATTACHMENT -> SHADER_READ_ONLY for the Composite Pass
    VulkanUtils::transitionImageLayout(cmd, hdrImage.image,
                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gpuProfiler.endPass(cmd, "Lighting");

    // === PASS 2.4: Bloom Pass ===
    //   hdrImage -> bloomMips[0] (threshold)
    //   bloomMips[i] -> bloomMips[i+1] (downsample, i = 0..5)
    //   bloomMips[i+1] -> bloomMips[i] (upsample, ADD blend, i = 5..0)
    // Final mip 0 holds the blurred bright pass and is sampled by Composite.
    gpuProfiler.beginPass(cmd, "Bloom");
    {
        // Helper: full-res / per-mip extents.
        auto mipExtent = [&](uint32_t m) -> VkExtent2D
        {
            return {std::max(1u, viewportExtent.width >> m),
                    std::max(1u, viewportExtent.height >> m)};
        };

        // Helper: render fullscreen triangle into target view at given extent.
        auto runFullscreen = [&](VkImageView targetView, VkExtent2D ext,
                                 VkPipeline pipe, VkPipelineLayout pLayout,
                                 VkDescriptorSet ds, const void *push, uint32_t pushSize)
        {
            VkRenderingAttachmentInfo att{};
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = targetView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea = {{0, 0}, ext};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;

            VkViewport vp{};
            vp.width = static_cast<float>(ext.width);
            vp.height = static_cast<float>(ext.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            VkRect2D sc{{0, 0}, ext};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pLayout, 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, pushSize, push);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ++curDrawCallStats.lighting; // count alongside lighting (post-process draw)
            vkCmdEndRendering(cmd);
        };

        // ---- (a) Threshold pass: hdrImage -> bloomMips[0] ----
        VulkanUtils::transitionImageLayout(cmd, bloomMips[0].image,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        struct ThresholdPush
        {
            float threshold;
            float softKnee;
            float p0;
            float p1;
        };
        ThresholdPush tPush{bloomThreshold, bloomSoftKnee, 0.0f, 0.0f};
        runFullscreen(bloomMips[0].imageView, mipExtent(0),
                      bloomThresholdPipeline, bloomThresholdPLayout,
                      bloomThresholdSets[currentFrame],
                      &tPush, sizeof(tPush));
        VulkanUtils::transitionImageLayout(cmd, bloomMips[0].image,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // ---- (b) Downsample chain: mip[i] -> mip[i+1], i = 0..5 ----
        struct ChainPush
        {
            float invX;
            float invY;
            float scatter;
            float p;
        };
        for (uint32_t i = 0; i < BLOOM_MIP_COUNT - 1; ++i)
        {
            VkExtent2D srcExt = mipExtent(i);
            VkExtent2D dstExt = mipExtent(i + 1);

            VulkanUtils::transitionImageLayout(cmd, bloomMips[i + 1].image,
                                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            ChainPush cp{
                1.0f / static_cast<float>(srcExt.width),
                1.0f / static_cast<float>(srcExt.height),
                1.0f, // scatter not used by downsample, but kept for layout uniformity
                0.0f,
            };
            runFullscreen(bloomMips[i + 1].imageView, dstExt,
                          bloomDownsamplePipeline, bloomChainPLayout,
                          bloomDownsampleSets[i][currentFrame],
                          &cp, sizeof(cp));

            VulkanUtils::transitionImageLayout(cmd, bloomMips[i + 1].image,
                                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // ---- (c) Upsample chain: mip[i+1] -> mip[i] (ADD), i = 5..0 ----
        // Each upsample reads the smaller mip and ADD-blends onto the next-finer
        // mip. The destination must transition back to COLOR_ATTACHMENT before
        // the pass and back to SHADER_READ_ONLY afterwards (so the next-finer
        // upsample / Composite sees it ready).
        for (int32_t i = static_cast<int32_t>(BLOOM_MIP_COUNT) - 2; i >= 0; --i)
        {
            VkExtent2D srcExt = mipExtent(i + 1);
            VkExtent2D dstExt = mipExtent(i);

            VulkanUtils::transitionImageLayout(cmd, bloomMips[i].image,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            ChainPush cp{
                1.0f / static_cast<float>(srcExt.width),
                1.0f / static_cast<float>(srcExt.height),
                bloomScatter,
                0.0f,
            };
            // NOTE: upsample uses LOAD_OP_LOAD so the additive blend can
            // accumulate onto the existing mip i contents. We override the
            // attachment loadOp here because runFullscreen() defaults to CLEAR.
            VkRenderingAttachmentInfo att{};
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = bloomMips[i].imageView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea = {{0, 0}, dstExt};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;

            VkViewport vp{};
            vp.width = static_cast<float>(dstExt.width);
            vp.height = static_cast<float>(dstExt.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            VkRect2D sc{{0, 0}, dstExt};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomUpsamplePipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    bloomChainPLayout, 0, 1,
                                    &bloomUpsampleSets[i][currentFrame], 0, nullptr);
            vkCmdPushConstants(cmd, bloomChainPLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(cp), &cp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            ++curDrawCallStats.lighting;
            vkCmdEndRendering(cmd);

            VulkanUtils::transitionImageLayout(cmd, bloomMips[i].image,
                                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    gpuProfiler.endPass(cmd, "Bloom");

    // === PASS 2.5: Composite Pass ===
    // hdrImage (FP16 linear) -> tone map + exposure -> ldrPreFxaaImage (sRGB LDR)
    auto &ldrPreImg = editorUI.getLdrPreFxaaImage();
    gpuProfiler.beginPass(cmd, "Composite");
    VulkanUtils::transitionImageLayout(cmd, ldrPreImg.image,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    {
        VkRenderingAttachmentInfo compAtt{};
        compAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        compAtt.imageView = ldrPreImg.imageView;
        compAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        compAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        compAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        compAtt.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderingInfo compRenderInfo{};
        compRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        compRenderInfo.renderArea = {{0, 0}, sceneExtent};
        compRenderInfo.layerCount = 1;
        compRenderInfo.colorAttachmentCount = 1;
        compRenderInfo.pColorAttachments = &compAtt;

        vkCmdBeginRendering(cmd, &compRenderInfo);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                compositePipelineLayout, 0, 1,
                                &compositeDescriptorSets[currentFrame], 0, nullptr);
        // Push constants: { float exposure; int tonemapMode; float bloomIntensity; int _pad1; }
        struct CompositePush
        {
            float exposure;
            int tonemapMode;
            float bloomIntensity;
            int p1;
        };
        CompositePush push{exposure, tonemapMode, bloomIntensity, 0};
        vkCmdPushConstants(cmd, compositePipelineLayout,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        ++curDrawCallStats.lighting; // count alongside lighting (full-screen post pass)
        vkCmdEndRendering(cmd);
    }
    VulkanUtils::transitionImageLayout(cmd, ldrPreImg.image,
                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gpuProfiler.endPass(cmd, "Composite");

    // === PASS 2.6: FXAA Pass ===
    // ldrPreFxaaImage (sRGB LDR) -> optional FXAA -> offscreenImage (final, ImGui-bound)
    gpuProfiler.beginPass(cmd, "FXAA");
    VulkanUtils::transitionImageLayout(cmd, offscreenImage.image,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    {
        VkRenderingAttachmentInfo fxAtt{};
        fxAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        fxAtt.imageView = offscreenImage.imageView;
        fxAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        fxAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        fxAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        fxAtt.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderingInfo fxRenderInfo{};
        fxRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        fxRenderInfo.renderArea = {{0, 0}, sceneExtent};
        fxRenderInfo.layerCount = 1;
        fxRenderInfo.colorAttachmentCount = 1;
        fxRenderInfo.pColorAttachments = &fxAtt;

        vkCmdBeginRendering(cmd, &fxRenderInfo);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaaPipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                fxaaPipelineLayout, 0, 1,
                                &fxaaDescriptorSets[currentFrame], 0, nullptr);
        struct FxaaPush
        {
            float invX;
            float invY;
            int enable;
            int p0;
            float edgeThreshold;
            float subpixelBlend;
            int debugShowEdges;
            int p1;
        };
        // Map fxaaQuality [0,1] -> edgeThreshold [0.166, 0.063]
        // (PRESET 12 console -> PRESET 39 extreme).
        const float kThrLow = 0.063f;  // extreme quality
        const float kThrHigh = 0.166f; // console quality
        float threshold = kThrHigh + (kThrLow - kThrHigh) * fxaaQuality;
        FxaaPush fxPush{
            1.0f / static_cast<float>(sceneExtent.width),
            1.0f / static_cast<float>(sceneExtent.height),
            useFxaa ? 1 : 0,
            0,
            threshold,
            fxaaSubpixel,
            fxaaDebugShowEdges ? 1 : 0,
            0,
        };
        vkCmdPushConstants(cmd, fxaaPipelineLayout,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(fxPush), &fxPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        ++curDrawCallStats.lighting; // post-process draw
        vkCmdEndRendering(cmd);
    }
    VulkanUtils::transitionImageLayout(cmd, offscreenImage.image,
                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gpuProfiler.endPass(cmd, "FXAA");

    // === PASS 3: ImGui Pass -> swapchain ===
    gpuProfiler.beginPass(cmd, "ImGui");
    VulkanUtils::transitionImageLayout(cmd, swapImg,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    {
        VkRenderingAttachmentInfo imguiColorAtt{};
        imguiColorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        imguiColorAtt.imageView = swapView;
        imguiColorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imguiColorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        imguiColorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        imguiColorAtt.clearValue.color = {{0.15f, 0.15f, 0.15f, 1.0f}};

        VkRenderingInfo imguiRenderInfo{};
        imguiRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        imguiRenderInfo.renderArea = {{0, 0}, swapExtent};
        imguiRenderInfo.layerCount = 1;
        imguiRenderInfo.colorAttachmentCount = 1;
        imguiRenderInfo.pColorAttachments = &imguiColorAtt;

        vkCmdBeginRendering(cmd, &imguiRenderInfo);
        // Accumulate this frame's ImGui vkCmdDrawIndexed total
        // (one per ImDrawCmd)
        if (ImDrawData *dd = ImGui::GetDrawData())
        {
            for (int i = 0; i < dd->CmdListsCount; ++i)
                curDrawCallStats.imgui += static_cast<uint32_t>(dd->CmdLists[i]->CmdBuffer.Size);
            ImGui_ImplVulkan_RenderDrawData(dd, cmd);
        }
        vkCmdEndRendering(cmd);
    }

    VulkanUtils::transitionImageLayout(cmd, swapImg,
                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    gpuProfiler.endPass(cmd, "ImGui");

    gpuProfiler.endPass(cmd, "Frame Total");
    gpuProfiler.endFrame(cmd);

    // Commit this frame's draw call stats (gizmo / imgui were
    // accumulated this frame by GizmoRenderer / EditorUI via
    // addGizmoDraws / addImGuiDraws)
    lastDrawCallStats = curDrawCallStats;

    vkEndCommandBuffer(cmd);
}

// ============================================================================
// recreateSwapchainResources()
// ============================================================================
void Renderer::recreateSwapchainResources()
{
    vkDeviceWaitIdle(vulkanContext.getDevice());
    swapchain.recreate(vulkanContext, window);
}

// ============================================================================
// ~Renderer() - out-of-line destructor
// ----------------------------------------------------------------------------
// Declared in the header but defined here so std::unique_ptr<ScriptEngine>'s
// deleter sees the full ScriptEngine type. cleanup() does the real work; the
// destructor only runs the unique_ptr deleters that haven't run yet (which
// is the normal case if the user forgot to call cleanup() before destruction).
// ============================================================================
Renderer::Renderer() = default;
Renderer::~Renderer() = default;

// ============================================================================
// cleanup()
// ============================================================================
void Renderer::cleanup()
{
    VkDevice device = vulkanContext.getDevice();
    vkDeviceWaitIdle(device);

    // Tear down the Python interpreter FIRST. User scripts may
    // hold references (Entity / Component handles) into the renderer; if we
    // freed Vulkan / ECS resources before Python decref'd those handles we'd
    // be at the mercy of GC ordering. Doing it first makes the contract
    // explicit: after this line, no Python code can run for the rest of
    // shutdown.
    if (scriptEngine)
    {
        scriptEngine->shutdown();
        scriptEngine.reset();
    }

    gpuProfiler.cleanup();                       // GPU profiler
    hiZPass.cleanup(device, allocator.getVma()); // Hi-Z pass
    shadowPass.cleanup(device, allocator);       // Directional shadow pass
    spotShadowPass.cleanup(device, allocator);   // Spot shadow pass
    pointShadowPass.cleanup(device, allocator);  // Point shadow pass
    pointShadowPass.cleanup(device, allocator);  // Point shadow pass
    // Composite holds a descriptor referencing hdrImage.imageView,
    // which editorUI.cleanup() will destroy -> tear down composite first.
    cleanupCompositeResources();
    // FXAA holds a descriptor referencing ldrPreFxaaImage.imageView.
    cleanupFxaaResources();
    // SSAO holds GBuffer view refs.
    cleanupSsaoResources();
    // Bloom holds editorUI.hdrImage.imageView ref + own pyramid.
    cleanupBloomResources();
    editorUI.cleanup();

    if (cubeMeshRef)
    {
        allocator.destroyBuffer(cubeMeshRef->vertexBuffer);
        allocator.destroyBuffer(cubeMeshRef->indexBuffer);
        cubeMeshRef.reset();
    }
    if (planeMeshRef)
    {
        allocator.destroyBuffer(planeMeshRef->vertexBuffer);
        allocator.destroyBuffer(planeMeshRef->indexBuffer);
        planeMeshRef.reset();
    }
    for (auto &meshRef : importedMeshes)
    {
        if (meshRef)
        {
            allocator.destroyBuffer(meshRef->vertexBuffer);
            allocator.destroyBuffer(meshRef->indexBuffer);
        }
    }
    importedMeshes.clear();

    gizmoRenderer.cleanup(device, allocator);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vmaUnmapMemory(allocator.getVma(), geomUniformBuffers[i].allocation);
        allocator.destroyBuffer(geomUniformBuffers[i]);
    }
    // Per-frame instance SSBO
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < instanceSSBOs.size() && instanceSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), instanceSSBOs[i].allocation);
            allocator.destroyBuffer(instanceSSBOs[i]);
        }
    }
    instanceSSBOs.clear();
    instanceSSBOsMapped.clear();
    vkDestroyDescriptorPool(device, geomDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, geomDescriptorSetLayout, nullptr);
    vkDestroyPipeline(device, geomPipeline, nullptr);
    vkDestroyPipelineLayout(device, geomPipelineLayout, nullptr);

    // Tex pool / layout / default set are owned by the texture
    // resources subsystem (created once in init(), survives every resize).
    cleanupTextureResources();
    textureManager.cleanup();

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vmaUnmapMemory(allocator.getVma(), lightUniformBuffers[i].allocation);
        allocator.destroyBuffer(lightUniformBuffers[i]);
    }
    // dirShadowUBOs
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < dirShadowUBOs.size() && dirShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), dirShadowUBOs[i].allocation);
            allocator.destroyBuffer(dirShadowUBOs[i]);
        }
    }
    dirShadowUBOs.clear();
    dirShadowUBOsMapped.clear();
    // spotShadowUBOs
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < spotShadowUBOs.size() && spotShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), spotShadowUBOs[i].allocation);
            allocator.destroyBuffer(spotShadowUBOs[i]);
        }
    }
    spotShadowUBOs.clear();
    spotShadowUBOsMapped.clear();
    // pointShadowUBOs
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < pointShadowUBOs.size() && pointShadowUBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), pointShadowUBOs[i].allocation);
            allocator.destroyBuffer(pointShadowUBOs[i]);
        }
    }
    pointShadowUBOs.clear();
    pointShadowUBOsMapped.clear();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (i < lightSSBOs.size() && lightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), lightSSBOs[i].allocation);
            allocator.destroyBuffer(lightSSBOs[i]);
        }
        if (i < dirLightSSBOs.size() && dirLightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), dirLightSSBOs[i].allocation);
            allocator.destroyBuffer(dirLightSSBOs[i]);
        }
        if (i < spotLightSSBOs.size() && spotLightSSBOs[i].buffer)
        {
            vmaUnmapMemory(allocator.getVma(), spotLightSSBOs[i].allocation);
            allocator.destroyBuffer(spotLightSSBOs[i]);
        }
    }
    vkDestroyDescriptorPool(device, lightDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, lightDescriptorSetLayout, nullptr);
    vkDestroyPipeline(device, lightPipeline, nullptr);
    vkDestroyPipelineLayout(device, lightPipelineLayout, nullptr);

    // Light volume resource release
    if (lightVolumePointPipeline)
        vkDestroyPipeline(device, lightVolumePointPipeline, nullptr);
    if (lightVolumeSpotPipeline)
        vkDestroyPipeline(device, lightVolumeSpotPipeline, nullptr);
    if (lightVolumePipelineLayout)
        vkDestroyPipelineLayout(device, lightVolumePipelineLayout, nullptr);
    if (lightVolumeVB.buffer)
        allocator.destroyBuffer(lightVolumeVB);
    if (lightVolumeIB.buffer)
        allocator.destroyBuffer(lightVolumeIB);

    vkDestroySampler(device, gbufferSampler, nullptr);
    gbuffer.cleanup(device, allocator.getVma());
    DepthUtils::destroyDepthImage(device, allocator.getVma(), depthImage);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    for (auto &sem : renderFinishedSemaphores)
    {
        vkDestroySemaphore(device, sem, nullptr);
    }

    commandManager.cleanup(device);
    allocator.cleanup();
    swapchain.cleanup(device);
    vulkanContext.cleanup();

    std::cout << "[Renderer] Fully cleaned up.\n";
}

// ============================================================================
// shouldClose()
// ============================================================================
bool Renderer::shouldClose() const
{
    if (glfwWindowShouldClose(window))
        return true;

    // Headless benchmark auto-exit
    //   1) frameLimit > 0 and reached -> exit
    //   2) autoExitOnBenchmarkDone and the benchmark returned from non-Idle to
    //      Idle -> exit (that second condition is triggered by the caller after
    //      each frame's tick via a setShouldClose-equivalent; here only
    //      frameLimit is the fallback, with main.cpp watching benchmark state)
    if (frameLimit > 0 && frameCount >= frameLimit)
        return true;

    return false;
}
