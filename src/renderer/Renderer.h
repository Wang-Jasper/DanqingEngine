#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <string>
#include <array>
#include <memory>

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandManager.h"
#include "core/Allocator.h"
#include "utils/VulkanUtils.h"
#include "utils/DepthUtils.h"
#include "renderer/GBuffer.h"
#include "renderer/GizmoRenderer.h"
#include "renderer/GpuProfiler.h"
#include "renderer/CpuProfiler.h"
#include "renderer/BenchmarkRunner.h"
#include "renderer/HiZPass.h"
#include "renderer/RenderBVH.h"
#include "renderer/ShadowPass.h"
#include "renderer/SpotShadowPass.h"
#include "renderer/PointShadowPass.h"
#include "scene/Vertex.h"
#include "scene/Camera.h"
#include "scene/Mesh.h"
#include "scene/Light.h"
#include "ecs/ECSScene.h"
#include "ecs/Systems.h"
#include "utils/TextureManager.h"
#include "physics/PhysicsWorld.h"
#include "ecs/PhysicsSystem.h"
#include "editor/EditorUI.h"

// Forward declaration so we can hold a unique_ptr<ScriptEngine>
// without dragging pybind11/Python headers into every TU that includes
// Renderer.h. The full type is included in Renderer.cpp.
class ScriptEngine;

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

// ============================================================================
// Renderer - two-pass deferred renderer
// ============================================================================
class Renderer
{
    friend class EditorUI; // EditorUI needs access to internal state
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    void init(GLFWwindow *window);
    void drawFrame();
    void cleanup();

    // Declared (not defaulted) here so the compiler-generated
    // destructor sits in Renderer.cpp where ScriptEngine is a
    // complete type. Without this, std::unique_ptr<ScriptEngine>'s deleter
    // fires from any TU that destroys a Renderer (or aborts mid-
    // construction), and ScriptEngine is forward-declared here only.
    // Constructor is also declared (not defaulted) for the same reason: a
    // header-defined default ctor instantiates default_delete<ScriptEngine>
    // in every including TU and MSVC's default_delete carries a
    // can't-delete-incomplete-type static_assert that fires at template
    // instantiation time, not at first call.
    Renderer();
    ~Renderer();

    void onFramebufferResize() { framebufferResized = true; }
    bool shouldClose() const;
    Camera &getCamera() { return camera; }
    const GpuProfiler &getGpuProfiler() const { return gpuProfiler; }
    const CpuProfiler &getCpuProfiler() const { return cpuProfiler; }

    // Expose the embedded Python engine so EditorUI can fire
    // on_start / on_stop at the Play / Stop transitions. May return nullptr
    // if the interpreter failed to initialise (caller must null-check).
    // Returning the raw pointer (not the unique_ptr) keeps this API free of
    // ScriptEngine.h includes for callers that just want a forward decl.
    ScriptEngine *getScriptEngine() { return scriptEngine.get(); }

    // Benchmark runner (exposed to EditorUI Run/Stop/Progress)
    BenchmarkRunner &getBenchmarkRunner() { return benchmarkRunner; }
    const BenchmarkRunner &getBenchmarkRunner() const { return benchmarkRunner; }

    // Frustum culling controls / stats access
    bool isFrustumCullingEnabled() const { return frustumCullingEnabled; }
    void setFrustumCullingEnabled(bool e) { frustumCullingEnabled = e; }
    uint32_t getCullVisibleCount() const { return lastCullStats.visible; }
    uint32_t getCullTotalCount() const { return lastCullStats.total; }

    // BVH culling control (depends on frustumCullingEnabled;
    // culling is disabled entirely when the latter is off)
    bool isBVHCullingEnabled() const { return bvhCullingEnabled; }
    void setBVHCullingEnabled(bool e) { bvhCullingEnabled = e; }
    int getBVHLeafCount() const { return renderBVH.leafCount(); }
    int getBVHNodeCount() const { return renderBVH.nodeCount(); }

    // Light frustum culling controls / stats
    bool isLightFrustumCullEnabled() const { return lightFrustumCullEnabled; }
    void setLightFrustumCullEnabled(bool e) { lightFrustumCullEnabled = e; }
    uint32_t getPointLightVisible() const { return lastLightCullStats.pointVisible; }
    uint32_t getPointLightTotal() const { return lastLightCullStats.pointTotal; }
    uint32_t getSpotLightVisible() const { return lastLightCullStats.spotVisible; }
    uint32_t getSpotLightTotal() const { return lastLightCullStats.spotTotal; }

    // Light volume toggle / stats
    bool isLightVolumeEnabled() const { return useLightVolume; }
    void setLightVolumeEnabled(bool e) { useLightVolume = e; }
    uint32_t getLightVolumePointDraws() const { return lastLightVolumeStats.pointDraws; }
    uint32_t getLightVolumeSpotDraws() const { return lastLightVolumeStats.spotDraws; }

    // HDR exposure (default scene PBR /PI diffuse Lo is
    // ~0.1-0.3, so ~3x exposure is needed to land mid-curve on Reinhard; the
    // user can adjust it in the Overlay)
    float getExposure() const { return exposure; }
    void setExposure(float e) { exposure = e; }

    // Tone mapping curve selector for the Composite Pass:
    //   0 = Linear (clamp [0,1])
    //   1 = Reinhard  (default)
    //   2 = ACES Filmic (Krzysztof Narkowicz approximation)
    int getTonemapMode() const { return tonemapMode; }
    void setTonemapMode(int m) { tonemapMode = (m < 0 ? 0 : (m > 2 ? 2 : m)); }

    // FXAA toggle. Default OFF; enable from
    // Render Settings to layer cheap edge AA on top of (or instead of) SSAA.
    bool getFxaaEnabled() const { return useFxaa; }
    void setFxaaEnabled(bool e) { useFxaa = e; }

    // Super-Sampling Anti-Aliasing (SSAA) factor. The full
    // render pipeline runs at `viewportExtent`, which EditorUI now sizes as
    // (panel_pixels x renderScale). The ImGui Image widget keeps displaying at
    // panel_pixels, so the GPU's bilinear sampler on the offscreen image
    // performs the box-filter down-sample for free. Allowed values:
    //   1.0 = OFF (default)
    //   1.5 = 2.25x pixels
    //   2.0 = 4x pixels
    //   4.0 = 16x pixels (very expensive)
    // Anything else snaps to the nearest legal value.
    float getRenderScale() const { return renderScale; }
    void setRenderScale(float s)
    {
        // snap to nearest legal step so combobox state can never desync
        const float legal[] = {1.0f, 1.5f, 2.0f, 4.0f};
        float best = 1.0f;
        float bestErr = 1e9f;
        for (float v : legal)
        {
            float err = (v > s ? v - s : s - v);
            if (err < bestErr)
            {
                bestErr = err;
                best = v;
            }
        }
        renderScale = best;
    }

    // FXAA quality slider in [0, 1].
    //   0   -> PRESET 12 ("console", edgeThreshold = 0.166) - cheap, subtle.
    //   1   -> PRESET 39 ("extreme quality", edgeThreshold = 0.063) - strongest.
    // Maps linearly between the two so users can tune the AA aggressiveness
    // without recompiling shaders. Default 0.75 lands close to PRESET 29.
    float getFxaaQuality() const { return fxaaQuality; }
    void setFxaaQuality(float q) { fxaaQuality = q < 0.0f ? 0.0f : (q > 1.0f ? 1.0f : q); }

    // FXAA sub-pixel blend factor in [0, 1].
    //   0 = detected edges keep their raw color (visible only via early-out).
    //   1 = full Lottes smoothing.
    // Lottes' default is 0.75; lower if you find FXAA blurs text or fine
    // detail too aggressively.
    float getFxaaSubpixel() const { return fxaaSubpixel; }
    void setFxaaSubpixel(float s) { fxaaSubpixel = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s); }

    // FXAA edge-debug toggle. When ON, every pixel that the
    // shader classifies as an "edge" (i.e. survives the early-out and would
    // otherwise be smoothed) is painted solid red. Lets you visually confirm
    // exactly where FXAA is doing work - handy when the smoothed output is
    // visually too subtle to distinguish from the raw input.
    bool getFxaaDebugShowEdges() const { return fxaaDebugShowEdges; }
    void setFxaaDebugShowEdges(bool d) { fxaaDebugShowEdges = d; }

    // SSAO parameters. Setting intensity = 0 effectively
    // disables AO without recompiling pipelines.
    float getSsaoRadius() const { return ssaoRadius; }
    float getSsaoIntensity() const { return ssaoIntensity; }
    float getSsaoBias() const { return ssaoBias; }
    void setSsaoRadius(float r) { ssaoRadius = r < 0.0f ? 0.0f : r; }
    void setSsaoIntensity(float i) { ssaoIntensity = i < 0.0f ? 0.0f : i; }
    void setSsaoBias(float b) { ssaoBias = b; }

    // Bloom parameters. Setting intensity = 0 effectively
    // disables bloom (composite multiplies bloom by 0). Threshold is in
    // linear HDR luminance units; softKnee smooths the threshold transition.
    float getBloomIntensity() const { return bloomIntensity; }
    float getBloomThreshold() const { return bloomThreshold; }
    float getBloomSoftKnee() const { return bloomSoftKnee; }
    float getBloomScatter() const { return bloomScatter; }
    void setBloomIntensity(float i) { bloomIntensity = i < 0.0f ? 0.0f : i; }
    void setBloomThreshold(float t) { bloomThreshold = t < 0.0f ? 0.0f : t; }
    void setBloomSoftKnee(float k) { bloomSoftKnee = k < 0.0f ? 0.0f : (k > 1.0f ? 1.0f : k); }
    void setBloomScatter(float s) { bloomScatter = s < 0.1f ? 0.1f : (s > 2.0f ? 2.0f : s); }

    // Directional shadow parameters.
    //   shadowDistance      : half-side of the orthographic light frustum (world units).
    //   shadowMapResolution : NxN depth map size; recreated when changed.
    //   shadowsEnabled      : global toggle, independent of per-light castShadows flag.
    float getShadowDistance() const { return shadowDistance; }
    int getShadowMapResolution() const { return shadowMapResolution; }
    bool getShadowsEnabled() const { return shadowsEnabled; }
    void setShadowDistance(float d) { shadowDistance = d < 1.0f ? 1.0f : d; }
    void setShadowMapResolution(int r);
    void setShadowsEnabled(bool e) { shadowsEnabled = e; }

    // Frame rate control (Unity perf-comparison feature).
    // VSync control requires rebuilding the swapchain (FIFO <-> MAILBOX/IMMEDIATE
    // present mode); the minimal implementation here only sleeps to a target FPS
    // on the CPU. VSync is deferred to dynamic resolution, together
    // with swapchain rebuilding.
    int getTargetFps() const { return targetFps; }
    void setTargetFps(int fps) { targetFps = std::max(0, fps); }

    // Headless / auto-exit controls (for CLI benchmarks)
    // ----------------------------------------------------------------
    // setFrameLimit(N)  : shouldClose() returns true after N frames
    // setAutoExitOnBenchmarkDone(true): closes the window when the benchmark
    // returns to Idle
    // Both default OFF (interactive mode is unaffected)
    void setFrameLimit(uint64_t n) { frameLimit = n; }
    uint64_t getFrameLimit() const { return frameLimit; }
    uint64_t getFrameCount() const { return frameCount; }
    void setAutoExitOnBenchmarkDone(bool e) { autoExitOnBenchmarkDone = e; }
    bool isAutoExitOnBenchmarkDone() const { return autoExitOnBenchmarkDone; }

    // Loads a benchmark preset by name and starts the benchmark runner.
    // Returns false if the preset name is invalid; the scene / runner are left
    // untouched. Valid names: Empty / Stress100 / Stress1000 / LightStress /
    // TextureStress (case-sensitive)
    bool startBenchmarkByName(const std::string &presetName);

    // Load the TextureStress benchmark preset (programmatic 4-color
    // texture preset). Exposed separately from startBenchmarkByName() so the
    // editor menu can load the scene without immediately kicking off a 15s
    // benchmark run. CLI's --benchmark TextureStress goes through
    // startBenchmarkByName(), which calls this and then start()s the runner.
    void loadTextureStressScene();

    // Allocate a texture descriptor set bound to (sampler, view)
    // of the given texture. Used by SceneSetup::loadBenchmarkScene to create
    // textured cubes for the TextureStress preset, but exposed publicly so any
    // caller (Inspector "Load Material", future asset import, etc.) can wire
    // a TextureResource into a MaterialComponent::textureDescriptorSet.
    //
    // Returns VK_NULL_HANDLE on allocation failure (e.g. pool exhausted; pool
    // size is currently 64 sets, see createGeometryPassResources()).
    VkDescriptorSet allocateTextureDescriptorSet(const TextureResource &tex);

    // Free all descriptor sets previously returned by
    // allocateTextureDescriptorSet(). Called by SceneSetup before re-loading a
    // textured benchmark preset to avoid leaking sets across reloads.
    void freeAllAllocatedTextureDescriptorSets();

    // Draw call counters (Unity perf-comparison feature)
    // ----------------------------------------------------------------
    // Reset every frame; accumulated by the vkCmdDraw* calls inside each pass.
    // The Overlay shows per-category totals. GizmoRenderer reports via
    // addGizmoDraws(n) (keeping GizmoRenderer free of Renderer
    // references); EditorUI back-fills via ImGui::GetDrawData() after the ImGui
    // pass (calls addImGuiDraws from EditorUI.cpp).
    struct DrawCallStats
    {
        uint32_t geometry = 0;    // PASS 1: Geometry (1 drawIndexed per mesh)
        uint32_t lightVolume = 0; // PASS 2: Light Volume (1 instanced draw per point/spot)
        uint32_t lighting = 0;    // PASS 2: fullscreen lighting pass (vkCmdDraw 3)
        uint32_t gizmo = 0;       // Gizmo / collider debug lines
        uint32_t imgui = 0;       // cmdLists submitted by the ImGui DrawList
        uint32_t total() const { return geometry + lightVolume + lighting + gizmo + imgui; }
    };
    const DrawCallStats &getDrawCallStats() const { return lastDrawCallStats; }
    void addGizmoDraws(uint32_t n) { curDrawCallStats.gizmo += n; }
    void addImGuiDraws(uint32_t n) { curDrawCallStats.imgui += n; }

    void setDebugMode(int mode) { debugMode = mode; }
    void setCursorCaptured(bool captured) { cursorCaptured = captured; }
    bool isCursorCaptured() const { return cursorCaptured; }

    // Public interface for EditorUI
    void doImportModel(const std::string &filepath);
    void doSyncLightsToGPU();

    // Viewport resize split into fast / heavy paths and
    // deferred to the start of the next drawFrame. EditorUI no longer calls any
    // resize function directly; requestViewportResize() just sets a pending
    // flag, and drawFrame() runs the fast-path rebuild after vkWaitForFences -
    // by then in-flight frames are retired, so vkDeviceWaitIdle is unnecessary.
    //
    //   - onViewportResize()      : heavy path (old behavior), only for cases
    //                               that really change layout/format, e.g.
    //                               setShadowResolution.
    //   - onViewportResize_RT()   : fast path, rebuilds only size-dependent RTs
    //                               and the descriptors referencing them, <10ms.
    //   - requestViewportResize() : called by EditorUI, sets the pending flag.
    void onViewportResize();
    void onViewportResize_RT();
    void requestViewportResize(VkExtent2D newExtent)
    {
        pendingViewportExtent = newExtent;
        viewportResizePending = true;
    }

private:
    void createGBufferResources();
    void createDepthResources();
    void createSampler();
    void createGeometryPassResources();
    void createLightingPassResources();

    // Extracts the writes of the 14 lighting descriptor
    // bindings so onViewportResize_RT() can re-run them after rebuilding
    // GBuffer / HiZ without destroying pool / pipeline / layout. Buffer-type
    // bindings (UBO/SSBO/shadow UBO) are refreshed too (cheap re-write).
    void updateLightingDescriptorSets();

    // Texture resources (texDescriptorSetLayout / texDescriptorPool /
    // defaultTexDescriptorSet) are created ONCE during init() and survive every
    // onViewportResize() rebuild. This is required so MaterialComponent::
    // textureDescriptorSet handles handed out by allocateTextureDescriptorSet()
    // remain valid across resizes (the geometry pipeline layout still references
    // texDescriptorSetLayout, but rebuilding the pool would invalidate every
    // outstanding set).
    void createTextureResources();
    void cleanupTextureResources();
    void createLightVolumeResources();

    // Composite Pass: tone-maps the FP16 HDR lighting RT into
    // the sRGB LDR RT bound to the ImGui viewport. createCompositeResources()
    // is called from init() and from onViewportResize() (descriptor must be
    // re-written when hdrImage view changes); cleanup() / onViewportResize()
    // call cleanupCompositeResources() before re-creating.
    void createCompositeResources();
    void cleanupCompositeResources();

    // FXAA Pass: samples editorUI.ldrPreFxaaImage and writes
    // editorUI.offscreenImage. Lifecycle mirrors composite (rebuilt on resize).
    void createFxaaResources();
    void cleanupFxaaResources();

    // SSAO Pass: samples GBuffer Position+Normal, writes ssaoImage.
    // Lifecycle mirrors composite (the descriptor set references GBuffer views
    // that get recreated on viewport resize).
    void createSsaoResources();
    void cleanupSsaoResources();

    // Bloom Pass: 7-mip pyramid (mip0 full-res, mip6 = 1/64).
    // Lifecycle mirrors composite/ssao - rebuilt on every viewport resize
    // because every mip's image view depends on viewportExtent.
    void createBloomResources();
    void cleanupBloomResources();

    void createSyncObjects();

    void updateUniformBuffer(uint32_t frameIndex);
    void updateLightBuffers(uint32_t frameIndex);
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void recreateSwapchainResources();

    // --- subsystem instances ---
    GLFWwindow *window = nullptr;
    VulkanContext vulkanContext;
    Swapchain swapchain;
    CommandManager commandManager;
    Allocator allocator;
    Camera camera;
    ECSScene scene;
    float lastFrameTime = 0.0f;

    // --- submodules ---
    EditorUI editorUI;
    GizmoRenderer gizmoRenderer;
    GpuProfiler gpuProfiler;         // GPU performance analysis
    CpuProfiler cpuProfiler;         // CPU performance analysis
    BenchmarkRunner benchmarkRunner; // Benchmark mode
    HiZPass hiZPass;                 // Hi-Z depth pyramid (debug view & future GPU occlusion)
    ShadowPass shadowPass;           // Directional light shadow depth-only pass
    SpotShadowPass spotShadowPass;   // Spot light shadows (up to 4 slots)
    PointShadowPass pointShadowPass; // Point light cubemap shadows (up to 4 slots)

    // --- G-Buffer + depth + sampler ---
    GBuffer gbuffer;
    AllocatedImage depthImage{};
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampler gbufferSampler = VK_NULL_HANDLE;

    // === Geometry Pass resources ===
    VkDescriptorSetLayout geomDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout geomPipelineLayout = VK_NULL_HANDLE;
    VkPipeline geomPipeline = VK_NULL_HANDLE;
    VkDescriptorPool geomDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> geomDescriptorSets;
    std::vector<AllocatedBuffer> geomUniformBuffers;
    std::vector<void *> geomUniformBuffersMapped;

    // Per-frame host-mapped instance SSBO (binding 2 of geom set 0).
    // Each visible entity writes one InstanceData entry; identical (mesh, texture)
    // tuples are issued as a single instanced draw call.
    static constexpr uint32_t MAX_INSTANCES = 4096;
    std::vector<AllocatedBuffer> instanceSSBOs;
    std::vector<void *> instanceSSBOsMapped;

    // === Lighting Pass resources ===
    VkDescriptorSetLayout lightDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout lightPipelineLayout = VK_NULL_HANDLE;
    VkPipeline lightPipeline = VK_NULL_HANDLE;
    VkDescriptorPool lightDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> lightDescriptorSets;
    std::vector<AllocatedBuffer> lightUniformBuffers;
    std::vector<void *> lightUniformBuffersMapped;

    // === Composite Pass resources ===
    // Reads HDR FP16 lighting RT (editorUI.hdrImage), writes LDR sRGB
    // (editorUI.ldrPreFxaaImage in the FXAA path, previously editorUI.offscreenImage).
    VkDescriptorSetLayout compositeDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout compositePipelineLayout = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkDescriptorPool compositeDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> compositeDescriptorSets; // per-frame, samples hdrImage
    VkSampler hdrSampler = VK_NULL_HANDLE;
    int tonemapMode = 1; // 0=Linear, 1=Reinhard (default), 2=ACES Filmic

    // === FXAA Pass resources ===
    // Reads editorUI.ldrPreFxaaImage (sRGB output of composite), writes
    // editorUI.offscreenImage (final, ImGui-bound). When `useFxaa=false` the
    // shader takes a passthrough path so the pipeline keeps a stable structure.
    VkDescriptorSetLayout fxaaDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout fxaaPipelineLayout = VK_NULL_HANDLE;
    VkPipeline fxaaPipeline = VK_NULL_HANDLE;
    VkDescriptorPool fxaaDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> fxaaDescriptorSets; // per-frame, samples ldrPreFxaaImage
    VkSampler ldrSampler = VK_NULL_HANDLE;
    bool useFxaa = false; // Default OFF, enabled manually from Render Settings
    // Quality slider state. Defaults pushed to max so the
    // FXAA effect is unambiguous on launch; users can dial them down if the
    // smoothing is too aggressive on text / fine geometry.
    float fxaaQuality = 1.0f;        // 0 = console (0.166), 1 = extreme (0.063)
    float fxaaSubpixel = 1.0f;       // 0 = no smoothing, 1 = full Lottes blend
    bool fxaaDebugShowEdges = false; // diagnostic mode - paint edges red

    // === SSAO Pass resources ===
    // Pipeline: GBuffer (Position + Normal) -> ssaoRawImage (half-res, raw)
    //        -> blur X       -> ssaoBlurXImage (half-res)
    //        -> blur Y + 2x upsample -> ssaoImage (FULL-res, final, sampled by composite)
    //
    // The Composite Pass multiplies `ssaoImage` onto the lit HDR. SSAO always
    // runs; setting `ssaoIntensity = 0` makes the output a constant 1.0 so we
    // don't need branching pipeline assembly.
    AllocatedImage ssaoRawImage{};   // R8_UNORM, half-res - raw SSAO
    AllocatedImage ssaoBlurXImage{}; // R8_UNORM, half-res - after horizontal blur
    AllocatedImage ssaoImage{};      // R8_UNORM, full-res - after vertical blur + upsample

    // Raw SSAO pipeline (samples GBuffer Position+Normal + UBO with view/proj).
    VkDescriptorSetLayout ssaoDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline = VK_NULL_HANDLE;
    VkDescriptorPool ssaoDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> ssaoDescriptorSets; // per-frame: gPos + gNorm + UBO
    std::vector<AllocatedBuffer> ssaoUniformBuffers;
    std::vector<void *> ssaoUniformBuffersMapped;

    // Bilateral blur pipeline (samples aoIn + gPosition + UBO with view/inv-size).
    // Same pipeline reused for X and Y passes; the only difference is the push
    // constant `dir` and the descriptor set bound (X: ssaoRaw -> ssaoBlurX,
    // Y: ssaoBlurX -> ssaoImage).
    VkDescriptorSetLayout ssaoBlurDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline = VK_NULL_HANDLE;
    VkDescriptorPool ssaoBlurDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> ssaoBlurXDescriptorSets; // per-frame: ssaoRaw + gPos + UBO
    std::vector<VkDescriptorSet> ssaoBlurYDescriptorSets; // per-frame: ssaoBlurX + gPos + UBO
    std::vector<AllocatedBuffer> ssaoBlurUniformBuffers;
    std::vector<void *> ssaoBlurUniformBuffersMapped;

    float ssaoRadius = 0.5f;
    float ssaoIntensity = 1.0f;
    float ssaoBias = 0.025f;

    // === Bloom Pass resources ===
    // 7-mip pyramid in R16G16B16A16_SFLOAT. Mip 0 is full-res; mip 6 is 1/64.
    //   threshold pass : hdrImage  -> bloomMips[0]
    //   downsample     : bloomMips[i]   -> bloomMips[i+1]   (for i = 0..5)
    //   upsample (ADD) : bloomMips[i+1] -> bloomMips[i]     (for i = 5..0)
    // After the upsample chain, bloomMips[0] holds the final blurred bright
    // pass which the composite samples and additively blends into the lit HDR.
    static constexpr uint32_t BLOOM_MIP_COUNT = 7;
    AllocatedImage bloomMips[BLOOM_MIP_COUNT]{}; // R16G16B16A16_SFLOAT, each mip's own image+view

    // Threshold pipeline (samples editorUI.hdrImage, writes bloomMips[0])
    VkDescriptorSetLayout bloomThresholdDSLayout = VK_NULL_HANDLE;
    VkPipelineLayout bloomThresholdPLayout = VK_NULL_HANDLE;
    VkPipeline bloomThresholdPipeline = VK_NULL_HANDLE;

    // Downsample / upsample share descriptor SET LAYOUT (single sampled
    // image at binding 0) and PIPELINE LAYOUT (16-byte push constant with
    // srcTexel + scatter), but DIFFERENT pipelines (different blend state:
    // downsample writes opaque; upsample adds).
    VkDescriptorSetLayout bloomChainDSLayout = VK_NULL_HANDLE;
    VkPipelineLayout bloomChainPLayout = VK_NULL_HANDLE;
    VkPipeline bloomDownsamplePipeline = VK_NULL_HANDLE;
    VkPipeline bloomUpsamplePipeline = VK_NULL_HANDLE;

    // Descriptor pool + per-frame descriptor sets:
    //   threshold[i]      : binding 0 = editorUI.hdrImage    (per-frame)
    //   downsample[mip][i]: binding 0 = bloomMips[mip]        (mip = 0..5, per-frame)
    //   upsample[mip][i]  : binding 0 = bloomMips[mip+1]      (mip = 0..5, per-frame)
    // total sets = (1 + 6 + 6) x MAX_FRAMES_IN_FLIGHT = 13 x MAX_FRAMES_IN_FLIGHT
    VkDescriptorPool bloomDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> bloomThresholdSets;               // [frame]
    std::vector<std::vector<VkDescriptorSet>> bloomDownsampleSets; // [mip][frame]
    std::vector<std::vector<VkDescriptorSet>> bloomUpsampleSets;   // [mip][frame]
    VkSampler bloomSampler = VK_NULL_HANDLE;                       // linear, clamp-to-edge

    // Bloom UI parameters
    float bloomIntensity = 0.3f; // Unity URP default ~ 0.3
    float bloomThreshold = 1.0f; // luminance threshold for bright-pass
    float bloomSoftKnee = 0.5f;  // 0..1, smoothness of the threshold transition
    float bloomScatter = 0.7f;   // upsample tent kernel size (0.5..1.0 typical)

    // === Shadow Pass resources ===
    // Per-frame UBO carrying mat4 lightViewProj + int enabled + float invMapSize.
    // Bound at lighting set 0 binding 9. shadowPass owns its own per-frame UBO
    // that drives shadow.vert; this one drives lighting.frag's PCF lookup.
    std::vector<AllocatedBuffer> dirShadowUBOs;
    std::vector<void *> dirShadowUBOsMapped;
    bool shadowsEnabled = true;
    float shadowDistance = 25.0f;     // Unity URP default
    int shadowMapResolution = 2048;   // 2K (Unity URP "High")
    bool pendingShadowResize = false; // deferred to onViewportResize-style sync
    // Written each frame by updateUniformBuffer(): true when a first
    // castShadows directional light was found (record actually renders
    // ShadowPass); false skips ShadowPass recording entirely.
    bool currentFrameHasShadowCaster = false;

    // === Spot Shadow Pass resources ===
    // One per-frame UBO holding the 4 slots' spotLightViewProj plus an int4
    // slotMask marking which slots are valid. Bound at lighting set 0 binding 11
    // (strictly aligned with the shader).
    std::vector<AllocatedBuffer> spotShadowUBOs;
    std::vector<void *> spotShadowUBOsMapped;
    int spotShadowMapResolution = 2048; // 2K (same tier as directional)
    // Active slot count this frame (0..MAX_SPOT_SHADOW_SLOTS). record decides
    // how many slots to record; the rest go through transitionSlotToReadOnly to
    // keep the layout readable.
    uint32_t currentFrameSpotShadowSlots = 0;
    // Per-slot info saved after selection (light world pos / dir / VP / cone
    // angle / range), used by record for independent caster frustum culling and
    // UBO writes.
    struct SelectedSpotShadow
    {
        glm::mat4 lightViewProj;
        glm::vec3 lightPos;
        glm::vec3 lightDir;
        float outerAngle;
        float range;
    };
    std::array<SelectedSpotShadow, SpotShadowPass::MAX_SPOT_SHADOW_SLOTS> currentFrameSpotShadows{};

    // === Point Shadow Pass resources ===
    // One per-frame UBO holding 4 slots' lightPosRange (vec4: xyz=worldPos,
    // w=range) + validSlots. Bound at lighting set 0 binding 13 (strictly
    // aligned with the shader).
    std::vector<AllocatedBuffer> pointShadowUBOs;
    std::vector<void *> pointShadowUBOsMapped;
    int pointShadowMapResolution = 1024; // 1K cubemap (6 faces total ~24MB x 4 slots)
    // Active slot count this frame (0..MAX_POINT_SHADOW_SLOTS). record decides
    // how many slots to record; the rest go through transitionSlotToReadOnly to
    // keep the layout readable.
    uint32_t currentFramePointShadowSlots = 0;
    // Per-slot info saved after selection (light world pos / range), used by
    // record for caster distance-sphere culling + the drawFn interface.
    struct SelectedPointShadow
    {
        glm::vec3 lightPos;
        float range;
        // Unity-style per-light shadow bias (already scaled by
        // the base factor; can go straight to vkCmdSetDepthBias / the shader).
        float rasterConstantBias;
        float rasterSlopeBias;
        float fragNormalBiasMul;
        float fragDepthBiasMul;
    };
    std::array<SelectedPointShadow, PointShadowPass::MAX_POINT_SHADOW_SLOTS> currentFramePointShadows{};

    // Light SSBOs are host-mapped per-frame buffers (fixed capacity)
    // Design:
    //   - updateLightBuffers(frameIdx) gathers + frustum-culls + memcpys into
    //     the frame slot every frame
    //   - descriptor sets are bound once in createLightingPassResources, never rebuilt
    //   - exceeding the cap prints a warning and truncates
    static constexpr uint32_t MAX_POINT_LIGHTS = 256;
    static constexpr uint32_t MAX_DIR_LIGHTS = 16;
    static constexpr uint32_t MAX_SPOT_LIGHTS = 256;

    std::vector<AllocatedBuffer> lightSSBOs; // per-frame point light SSBO
    std::vector<void *> lightSSBOsMapped;
    std::vector<AllocatedBuffer> dirLightSSBOs;
    std::vector<void *> dirLightSSBOsMapped;
    std::vector<AllocatedBuffer> spotLightSSBOs;
    std::vector<void *> spotLightSSBOsMapped;

    // CPU-side buffers (memcpy'd to the GPU mapped region after each frame's gather)
    std::vector<PointLight> lights;
    std::vector<DirectionalLight> dirLights;
    std::vector<SpotLight> spotLights;

    int debugMode = 0;
    bool lightFrustumCullEnabled = true; // Light frustum culling toggle
    float exposure = 3.0f;               // HDR exposure
    int targetFps = 0;                   // 0 = uncapped

    // Headless benchmark (all OFF by default; interactive mode unaffected)
    uint64_t frameCount = 0;              // frame counter (incremented at the end of drawFrame)
    uint64_t frameLimit = 0;              // 0 = unlimited; >0 exits after N frames
    bool autoExitOnBenchmarkDone = false; // auto-close the window when the benchmark finishes
    bool prevBenchmarkRunning = false;    // edge detection: true->false triggers auto-exit

    // Draw call stats - curDrawCallStats is zeroed at the start
    // of drawFrame and accumulated by each pass's vkCmdDraw*; copied to
    // lastDrawCallStats at the end for the Overlay (unlike GpuProfiler's
    // last/avg, this is the current frame's live value).
    DrawCallStats curDrawCallStats;
    DrawCallStats lastDrawCallStats;
    struct LightCullStats
    {
        uint32_t pointVisible = 0, pointTotal = 0,
                 spotVisible = 0, spotTotal = 0;
    };
    LightCullStats lastLightCullStats{};
    // Kept for compatibility - legacy code (EditorUI) still
    // writes it, but it is now synced automatically each frame and is a no-op
    // (no longer triggers SSBO rebuilds).
    bool lightsDirty = false;

    // --- mesh resources ---
    std::shared_ptr<MeshReference> cubeMeshRef;
    std::shared_ptr<MeshReference> planeMeshRef;
    std::vector<std::shared_ptr<MeshReference>> importedMeshes;

    // --- texture system ---
    TextureManager textureManager;
    VkDescriptorSetLayout texDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool texDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet defaultTexDescriptorSet = VK_NULL_HANDLE;
    // Tracks every set returned by allocateTextureDescriptorSet().
    // freeAllAllocatedTextureDescriptorSets() walks this list and releases them
    // via vkFreeDescriptorSets (pool was created with FREE_DESCRIPTOR_SET_BIT).
    std::vector<VkDescriptorSet> allocatedTextureDescriptorSets;

    // --- sync objects ---
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    // --- viewport ---
    // viewportExtent is now the "internal render size" =
    // panel_pixels x renderScale. All RTs / passes are created from it; the
    // ImGui Image widget still displays at panel pixels, and the GPU's LINEAR
    // sampler box-filters down on offscreenImage, implementing SSAA for free.
    // renderScale = 1.0 is exactly equivalent to the historical behavior.
    VkExtent2D viewportExtent = {1280, 720};
    float renderScale = 1.0f; // SSAA: {1.0, 1.5, 2.0, 4.0}
    bool cursorCaptured = false;

    // Deferred viewport resize. When EditorUI detects a
    // panel size change mid-ImGui-frame it only writes these two fields;
    // drawFrame() consumes them at the start after vkWaitForFences (calling
    // onViewportResize_RT()), avoiding vkDeviceWaitIdle + rebuilds mid-frame.
    bool viewportResizePending = false;
    VkExtent2D pendingViewportExtent = {0, 0};

    // --- physics ---
    PhysicsWorld physicsWorld;

    // --- CPU frustum culling ---
    bool frustumCullingEnabled = true;
    struct CullStats
    {
        uint32_t visible = 0;
        uint32_t total = 0;
    };
    CullStats lastCullStats{};

    // --- Dynamic AABB tree for faster frustum culling ---
    // drawFrame rebuilds it every frame via renderBVH.rebuildFromScene; the
    // Geometry Pass uses renderBVH.queryFrustum instead of an O(N) entity loop.
    // bvhCullingEnabled toggles between the brute-force and BVH paths for
    // correctness / performance comparison.
    bool bvhCullingEnabled = true;
    RenderBVH renderBVH;

    // --- Light volume (point/spot proxy geometry + additive blend) ---
    // When enabled: the fullscreen lighting pass renders only ambient +
    // directional (the UBO forces lightCount / spotLightCount = 0 to skip the
    // loops); instanced light proxy icospheres are drawn in the same
    // beginRendering with additive blend. When disabled, falls back to the
    // original fullscreen all-lights loop path.
    bool useLightVolume = true;
    AllocatedBuffer lightVolumeVB{};
    AllocatedBuffer lightVolumeIB{};
    uint32_t lightVolumeIndexCount = 0;
    VkPipelineLayout lightVolumePipelineLayout = VK_NULL_HANDLE;
    VkPipeline lightVolumePointPipeline = VK_NULL_HANDLE; // specConstant LIGHT_TYPE = 0
    VkPipeline lightVolumeSpotPipeline = VK_NULL_HANDLE;  // specConstant LIGHT_TYPE = 1
    struct LightVolumeStats
    {
        uint32_t pointDraws = 0;
        uint32_t spotDraws = 0;
    };
    LightVolumeStats lastLightVolumeStats{};

    // Play/Stop mode: snapshots the Transform for restore
    struct TransformSnapshot
    {
        entt::entity entity;
        glm::vec3 position;
        glm::vec3 rotation;
        glm::quat orientation;
        glm::vec3 scale;
    };
    std::vector<TransformSnapshot> transformSnapshots;

    // Embedded Python script engine. Constructed at the very END
    // of init() (so all subsystems are ready when bindings start to surface
    // them) and destroyed at the very START of cleanup() (before any Vulkan /
    // ECS resource is freed - user scripts may still hold references to
    // Components / Entities).
    //
    // Owned by unique_ptr so:
    //   1. Renderer.h stays free of pybind11 / Python headers
    //      (forward declaration above + heap allocation).
    //   2. shutdown is explicit: we reset() before any other cleanup, which
    //      makes the destruction order obvious in Renderer::cleanup().
    std::unique_ptr<ScriptEngine> scriptEngine;
};
