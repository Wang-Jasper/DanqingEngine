// ============================================================================
// ShadowPass.h - directional light shadow pass
// ----------------------------------------------------------------------------
// Records a depth-only render into a 2D depth image before/after the Geometry
// Pass, using the directional light's view-projection. The Lighting Pass
// samples this depth map with sampler2DShadow for PCF shadow attenuation.
//
// Design:
//   1. Single cascade, single directional light (the "Main Light Shadow"
//      feature, matching Unity URP defaults). Multiple cascades are
//      deferred to the high-end feature tier.
//   2. Directional frustum: orthographic shadowDistance x shadowDistance square
//      centered on the camera; near/far = +-shadowDistance.
//   3. Shares the Geometry Pass instance SSBO (avoids duplicate upload); caller
//      passes instance buffer + drawItems to record().
//   4. Not rebuilt when the viewport resizes (unlike Bloom/SSAO, the shadow map
//      resolution is independent); call resize() when shadowMapResolution changes.
//   5. PCF is done inside lighting.frag (4x4, bilinear); this class only produces
//      the shadowMap + sampler + light-space VP matrix.
//
// Not thread-safe: all methods must be called sequentially on the render thread.
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "core/Allocator.h"
#include "utils/DepthUtils.h" // AllocatedImage
#include "scene/Light.h"
#include <cstdint>
#include <functional>
#include <vector>

class ShadowPass
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // Resource lifetime: init() once; resize() when shadowMapResolution changes;
    // cleanup() at engine shutdown.
    void init(VkDevice device, Allocator &allocator, uint32_t shadowMapResolution);
    void resize(VkDevice device, Allocator &allocator, uint32_t newResolution);
    void cleanup(VkDevice device, Allocator &allocator);

    // Per-frame: compute light-space VP given light direction + camera position
    // + shadowDistance, and upload to the per-frame UBO.
    void updateLightSpaceVP(uint32_t frameIndex,
                            const glm::vec3 &lightDirection,
                            const glm::vec3 &cameraPosition,
                            float shadowDistance);

    // ShadowPass owns a per-frame instance SSBO, avoiding
    // the shared Geometry Pass buffer where camera-frustum-only culling made
    // shadows vanish. Returns the current frame's host-mapped pointer; caller
    // fills it, then calls record().
    static constexpr uint32_t MAX_SHADOW_INSTANCES = 4096;
    void *getInstanceSSBOMapped(uint32_t frameIndex) const;
    VkDeviceSize getInstanceSSBOSize() const;

    // Bind the instance SSBO (Geometry Pass owns this buffer; one slot per
    // frame). Must be called after createGeometryPassResources()/onViewportResize
    // because the buffer handles change.
    // Deprecated: kept for API compatibility, but the external SSBO is no longer
    // used; this only sets instanceBuffersBound_ = true as an "available" flag.
    void bindInstanceBuffers(VkDevice device,
                             const std::vector<AllocatedBuffer> &instanceSSBOs,
                             VkDeviceSize bufferSize);

    // Record depth-only render. Caller pushes (instanceOffset) per batch and
    // issues vkCmdDrawIndexed. ShadowPass handles:
    //   * layout transition (UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY)
    //   * begin/end rendering with depth attachment
    //   * pipeline / descriptor binding
    // Caller passes a callback that records the actual draw commands.
    using RecordDrawsFn = std::function<void(VkCommandBuffer cmd,
                                             VkPipelineLayout layout)>;
    void record(VkCommandBuffer cmd, uint32_t frameIndex,
                const RecordDrawsFn &drawFn);

    // Accessors for Lighting Pass descriptor wiring.
    VkImageView shadowMapView() const { return shadowMapView_; }
    VkImage shadowMapImage() const { return shadowMap_.image; }
    VkSampler shadowSampler() const { return shadowSampler_; }
    uint32_t resolution() const { return resolution_; }

    // Bring the shadow map to SHADER_READ_ONLY layout without
    // actually rendering anything (used when no directional caster exists this
    // frame, but the lighting descriptor still points at this image).
    // Idempotent across frames: safe to call every frame in the no-caster path.
    void transitionToReadOnly(VkCommandBuffer cmd, bool firstUse);

    // Light-space VP matrix used in the most recent updateLightSpaceVP() call;
    // the lighting fragment shader receives this through its own UBO so it can
    // project worldPos to shadow-map UV.
    const glm::mat4 &lastLightViewProj() const { return lastLightVP_; }

private:
    void createImage(VkDevice device, Allocator &allocator);
    void createSampler(VkDevice device);
    void createPipeline(VkDevice device);
    void createDescriptors(VkDevice device, Allocator &allocator);

    // --- depth image / view / sampler ---
    uint32_t resolution_ = 0;
    VkFormat format_ = VK_FORMAT_D32_SFLOAT;
    AllocatedImage shadowMap_{};
    VkImageView shadowMapView_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;

    // --- pipeline (depth-only, no fragment shader) ---
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // --- descriptors ---
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_; // [frame]

    // --- per-frame UBO (lightViewProj) ---
    std::vector<AllocatedBuffer> uboBuffers_;
    std::vector<void *> uboMapped_;

    // ShadowPass-owned instance SSBO (no longer shared with Geometry)
    std::vector<AllocatedBuffer> instanceSSBOs_;
    std::vector<void *> instanceSSBOsMapped_;

    glm::mat4 lastLightVP_ = glm::mat4(1.0f);

    // Current shadowMap layout (only used as the transition source for
    // transitionToReadOnly). record() drives the full lifecycle; this tracking
    // mainly serves the no-caster path: first call sees UNDEFINED, later calls skip.
    VkImageLayout currentLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    // --- track instance SSBO (owned by Renderer) for descriptor write ---
    bool instanceBuffersBound_ = false;
};
