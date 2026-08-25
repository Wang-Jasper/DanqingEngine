// ============================================================================
// SpotShadowPass.h - spot light shadow pass
// ----------------------------------------------------------------------------
// Renders up to MAX_SPOT_SHADOW_SLOTS depth-only shadow maps for the top-N
// brightest spot lights (selected by Renderer). Lighting Pass samples
// them through sampler2DShadow array uSpotShadowMap[N] for PCF shadowing.
//
// Design:
//   1. 4 independent 2Kx2K D32_SFLOAT depth images (no texture array); each has
//      its own ImageView, bound via a descriptor array.
//   2. Perspective projection: FOV = outerAngle*2, aspect=1.0, near=0.1, far=range.
//   3. Reuses the directional shadow.vert / shadow.frag (depth-only,
//      lightViewProj injected through the UBO).
//   4. Each slot owns a per-frame UBO (lightViewProj) + instance SSBO
//      (caster list culled against the spot's own frustum).
//   5. Top-N selection is done by Renderer; this class only turns
//      "light data + caster list" into N depth maps. Slot 0..N-1 matches
//      SpotLight.shadowSlot.
//   6. Shares ShadowPass's back-cull + slope-bias (1.5, 2.5) + normal-bias
//      formula for visual consistency; see shadow_mapping.md sec. 7.
//
// Not thread-safe: all methods must be called sequentially on the render thread.
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "core/Allocator.h"
#include "utils/DepthUtils.h"
#include <cstdint>
#include <functional>
#include <vector>
#include <array>

class SpotShadowPass
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    // Must match lighting.frag uSpotShadowMap[4]; resize the shader array when changed.
    static constexpr uint32_t MAX_SPOT_SHADOW_SLOTS = 4;
    // Max casters per slot per frame. 1024 covers typical scenes; excess is truncated.
    static constexpr uint32_t MAX_INSTANCES_PER_SLOT = 1024;

    void init(VkDevice device, Allocator &allocator, uint32_t shadowMapResolution);
    void resize(VkDevice device, Allocator &allocator, uint32_t newResolution);
    void cleanup(VkDevice device, Allocator &allocator);

    // Computes and writes the light-space VP (perspective) for the given slot.
    void updateLightSpaceVP(uint32_t frameIndex, uint32_t slot,
                            const glm::vec3 &lightPos,
                            const glm::vec3 &lightDir,
                            float outerAngleDeg,
                            float range);

    // Returns the instance SSBO host pointer for (slot, frame); caller writes InstanceData[].
    void *getInstanceSSBOMapped(uint32_t frameIndex, uint32_t slot) const;
    VkDeviceSize getInstanceSSBOSize() const;

    // Renders one slot's depth-only pass. drawFn is provided by the caller and must
    // issue vkCmdPushConstants + vkCmdBindVertex/IndexBuffer + vkCmdDrawIndexed.
    using RecordDrawsFn = std::function<void(VkCommandBuffer cmd,
                                             VkPipelineLayout layout)>;
    void recordSlot(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot,
                    const RecordDrawsFn &drawFn);

    // Transitions all slot shadow maps to SHADER_READ_ONLY (on frames with no
    // casters the lighting pass descriptor must still point at readable images).
    // Caller should ensure all 4 slots are SHADER_READ_ONLY before each frame ends.
    void transitionAllToReadOnly(VkCommandBuffer cmd);
    // Single-slot variant, for the "slot k has no casters this frame" path.
    void transitionSlotToReadOnly(VkCommandBuffer cmd, uint32_t slot);

    // Lighting pass descriptor binding interface
    VkImageView shadowMapView(uint32_t slot) const
    {
        return (slot < MAX_SPOT_SHADOW_SLOTS) ? slots_[slot].view : VK_NULL_HANDLE;
    }
    VkSampler shadowSampler() const { return shadowSampler_; }
    uint32_t resolution() const { return resolution_; }

    // Last written lightViewProj (copied into lighting.frag's spotShadow UBO).
    const glm::mat4 &lastLightViewProj(uint32_t slot) const
    {
        static const glm::mat4 kIdent(1.0f);
        return (slot < MAX_SPOT_SHADOW_SLOTS) ? slots_[slot].lastLightVP : kIdent;
    }

private:
    struct SlotResources
    {
        AllocatedImage image{};
        VkImageView view = VK_NULL_HANDLE;
        // per-frame UBO (lightViewProj)
        std::vector<AllocatedBuffer> uboBuffers;
        std::vector<void *> uboMapped;
        // per-frame instance SSBO
        std::vector<AllocatedBuffer> instSSBOs;
        std::vector<void *> instMapped;
        // per-frame descriptor set (binding 0 UBO / binding 1 instance SSBO)
        std::vector<VkDescriptorSet> descSets;
        // Current image layout, used as the transition source
        VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::mat4 lastLightVP = glm::mat4(1.0f);
    };

    void createImages(VkDevice device, Allocator &allocator);
    void createSampler(VkDevice device);
    void createPipeline(VkDevice device);
    void createDescriptors(VkDevice device, Allocator &allocator);

    uint32_t resolution_ = 0;
    VkFormat format_ = VK_FORMAT_D32_SFLOAT;

    std::array<SlotResources, MAX_SPOT_SHADOW_SLOTS> slots_{};

    // Shared pipeline / descriptor resources
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
};
