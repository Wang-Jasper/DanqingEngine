// ============================================================================
// PointShadowPass.h - point light cubemap shadow pass
// ----------------------------------------------------------------------------
// Renders up to MAX_POINT_SHADOW_SLOTS cubemap depth-only shadow maps for the
// top-N brightest point lights (selected by Renderer). Lighting Pass
// samples them through samplerCubeShadow array uPointShadowMap[N] for PCF
// shadowing.
//
// Design:
//   1. One 1Kx1Kx6 D32_SFLOAT cubemap depth image per slot:
//      - one viewType=CUBE ImageView for lighting frag sampling
//      - 6 viewType=2D views (layerCount=1, base layer = i) as per-face render targets
//   2. Perspective projection: FOV=90 deg, aspect=1.0, near=0.1, far=range; 6 view
//      matrices built for +X/-X/+Y/-Y/+Z/-Z.
//   3. Reuses a dedicated shadow_point.vert / shadow_point.frag:
//      - vert: instance.model * inPosition -> worldPos -> cubeVP[face] * worldPos,
//        and passes vWorld to frag
//      - frag: gl_FragDepth = length(vWorld - lightPos) / range, i.e. writes
//        "linear distance / range" to the depth buffer; the shader compares
//        against the same normalized range.
//   4. Each slot owns a per-frame UBO (cubeVP[6] + lightPos + range) +
//      instance SSBO (caster list).
//   5. Top-N selection is done by Renderer; this class only turns
//      "light data + caster list" into N cubemaps. Slot 0..N-1 matches
//      PointLight.shadowSlot.
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

class PointShadowPass
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    // Must match lighting.frag uPointShadowMap[4]; resize the shader array when changed.
    static constexpr uint32_t MAX_POINT_SHADOW_SLOTS = 4;
    // Max casters per slot per frame. 1024 covers typical scenes; excess is truncated.
    static constexpr uint32_t MAX_INSTANCES_PER_SLOT = 1024;
    // Cubemap has 6 faces
    static constexpr uint32_t FACE_COUNT = 6;

    void init(VkDevice device, Allocator &allocator, uint32_t shadowMapResolution);
    void resize(VkDevice device, Allocator &allocator, uint32_t newResolution);
    void cleanup(VkDevice device, Allocator &allocator);

    // Computes and writes the 6 face VPs + lightPos + range for the given slot.
    // Face order: +X (0), -X (1), +Y (2), -Y (3), +Z (4), -Z (5)
    void updateLightSpaceVP(uint32_t frameIndex, uint32_t slot,
                            const glm::vec3 &lightPos,
                            float range);

    // Returns the instance SSBO host pointer for (slot, frame); caller writes InstanceData[].
    void *getInstanceSSBOMapped(uint32_t frameIndex, uint32_t slot) const;
    VkDeviceSize getInstanceSSBOSize() const;

    // Renders one slot's 6-face depth-only pass. drawFn is provided by the
    // caller and reused for every face (the caster list is identical; only the
    // face index changes, passed to vert via push constant to pick cubeVP[face]).
    //
    // Unity-style per-light shadow bias:
    //   constantBias / slopeBias are taken from PointLightComponent during
    //   selection (already scaled by the base factor) and passed straight to
    //   vkCmdSetDepthBias. The engine no longer hardcodes a global bias, so
    //   artists/JSON can tune per-light.
    using RecordDrawsFn = std::function<void(VkCommandBuffer cmd,
                                             VkPipelineLayout layout)>;
    void recordSlot(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot,
                    float constantBias, float slopeBias,
                    const RecordDrawsFn &drawFn);

    // Transitions all slot cubemaps to SHADER_READ_ONLY (on frames with no
    // casters the lighting pass descriptor must still point at readable images).
    void transitionAllToReadOnly(VkCommandBuffer cmd);
    // Single-slot variant, for the "slot k has no casters this frame" path.
    void transitionSlotToReadOnly(VkCommandBuffer cmd, uint32_t slot);

    // Lighting pass descriptor binding interface (CUBE view)
    VkImageView shadowCubeView(uint32_t slot) const
    {
        return (slot < MAX_POINT_SHADOW_SLOTS) ? slots_[slot].cubeView : VK_NULL_HANDLE;
    }
    VkSampler shadowSampler() const { return shadowSampler_; }
    uint32_t resolution() const { return resolution_; }

    // Last written lightPos / range (read by lighting.frag's PointShadowUBO).
    const glm::vec3 &lastLightPos(uint32_t slot) const
    {
        static const glm::vec3 kZero(0.0f);
        return (slot < MAX_POINT_SHADOW_SLOTS) ? slots_[slot].lastLightPos : kZero;
    }
    float lastRange(uint32_t slot) const
    {
        return (slot < MAX_POINT_SHADOW_SLOTS) ? slots_[slot].lastRange : 1.0f;
    }

private:
    struct SlotResources
    {
        AllocatedImage image{};                          // 6-layer cubemap
        VkImageView cubeView = VK_NULL_HANDLE;           // used by lighting frag sampling
        std::array<VkImageView, FACE_COUNT> faceViews{}; // used as per-face render targets
        // per-frame UBO (cubeVP[6] + lightPos + range)
        std::vector<AllocatedBuffer> uboBuffers;
        std::vector<void *> uboMapped;
        // per-frame instance SSBO
        std::vector<AllocatedBuffer> instSSBOs;
        std::vector<void *> instMapped;
        // per-frame descriptor set (binding 0 UBO / binding 1 instance SSBO)
        std::vector<VkDescriptorSet> descSets;
        // Current image layout (all 6 layers transition together), used as the transition source
        VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::vec3 lastLightPos = glm::vec3(0.0f);
        float lastRange = 1.0f;
    };

    void createImages(VkDevice device, Allocator &allocator);
    void createSampler(VkDevice device);
    void createPipeline(VkDevice device);
    void createDescriptors(VkDevice device, Allocator &allocator);

    uint32_t resolution_ = 0;
    VkFormat format_ = VK_FORMAT_D32_SFLOAT;

    std::array<SlotResources, MAX_POINT_SHADOW_SLOTS> slots_{};

    // Shared pipeline / descriptor resources
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
};
