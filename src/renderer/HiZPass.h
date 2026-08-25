// HiZPass — builds a conservative max-depth Hi-Z mip chain (R32_SFLOAT, same
// range as the D32 depth attachment) after Geometry Pass; later reused for
// GPU-driven occlusion culling. Not thread-safe; render-thread only.
#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <vector>
#include <cstdint>

// ============================================================================
// HiZPass
// ============================================================================
class HiZPass
{
public:
    static constexpr uint32_t MAX_MIP_LEVELS = 16; // Cap: covers up to 65536 pixels

    // Initialize: create the multi-mip Hi-Z image, compute pipeline, descriptor sets.
    //   extent       — Hi-Z root resolution (pass Geometry Pass viewportExtent)
    //   depthView    — Geometry Pass depth image view (aspect = DEPTH_BIT)
    //   depthSampler — sampler used to read depth in the first Hi-Z level
    void init(VkDevice device, VmaAllocator vma, VkExtent2D extent,
              VkImageView depthView, VkSampler depthSampler);

    // Call on viewport size change (e.g. editor resize); does cleanup + init internally.
    void resize(VkDevice device, VmaAllocator vma, VkExtent2D extent,
                VkImageView depthView, VkSampler depthSampler);

    // Record one frame's Hi-Z build commands into cmd.
    // Preconditions:
    //   * depthImage must be VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (caller
    //     transitions it after Geometry Pass via a pipeline barrier)
    //   * This internally moves the Hi-Z image (all mips) UNDEFINED -> GENERAL ->
    //     SHADER_READ_ONLY; caller can then sample it in the Lighting Pass.
    void dispatch(VkCommandBuffer cmd);

    // Destroy all Vulkan resources. Safe to call repeatedly.
    void cleanup(VkDevice device, VmaAllocator vma);

    // --- Accessors (Lighting Pass debug view) ---
    VkImageView fullView() const { return fullMipView_; } // View over all mips
    VkImageView mipView(uint32_t mip) const
    {
        return mip < mipViews_.size() ? mipViews_[mip] : VK_NULL_HANDLE;
    }
    VkSampler sampler() const { return hizSampler_; }
    uint32_t mipLevels() const { return mipLevels_; }
    VkExtent2D extent() const { return extent_; }

private:
    void createImage(VkDevice device, VmaAllocator vma);
    void createViews(VkDevice device);
    void createSampler(VkDevice device);
    void createPipeline(VkDevice device);
    void createDescriptorSets(VkDevice device,
                              VkImageView depthView, VkSampler depthSampler);

    // --- Resources ---
    VkExtent2D extent_ = {0, 0};
    uint32_t mipLevels_ = 0;
    VkFormat format_ = VK_FORMAT_R32_SFLOAT;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocator vma_ = VK_NULL_HANDLE;

    // One view per mip (used as src and dst), plus fullMipView for Lighting Pass sampling
    std::vector<VkImageView> mipViews_;
    VkImageView fullMipView_ = VK_NULL_HANDLE;

    VkSampler hizSampler_ = VK_NULL_HANDLE;           // linear clamp, for sampling from above
    VkSampler externalDepthSampler_ = VK_NULL_HANDLE; // Passed in by Renderer; owned by caller

    // Compute pipeline
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    // One descriptor set per mip (set i: src = mip i-1 or depth; dst = mip i)
    std::vector<VkDescriptorSet> descSets_;

    VkImageView depthView_ = VK_NULL_HANDLE; // Externally owned; only used to recreate descriptors
};
