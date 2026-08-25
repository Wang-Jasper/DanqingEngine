// G-Buffer: three color attachments written by the geometry pass, sampled by lighting.
//   RT0 POSITION: R16G16B16A16_SFLOAT — world-space position (xyz) + flag (w)
//   RT1 NORMAL:   R16G16B16A16_SFLOAT — world-space normal (xyz) + roughness (w)
//   RT2 ALBEDO:   R8G8B8A8_UNORM — albedo (rgb) + metalness (a)
// Images are COLOR_ATTACHMENT | SAMPLED; rebuilt when the window resizes.
#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <vector>
#include <array>

class Allocator;

// All Vulkan resources for one G-Buffer render target.
struct GBufferAttachment {
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   imageView  = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;  // Needed for cleanup
    VkFormat      format     = VK_FORMAT_UNDEFINED;
};

// Manages the G-Buffer attachments: creation, destruction, and accessors for
// pipeline creation, descriptor binding, and layout transitions.
class GBuffer {
public:
    static constexpr uint32_t POSITION = 0;
    static constexpr uint32_t NORMAL   = 1;
    static constexpr uint32_t ALBEDO   = 2;
    static constexpr uint32_t COUNT    = 3;

    // Creates all attachments; extent must match the swapchain resolution.
    void init(VkDevice device, VmaAllocator vma, VkExtent2D extent);

    void cleanup(VkDevice device, VmaAllocator vma);

    VkImageView getImageView(uint32_t index) const { return attachments[index].imageView; }

    // Formats for PipelineBuilder::setColorAttachmentFormats
    std::vector<VkFormat> getColorFormats() const;

    // All VkImages for layout-transition barriers; called every frame, so no heap allocation
    std::array<VkImage, COUNT> getImages() const;

private:
    void createAttachment(VkDevice device, VmaAllocator vma,
                          VkFormat format, VkExtent2D extent,
                          GBufferAttachment& attachment);

    GBufferAttachment attachments[COUNT];
    // Resolution, kept in sync with the swapchain
    VkExtent2D extent = {0, 0};
};
