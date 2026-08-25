// G-Buffer attachment creation and teardown.
// Format rationale: half-float for position (large, possibly negative range); signed
// half-float for normal ([-1, 1]) + roughness; 8-bit UNORM for albedo + metalness
// (colors are in [0, 1], 8 bits suffice and save memory).
#include "GBuffer.h"
#include <stdexcept>
#include <iostream>

void GBuffer::init(VkDevice device, VmaAllocator vma, VkExtent2D ext) {
    extent = ext;

    createAttachment(device, vma, VK_FORMAT_R16G16B16A16_SFLOAT, extent, attachments[POSITION]);
    createAttachment(device, vma, VK_FORMAT_R16G16B16A16_SFLOAT, extent, attachments[NORMAL]);
    createAttachment(device, vma, VK_FORMAT_R8G8B8A8_UNORM, extent, attachments[ALBEDO]);

    std::cout << "[GBuffer] Created " << COUNT << " attachments ("
              << extent.width << "x" << extent.height << ").\n";
}

// Destroys attachments: image views first, then images + VMA allocations.
void GBuffer::cleanup(VkDevice device, VmaAllocator vma) {
    for (uint32_t i = 0; i < COUNT; i++) {
        if (attachments[i].imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, attachments[i].imageView, nullptr);
            attachments[i].imageView = VK_NULL_HANDLE;
        }
        if (attachments[i].image != VK_NULL_HANDLE) {
            vmaDestroyImage(vma, attachments[i].image, attachments[i].allocation);
            attachments[i].image      = VK_NULL_HANDLE;
            attachments[i].allocation = VK_NULL_HANDLE;
        }
    }
}

std::vector<VkFormat> GBuffer::getColorFormats() const {
    std::vector<VkFormat> formats(COUNT);
    for (uint32_t i = 0; i < COUNT; i++) {
        formats[i] = attachments[i].format;
    }
    return formats;
}

// Images for layout transitions: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL each frame,
// then COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL after the geometry pass.
std::array<VkImage, GBuffer::COUNT> GBuffer::getImages() const {
    std::array<VkImage, COUNT> imgs;
    for (uint32_t i = 0; i < COUNT; i++) {
        imgs[i] = attachments[i].image;
    }
    return imgs;
}

// Creates one attachment. The usage flags let the same image be written as a color
// attachment in the geometry pass and sampled as a texture in the lighting pass.
void GBuffer::createAttachment(
    VkDevice device, VmaAllocator vma,
    VkFormat format, VkExtent2D ext,
    GBufferAttachment& attachment)
{
    attachment.format = format;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = format;
    imageInfo.extent.width  = ext.width;
    imageInfo.extent.height = ext.height;
    imageInfo.extent.depth  = 1;
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    // GPU-only memory; the CPU never reads the G-buffer
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(vma, &imageInfo, &allocInfo,
                       &attachment.image, &attachment.allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create G-Buffer attachment!");
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = attachment.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &attachment.imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create G-Buffer image view!");
    }
}
