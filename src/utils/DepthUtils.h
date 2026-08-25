// ============================================================================
// DepthUtils.h — depth buffer helpers: AllocatedImage, findDepthFormat,
// createDepthImage, destroyDepthImage.
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>
// VMA allocator (depth image memory allocation)
#include "vk_mem_alloc.h"
#include <stdexcept>
#include <iostream>

class Allocator;

// ============================================================================
// AllocatedImage — packs the image-related Vulkan handles: VkImage,
// VkImageView, VmaAllocation, format, extent.
// ============================================================================
struct AllocatedImage
{
    // GPU image
    VkImage image = VK_NULL_HANDLE;
    // Image view (how to interpret the image)
    VkImageView imageView = VK_NULL_HANDLE;
    // VMA allocation record (needed for destruction)
    VmaAllocation allocation = VK_NULL_HANDLE;
    // Pixel format
    VkFormat format = VK_FORMAT_UNDEFINED;
    // Resolution (width x height)
    VkExtent2D extent = {0, 0};
};

namespace DepthUtils
{

    // ============================================================================
    // findDepthFormat — best depth format the device supports (queried at runtime)
    // ============================================================================
    // Preference (high to low): D32_SFLOAT (best precision), D32_SFLOAT_S8_UINT,
    // D24_UNORM_S8_UINT (best compatibility).
    inline VkFormat findDepthFormat(VkPhysicalDevice physicalDevice)
    {
        VkFormat candidates[] = {
            VK_FORMAT_D32_SFLOAT,         // 32-bit float depth (no stencil)
            VK_FORMAT_D32_SFLOAT_S8_UINT, // 32-bit depth + 8-bit stencil
            VK_FORMAT_D24_UNORM_S8_UINT   // 24-bit depth + 8-bit stencil (most common)
        };

        // Query each candidate for device support
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            // props reports features per tiling mode: linear (CPU-readable),
            // optimal (best GPU performance), buffer.
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            // Pick the first candidate usable as a depth/stencil attachment in optimal tiling
            if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            {
                return format;
            }
        }
        // No supported format (unlikely): fail hard
        throw std::runtime_error("Failed to find supported depth format!");
    }

    // ============================================================================
    // createDepthImage — create the depth image and its image view
    // ============================================================================
    inline AllocatedImage createDepthImage(
        VkDevice device,
        VmaAllocator vma,
        VkExtent2D extent,
        VkFormat format)
    {
        AllocatedImage depthImage{};
        // Record format and extent
        depthImage.format = format;
        depthImage.extent = extent;

        // -----------------------------------------------------------------------
        // Step 1: create the VkImage
        // -----------------------------------------------------------------------
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        // 2D image
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        // For 2D images depth is always 1
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        // No mipmaps for a depth buffer (1 level)
        imageInfo.mipLevels = 1;
        // Single layer
        imageInfo.arrayLayers = 1;
        // No MSAA (1 sample)
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        // TILING_OPTIMAL: fastest GPU access; CPU cannot read/write directly
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // Depth/stencil attachment + shader-sampled (Hi-Z needs the sampled bit)
        imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        // GPU-only memory; the CPU never accesses the depth buffer
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        // vmaCreateImage creates the image, allocates memory, and binds in one call
        if (vmaCreateImage(vma, &imageInfo, &allocInfo,
                           &depthImage.image, &depthImage.allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create depth image!");
        }

        // -----------------------------------------------------------------------
        // Step 2: create the VkImageView
        // -----------------------------------------------------------------------
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = depthImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        // Depth aspect only; stencil formats (e.g. D24_S8) could also set STENCIL_BIT
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &depthImage.imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create depth image view!");
        }

        return depthImage;
    }

    // ============================================================================
    // destroyDepthImage — destroy the depth image resources (view first, then
    // the image; VMA frees the memory)
    // ============================================================================
    inline void destroyDepthImage(VkDevice device, VmaAllocator vma, AllocatedImage &img)
    {
        if (img.imageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, img.imageView, nullptr);
            img.imageView = VK_NULL_HANDLE;
        }
        // Destroy the image and free its VMA memory
        if (img.image != VK_NULL_HANDLE)
        {
            // vmaDestroyImage destroys the image and frees its memory together
            vmaDestroyImage(vma, img.image, img.allocation);
            img.image = VK_NULL_HANDLE;
            img.allocation = VK_NULL_HANDLE;
        }
    }

} // namespace DepthUtils
