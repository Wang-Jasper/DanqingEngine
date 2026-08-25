// ============================================================================
// VulkanUtils.h — inline helpers: transitionImageLayout (Vulkan 1.3
// Synchronization2), readShaderFile, createShaderModule.
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace VulkanUtils {

// ============================================================================
// transitionImageLayout — image layout transition via a pipeline barrier
// ============================================================================
// Uses Vulkan 1.3 Synchronization2 (VkImageMemoryBarrier2 /
// vkCmdPipelineBarrier2): stage and access masks live in one struct with
// 64-bit flags, which is harder to get wrong than the legacy API.
inline void transitionImageLayout(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT)
{
    // Synchronization2 image memory barrier
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // UNDEFINED = discard prior contents
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    // VK_QUEUE_FAMILY_IGNORED: no queue-family ownership transfer
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;

    // aspectMask: COLOR_BIT (color) or DEPTH_BIT (depth)
    barrier.subresourceRange.aspectMask     = aspectMask;
    // Mip range: start 0, 1 level
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    // Array layer range: start 0, 1 layer
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;

    // -----------------------------------------------------------------------
    // Stage/access masks per transition type
    // -----------------------------------------------------------------------

    // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL (frame start: swapchain image
    // becomes renderable)
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        // src: top of pipe, no prior work to wait on
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        // dst: color attachment output, wait for color attachment writes
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR (present after rendering)
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        // dst: bottom of pipe; the present engine handles its own sync
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = 0;
    }
    // UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL (depth attachment ready)
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        // dst: early fragment tests, wait for depth attachment writes
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    // UNDEFINED -> SHADER_READ_ONLY_OPTIMAL (texture samplable in shaders)
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        // dst: fragment shader, wait for shader reads
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    }
    // COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL (G-Buffer written,
    // then sampled in the lighting pass)
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    }
    // Fallback: most conservative sync parameters
    else {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkDependencyInfo depInfo{};
    depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    // Single image memory barrier
    depInfo.imageMemoryBarrierCount  = 1;
    depInfo.pImageMemoryBarriers     = &barrier;

    // Records the barrier into the command buffer; the GPU syncs here at execution
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

// ============================================================================
// readShaderFile — read a compiled SPIR-V shader file
// ============================================================================
// GLSL is compiled offline by glslc to .spv; this reads the binary back.
// Strategy: open binary + ate (cursor at EOF), tellg() gives the size,
// seekg(0) back, read it all.
inline std::vector<char> readShaderFile(const std::string& filename) {
    // ate: cursor at EOF so tellg() gives the size; binary: SPIR-V is binary, not text
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    // tellg() = file size (cursor is at EOF)
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    // Back to the start
    file.seekg(0);
    // Read the whole file
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

// ============================================================================
// createShaderModule — wrap SPIR-V bytecode in a VkShaderModule
// ============================================================================
// Needed to create a graphics pipeline; the module can be destroyed afterwards
// (the pipeline already compiled the code to hardware instructions).
inline VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    // Vulkan requires uint32_t*; reinterpret_cast is safe because
    // std::vector<char> is contiguous and SPIR-V is 4-byte aligned
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

} // namespace VulkanUtils
