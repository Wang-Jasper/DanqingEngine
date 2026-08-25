// Swapchain lifecycle: capability queries, format/present-mode/extent selection,
// creation, per-image image views, and rebuild on window resize.
#include "Swapchain.h"
#include "VulkanContext.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>

void Swapchain::init(VulkanContext& context, GLFWwindow* window) {
    createSwapchain(context, window);
    createImageViews(context.getDevice());
    std::cout << "[Swapchain] Created (" << extent.width << "x" << extent.height
              << ", " << images.size() << " images)\n";
}

// Destroy image views first, then the swapchain itself.
// Swapchain images are owned by the swapchain — never destroy them manually.
void Swapchain::cleanup(VkDevice device) {
    for (auto imageView : imageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    imageViews.clear();

    if (swapchain != VK_NULL_HANDLE) {
        // Destroying the swapchain also destroys its images.
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        // Reset handle to prevent double-destroy.
        swapchain = VK_NULL_HANDLE;
    }
    images.clear();
    std::cout << "[Swapchain] Cleaned up.\n";
}

// Rebuild after window resize, VK_ERROR_OUT_OF_DATE_KHR, or VK_SUBOPTIMAL_KHR.
void Swapchain::recreate(VulkanContext& context, GLFWwindow* window) {
    // Block until un-minimized — a 0x0 framebuffer cannot host a swapchain.
    int width = 0, height = 0;
    // Framebuffer size may differ from window size (larger on high-DPI displays).
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        // Blocks until a window event; cheaper than glfwPollEvents here.
        glfwWaitEvents();
    }

    // Ensure the GPU is done with the old resources before rebuilding.
    vkDeviceWaitIdle(context.getDevice());
    cleanup(context.getDevice());
    createSwapchain(context, window);
    createImageViews(context.getDevice());
    std::cout << "[Swapchain] Recreated (" << extent.width << "x" << extent.height << ")\n";
}

void Swapchain::createSwapchain(VulkanContext& context, GLFWwindow* window) {
    SwapchainSupportDetails support = querySwapchainSupport(
        context.getPhysicalDevice(), context.getSurface());

    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
    VkPresentModeKHR   presentMode   = choosePresentMode(support.presentModes);
    VkExtent2D         swapExtent    = chooseExtent(support.capabilities, window);

    // min+1 images for triple buffering: the GPU need not wait on presentation.
    uint32_t imageCount = support.capabilities.minImageCount + 1;
    // Clamp to maxImageCount (0 means unlimited).
    if (support.capabilities.maxImageCount > 0 &&
        imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = context.getSurface();
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = swapExtent;
    // 1 unless doing stereoscopic (VR) rendering.
    createInfo.imageArrayLayers = 1;
    // Color attachment (render target).
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = context.getQueueFamilies();
    uint32_t queueFamilyIndices[] = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    // CONCURRENT when graphics and present families differ: no ownership
    // transfers, at a small performance cost.
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        // EXCLUSIVE: best performance — same family on most hardware.
        createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    }

    // currentTransform: no extra rotation/flip.
    createInfo.preTransform   = support.capabilities.currentTransform;
    // Opaque: ignore the alpha channel.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode    = presentMode;
    // Allow clipping occluded pixels — cheaper.
    createInfo.clipped        = VK_TRUE;
    // NULL — the old swapchain was already destroyed in cleanup.
    createInfo.oldSwapchain   = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(context.getDevice(), &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swap chain!");
    }

    // Two-call pattern; the driver may create more images than requested.
    vkGetSwapchainImagesKHR(context.getDevice(), swapchain, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(context.getDevice(), swapchain, &imageCount, images.data());

    imageFormat = surfaceFormat.format;
    extent      = swapExtent;
}

// Pipelines sample images through image views, never VkImage directly.
void Swapchain::createImageViews(VkDevice device) {
    imageViews.resize(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = imageFormat;

        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        // COLOR aspect; depth images use VK_IMAGE_ASPECT_DEPTH_BIT.
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &imageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view!");
        }
    }
}

// Queries surface capabilities, supported formats, and present modes.
SwapchainSupportDetails Swapchain::querySwapchainSupport(
    VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

// Prefer B8G8R8A8_SRGB in SRGB_NONLINEAR (gamma-corrected); fall back to the first format.
VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& formats)
{
    for (const auto& format : formats) {
        if (format.format     == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats[0];
}

// Prefer IMMEDIATE (no vsync, uncapped frame rate), then MAILBOX (triple buffering).
// FIFO, the only mode the spec guarantees, is the fallback.
VkPresentModeKHR Swapchain::choosePresentMode(
    const std::vector<VkPresentModeKHR>& presentModes)
{
    for (const auto& mode : presentModes) {
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            return mode;
        }
    }
    for (const auto& mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

// UINT32_MAX currentExtent means the driver lets us pick; otherwise use its choice.
VkExtent2D Swapchain::chooseExtent(
    const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
    // Extra parens avoid clashing with the Windows max macro.
    if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
        return capabilities.currentExtent;
    }

    int width, height;
    // May be larger than the window on high-DPI displays.
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    // Clamp to the surface's supported range.
    actualExtent.width  = std::clamp(actualExtent.width,
        capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
        capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}
