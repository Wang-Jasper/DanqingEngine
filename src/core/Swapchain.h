// Swapchain owns the presentation image set, their image views, and rebuild-on-resize.
#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>

class VulkanContext;

// Result of querying surface capabilities, formats, and present modes.
struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

class Swapchain {
public:
    void init(VulkanContext& context, GLFWwindow* window);

    // Image views first, then the swapchain; images are swapchain-owned.
    void cleanup(VkDevice device);

    // Rebuild on resize or swapchain expiry; blocks while minimized.
    void recreate(VulkanContext& context, GLFWwindow* window);

    VkSwapchainKHR   getSwapchain()   const { return swapchain; }
    VkFormat         getImageFormat() const { return imageFormat; }
    VkExtent2D       getExtent()      const { return extent; }
    uint32_t         getImageCount()  const { return static_cast<uint32_t>(images.size()); }

    const std::vector<VkImage>&     getImages()     const { return images; }
    const std::vector<VkImageView>& getImageViews() const { return imageViews; }

private:
    void createSwapchain(VulkanContext& context, GLFWwindow* window);
    void createImageViews(VkDevice device);

    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
    VkSurfaceFormatKHR      chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR        choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes);
    VkExtent2D              chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);

    VkSwapchainKHR             swapchain   = VK_NULL_HANDLE;
    VkFormat                   imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D                 extent      = {0, 0};
    std::vector<VkImage>       images;
    std::vector<VkImageView>   imageViews;
};
