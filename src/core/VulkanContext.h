
#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <optional>
#include <stdexcept>
#include <iostream>

// ============================================================================
// Queue family indices for the graphics and present queues; std::optional
// marks a family that has not been found yet.

// ============================================================================
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;  // accepts draw commands
    std::optional<uint32_t> presentFamily;   // can present images to the window surface

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// ============================================================================
// VulkanContext — owns core Vulkan initialization:
// instance, debug callback, window surface, physical device, logical device.

// ============================================================================
class VulkanContext {
public:
    // Initialize the whole context (runs the 5 steps above in order)
    void init(GLFWwindow* window);
    // Destroy all Vulkan resources in reverse creation order
    void cleanup();

    // ---- Getters: expose Vulkan handles to other modules ----
    VkInstance       getInstance()       const { return instance; }        // Vulkan instance
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }  // physical device (GPU)
    VkDevice         getDevice()         const { return device; }          // logical device
    VkQueue          getGraphicsQueue()  const { return graphicsQueue; }   // graphics queue
    VkQueue          getPresentQueue()   const { return presentQueue; }    // present queue
    VkSurfaceKHR     getSurface()        const { return surface; }         // window surface
    QueueFamilyIndices getQueueFamilies() const { return queueFamilyIndices; } // queue family indices

    VkPhysicalDeviceProperties getDeviceProperties() const;

    // Timestamp period (ns/tick) for GPU time queries
    float    getTimestampPeriod()    const;
    // Graphics-queue timestamp valid bits (0 = unsupported)
    uint32_t getTimestampValidBits() const;

private:
    // ---- Init steps (in call order) ----
    void createInstance();                     // step 1: Vulkan instance
    void setupDebugMessenger();                // step 2: debug messenger (validation only)
    void createSurface(GLFWwindow* window);    // step 3: window surface (GLFW, cross-platform)
    void pickPhysicalDevice();                 // step 4: physical device (GPU)
    void createLogicalDevice();                // step 5: logical device and queues

    // ---- Helpers ----
    bool checkValidationLayerSupport();                       // required validation layers available?
    bool isDeviceSuitable(VkPhysicalDevice device);           // device meets requirements?
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);// device supports required extensions?
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device); // find the device's queue families
    std::vector<const char*> getRequiredExtensions();         // extensions the instance needs

    // Validation-layer debug callback. Static because Vulkan needs a C-style
    // function pointer; VKAPI_ATTR / VKAPI_CALL set the calling convention.
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,  // severity (info/warning/error)
        VkDebugUtilsMessageTypeFlagsEXT messageType,             // type (general/validation/performance)
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, // callback data (message text)
        void* pUserData);                                        // user data (unused)

    // ---- Vulkan handle members (VK_NULL_HANDLE = not created) ----
    VkInstance               instance        = VK_NULL_HANDLE;  // app's connection to the Vulkan library
    VkDebugUtilsMessengerEXT debugMessenger  = VK_NULL_HANDLE;  // receives validation-layer messages
    VkSurfaceKHR             surface         = VK_NULL_HANDLE;  // bridge between Vulkan and the OS window system
    VkPhysicalDevice         physicalDevice  = VK_NULL_HANDLE;  // actual GPU hardware
    VkDevice                 device          = VK_NULL_HANDLE;  // logical interface to the GPU
    VkQueue                  graphicsQueue   = VK_NULL_HANDLE;  // submits draw commands
    VkQueue                  presentQueue    = VK_NULL_HANDLE;  // submits presentation commands
    QueueFamilyIndices       queueFamilyIndices;                // cached queue family indices

    // Required validation layers (Khronos official; detect API misuse)
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    // Required device extensions (swap chain is required to present to a window)
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // Compile-time switch via ENABLE_VALIDATION_LAYERS: on in Debug, off in
    // Release to avoid the performance cost.
#if ENABLE_VALIDATION_LAYERS
    static constexpr bool enableValidation = true;
#else
    static constexpr bool enableValidation = false;
#endif
};
