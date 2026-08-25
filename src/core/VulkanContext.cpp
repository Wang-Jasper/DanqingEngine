#include "VulkanContext.h"
#include <set>      // std::set: dedupe queue family indices
#include <cstring>  // strcmp: compare validation layer names

// =============================================================================
// Debug messenger proxies: these extension functions are not exported by
// the Vulkan loader, so fetch their pointers via vkGetInstanceProcAddr.

// =============================================================================

// Proxy: create debug messenger
static VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,                                     // Vulkan instance
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,   // create info
    const VkAllocationCallbacks* pAllocator,                 // custom allocator (usually nullptr)
    VkDebugUtilsMessengerEXT* pDebugMessenger)               // out: created messenger handle
{
    // Fetch the vkCreateDebugUtilsMessengerEXT pointer via the instance
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {

        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    // Function not found — report "extension not present"
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// Proxy: destroy debug messenger
static void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,                       // Vulkan instance
    VkDebugUtilsMessengerEXT debugMessenger,   // messenger to destroy
    const VkAllocationCallbacks* pAllocator)   // custom allocator
{
    // Fetch the vkDestroyDebugUtilsMessengerEXT pointer via the instance
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// =============================================================================
// Public interface
// =============================================================================

// Initialize: run the 5 key steps in order
void VulkanContext::init(GLFWwindow* window) {
    createInstance();         // step 1: Vulkan instance
    setupDebugMessenger();    // step 2: debug messenger
    createSurface(window);   // step 3: window surface
    pickPhysicalDevice();    // step 4: physical device (GPU)
    createLogicalDevice();   // step 5: logical device and queues

    std::cout << "[VulkanContext] Initialized successfully.\n";
    std::cout << "[VulkanContext] Device: "
              << getDeviceProperties().deviceName << "\n";
}

// Destroy all Vulkan resources in reverse creation order (avoids dependency issues)
void VulkanContext::cleanup() {
    // Destroy the logical device (frees the queues created from it)
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    // Destroy the debug messenger (created only when validation is enabled)
    if (enableValidation && debugMessenger != VK_NULL_HANDLE) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }
    // Destroy the instance last — everything else must be freed first
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
    std::cout << "[VulkanContext] Cleaned up.\n";
}

VkPhysicalDeviceProperties VulkanContext::getDeviceProperties() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    return props;
}

// Timestamp period in nanoseconds per tick (converts query results to time)
float VulkanContext::getTimestampPeriod() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    return props.limits.timestampPeriod;
}

// Graphics-queue timestamp valid bits (0 = family doesn't support timestamps)
uint32_t VulkanContext::getTimestampValidBits() const {
    if (!queueFamilyIndices.graphicsFamily.has_value()) return 0;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, qprops.data());
    return qprops[queueFamilyIndices.graphicsFamily.value()].timestampValidBits;
}

// =============================================================================
// Step 1: create the Vulkan instance — the app's connection point to the
// Vulkan driver, unique per process.
// =============================================================================
void VulkanContext::createInstance() {
    // Validation layers requested but unavailable — throw
    if (enableValidation && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested but not available!");
    }

    // App info (optional, but drivers may optimize based on it)
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO; // sType must be set correctly
    appInfo.pApplicationName   = "Danqing";                        // app name
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);        // app version
    appInfo.pEngineName        = "DanqingEngine";                  // engine name
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);        // engine version
    appInfo.apiVersion         = VK_API_VERSION_1_3;               // require Vulkan 1.3 (Dynamic Rendering is a 1.3 core feature)

    // Instance extensions: GLFW window extensions + debug extension
    auto extensions = getRequiredExtensions();

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Configure validation layers and the debug callback when enabled
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidation) {

        createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        // Chain this temporary messenger via pNext so validation messages
        // during vkCreateInstance / vkDestroyInstance are still captured —
        // the real messenger doesn't exist yet (or is already destroyed).
        debugCreateInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT   // warnings
                                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;    // and errors
        debugCreateInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT       // general
                                        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT    // validation
                                        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;  // performance
        debugCreateInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext             = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }
}

// =============================================================================
// Step 2: set up the debug messenger — a persistent messenger receiving
// validation-layer messages for the whole application run.
// =============================================================================
void VulkanContext::setupDebugMessenger() {

    if (!enableValidation) return;

    // Same configuration as the temporary messenger in createInstance
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

// =============================================================================
// Step 3: create the window surface — VkSurfaceKHR abstracts the OS
// window system; GLFW's wrapper handles Windows/Linux/macOS differences.

// =============================================================================
void VulkanContext::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }
}

// =============================================================================
// Step 4: pick a physical device (GPU) — enumerate all GPUs and choose
// one meeting the requirements (discrete GPU preferred).
// =============================================================================
void VulkanContext::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Prefer discrete GPU over integrated GPU
    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (const auto& dev : devices) {
        if (isDeviceSuitable(dev)) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = dev;
                std::cout << "[VulkanContext] Selected discrete GPU: " << props.deviceName << "\n";
                return;
            }
            if (fallback == VK_NULL_HANDLE) {
                fallback = dev;
            }
        }
    }

    if (fallback != VK_NULL_HANDLE) {
        physicalDevice = fallback;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "[VulkanContext] No discrete GPU found, using: " << props.deviceName << "\n";
        return;
    }

    throw std::runtime_error("Failed to find a suitable GPU!");
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice dev) {

    QueueFamilyIndices indices = findQueueFamilies(dev);

    bool extensionsSupported = checkDeviceExtensionSupport(dev);

    // Require Vulkan 1.3 — Dynamic Rendering is a 1.3 core feature
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);

    if (props.apiVersion < VK_API_VERSION_1_3) {
        std::cout << "[VulkanContext] Skipping device " << props.deviceName
                  << " (Vulkan 1.3 not supported)\n";
        return false;
    }

    // Query the Dynamic Rendering feature via VkPhysicalDeviceFeatures2 +
    // pNext chain (introduced in Vulkan 1.1)
    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &dynamicRenderingFeatures;
    vkGetPhysicalDeviceFeatures2(dev, &features2);

    if (!dynamicRenderingFeatures.dynamicRendering) {
        std::cout << "[VulkanContext] Skipping device " << props.deviceName
                  << " (dynamic rendering not supported)\n";
        return false;
    }

    // Swap chain needs at least one surface format and one present mode
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentModeCount, nullptr);
        swapChainAdequate = formatCount > 0 && presentModeCount > 0;
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool VulkanContext::checkDeviceExtensionSupport(VkPhysicalDevice dev) {

    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, availableExtensions.data());

    // Erase each available extension from the set of required ones
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& ext : availableExtensions) {
        requiredExtensions.erase(ext.extensionName);
    }
    // Empty set means every required extension is supported
    return requiredExtensions.empty();
}

QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {

        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;
    }

    return indices;
}

// =============================================================================
// Step 5: create the logical device — the app's interface to the GPU; enables
// Dynamic Rendering and Synchronization2 (Vulkan 1.3 features used here).

// =============================================================================
void VulkanContext::createLogicalDevice() {

    queueFamilyIndices = findQueueFamilies(physicalDevice);

    // One VkDeviceQueueCreateInfo per unique family. std::set dedupes so a
    // family shared by graphics and present is created only once.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        queueFamilyIndices.graphicsFamily.value(),
        queueFamilyIndices.presentFamily.value()
    };

    // Queue priority (0.0-1.0); 1.0 is the highest
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount       = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // ======== pNext chain: enable Vulkan 1.3 core features ========
    // Feature structs are linked together through their pNext pointers.

    // Dynamic Rendering (Vulkan 1.3 core): replaces traditional VkRenderPass
    // with vkCmdBeginRendering / vkCmdEndRendering
    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

    // Synchronization2 (Vulkan 1.3 core): cleaner pipeline-barrier API
    // (VkImageMemoryBarrier2 + vkCmdPipelineBarrier2)
    VkPhysicalDeviceSynchronization2Features sync2Features{};
    sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    sync2Features.synchronization2 = VK_TRUE;
    // pNext chain: sync2Features -> dynamicRenderingFeatures
    sync2Features.pNext = &dynamicRenderingFeatures;

    // Base device features (Vulkan 1.0)
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy       = VK_TRUE;  // enable anisotropic filtering (better texture quality)
    deviceFeatures.wideLines               = VK_TRUE;  // enable wide lines (lineWidth > 1.0)
    deviceFeatures.pipelineStatisticsQuery = VK_TRUE;  // enable pipeline statistics queries

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.pEnabledFeatures        = &deviceFeatures;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();    // device extensions (VK_KHR_swapchain)
    // pNext chain head: sync2Features -> dynamicRenderingFeatures
    createInfo.pNext                   = &sync2Features;

    // Device-level layers are deprecated in newer Vulkan; kept for compatibility
    if (enableValidation) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    // Index 0 = the first queue of the family
    vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
}

// =============================================================================
// Helper methods
// =============================================================================

bool VulkanContext::checkValidationLayerSupport() {

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) return false;
    }
    return true;
}

std::vector<const char*> VulkanContext::getRequiredExtensions() {
    // GLFW provides the extensions needed to create a Vulkan surface (e.g. VK_KHR_surface, VK_KHR_win32_surface)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    // Add the debug-utils extension when validation is enabled
    if (enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// Debug callback invoked when the validation layer detects a problem.
// Returning VK_FALSE does not abort the Vulkan call that triggered it.
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,      // severity
    VkDebugUtilsMessageTypeFlagsEXT messageType,                 // type
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,   // message text
    void* pUserData)                                             // user data (unused)
{
    // Only print warning and above, filtering out info-level noise
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan Validation] " << pCallbackData->pMessage << "\n";
    }

    return VK_FALSE;
}
