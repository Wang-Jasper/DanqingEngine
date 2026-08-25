// CommandManager owns the command pool and per-frame command buffers,
// plus helpers for one-shot commands.
#pragma once

#include <vulkan/vulkan.h>
#include <vector>

class VulkanContext;

class CommandManager {
public:
    // Pool bound to the graphics queue family; one buffer per frame in flight.
    void init(VulkanContext& context, uint32_t framesInFlight);

    // Destroying the pool frees all buffers allocated from it.
    void cleanup(VkDevice device);

    VkCommandPool                      getCommandPool()       const { return commandPool; }
    const std::vector<VkCommandBuffer>& getCommandBuffers()   const { return commandBuffers; }
    VkCommandBuffer                    getCommandBuffer(uint32_t frame) const { return commandBuffers[frame]; }

    // One-shot commands: staging copies, image layout transitions.
    VkCommandBuffer beginSingleTimeCommands(VkDevice device);
    void            endSingleTimeCommands(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer);

private:
    VkCommandPool                commandPool = VK_NULL_HANDLE;
    // One per frame — the previous frame may still be executing on the GPU.
    std::vector<VkCommandBuffer> commandBuffers;
};
