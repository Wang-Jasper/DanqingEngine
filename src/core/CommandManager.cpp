// Command pool and command buffer management, plus one-shot command helpers.
#include "CommandManager.h"
#include "VulkanContext.h"

void CommandManager::init(VulkanContext& context, uint32_t framesInFlight) {
    // 1. Create the command pool.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // Allows resetting individual buffers each frame instead of the whole pool.
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // Buffers from this pool can only submit to this queue family.
    poolInfo.queueFamilyIndex = context.getQueueFamilies().graphicsFamily.value();

    if (vkCreateCommandPool(context.getDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }

    // 2. One command buffer per frame in flight.
    commandBuffers.resize(framesInFlight);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = commandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(context.getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers!");
    }

    std::cout << "[CommandManager] Created pool + " << framesInFlight << " command buffers.\n";
}

// Destroying the pool frees all command buffers allocated from it.
void CommandManager::cleanup(VkDevice device) {
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    commandBuffers.clear();
    std::cout << "[CommandManager] Cleaned up.\n";
}

// One-shot commands (staging copies, image layout transitions).
VkCommandBuffer CommandManager::beginSingleTimeCommands(VkDevice device) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // Hints the driver the buffer is submitted once (enables optimizations).
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

// vkQueueWaitIdle blocks until the GPU finishes — fine for one-off ops;
// use fences for frequent submits.
void CommandManager::endSingleTimeCommands(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;

    // No fence — vkQueueWaitIdle below handles sync.
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    // Simplest sync; acceptable for one-off uploads.
    vkQueueWaitIdle(queue);

    // Free only this temp buffer, not the whole pool.
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}
