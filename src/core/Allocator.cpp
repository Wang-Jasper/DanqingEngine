// VMA allocator wrapper: initialization, generic buffer creation, staging uploads.
#include "Allocator.h"
#include "VulkanContext.h"
#include "CommandManager.h"
#include <cstring>

void Allocator::init(VulkanContext& context, CommandManager* cmdMgr) {
    commandManager = cmdMgr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.physicalDevice   = context.getPhysicalDevice();
    allocatorInfo.device           = context.getDevice();
    allocatorInfo.instance         = context.getInstance();

    if (vmaCreateAllocator(&allocatorInfo, &vmaAllocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator!");
    }

    std::cout << "[Allocator] VMA initialized.\n";
}

// Must destroy all VMA-created buffers/images before the allocator.
void Allocator::cleanup() {
    if (vmaAllocator != VK_NULL_HANDLE) {
        // Unreleased resources are reported as leaks in debug builds.
        vmaDestroyAllocator(vmaAllocator);
        vmaAllocator = VK_NULL_HANDLE;
    }
    std::cout << "[Allocator] Cleaned up.\n";
}

AllocatedBuffer Allocator::createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = size;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;

    AllocatedBuffer buffer;
    // Creates buffer and allocation in one call; nullptr skips VmaAllocationInfo.
    if (vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocInfo,
                        &buffer.buffer, &buffer.allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer via VMA!");
    }

    return buffer;
}

AllocatedBuffer Allocator::createBufferWithStaging(
    VulkanContext& context,
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage)
{
    // 1. CPU-visible staging buffer.
    AllocatedBuffer staging = createBuffer(
        size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    // 2. Copy data in from the CPU side.
    void* mapped;
    vmaMapMemory(vmaAllocator, staging.allocation, &mapped);
    memcpy(mapped, data, size);
    vmaUnmapMemory(vmaAllocator, staging.allocation);

    // 3. GPU-only destination buffer.
    AllocatedBuffer gpuBuffer = createBuffer(
        size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // 4. One-shot copy from staging to the GPU buffer.
    VkDevice device = context.getDevice();
    VkCommandBuffer cmd = commandManager->beginSingleTimeCommands(device);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, gpuBuffer.buffer, 1, &copyRegion);

    commandManager->endSingleTimeCommands(device, context.getGraphicsQueue(), cmd);

    // 5. Free staging.
    destroyBuffer(staging);

    return gpuBuffer;
}

void Allocator::destroyBuffer(AllocatedBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        // Destroys buffer and frees its allocation in one call.
        vmaDestroyBuffer(vmaAllocator, buffer.buffer, buffer.allocation);
        // Reset handles to prevent double-destroy.
        buffer.buffer     = VK_NULL_HANDLE;
        buffer.allocation = VK_NULL_HANDLE;
    }
}

std::vector<Allocator::HeapStats> Allocator::getHeapStats() const {
    std::vector<HeapStats> result;
    if (vmaAllocator == VK_NULL_HANDLE) return result;

    // Heap flags/sizes come from device memory properties.
    // VMA returns a pointer to its internal cache — do not free it.
    const VkPhysicalDeviceMemoryProperties *props = nullptr;
    vmaGetMemoryProperties(vmaAllocator, &props);
    if (!props || props->memoryHeapCount == 0) return result;

    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(vmaAllocator, budgets);

    result.reserve(props->memoryHeapCount);
    for (uint32_t i = 0; i < props->memoryHeapCount; ++i) {
        HeapStats s;
        s.heapIndex       = i;
        s.flags           = props->memoryHeaps[i].flags;
        s.heapSize        = props->memoryHeaps[i].size;
        s.usage           = budgets[i].usage;
        s.budget          = budgets[i].budget;
        s.blockBytes      = budgets[i].statistics.blockBytes;
        s.allocationBytes = budgets[i].statistics.allocationBytes;
        s.allocationCount = budgets[i].statistics.allocationCount;
        result.push_back(s);
    }
    return result;
}
