// Allocator wraps the Vulkan Memory Allocator: buffer creation, staging uploads, destruction.
#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <vector>
#include <cstdint>

class VulkanContext;
class CommandManager;

// Pairs a VkBuffer with its VMA allocation so they stay together;
// not RAII — call destroyBuffer explicitly.
struct AllocatedBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    // Needed by vmaDestroyBuffer to free the allocation.
    VmaAllocation allocation = VK_NULL_HANDLE;
};

// Wraps the VMA allocator: init/cleanup, buffer creation, staging uploads.
class Allocator {
public:
    void init(VulkanContext& context, CommandManager* cmdMgr = nullptr);
    // Call after destroying all resources allocated through VMA.
    void cleanup();

    VmaAllocator getVma() const { return vmaAllocator; }

    // Staging upload for vertex/index data: GPU-only destination is faster
    // (DEVICE_LOCAL bandwidth beats host-visible memory) and is the standard
    // pattern for frequently-read buffers.
    AllocatedBuffer createBufferWithStaging(
        VulkanContext& context,
        const void* data,
        VkDeviceSize size,
        VkBufferUsageFlags usage);

    // Caller picks the memory type — e.g. GPU_ONLY for vertex/index, CPU_TO_GPU for uniforms.
    AllocatedBuffer createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VmaMemoryUsage memoryUsage);

    void destroyBuffer(AllocatedBuffer& buffer);

    struct HeapStats {
        uint32_t        heapIndex      = 0;
        VkMemoryHeapFlags flags        = 0;        // e.g. DEVICE_LOCAL / MULTI_INSTANCE
        VkDeviceSize    heapSize       = 0;        // Total heap capacity (per the OS)
        VkDeviceSize    usage          = 0;        // Bytes VMA has allocated
        VkDeviceSize    budget         = 0;        // Cap, including OS-side contention
        VkDeviceSize    blockBytes     = 0;        // VMA blocks total, including unused space
        VkDeviceSize    allocationBytes = 0;       // Bytes actually handed out
        uint32_t        allocationCount = 0;       // Live allocation count
    };
    // Per-heap stats from vmaGetHeapBudgets; thread-safe (VMA locks internally).
    std::vector<HeapStats> getHeapStats() const;

private:
    VmaAllocator vmaAllocator = VK_NULL_HANDLE;
    CommandManager* commandManager = nullptr;
};
