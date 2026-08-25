// HiZPass.cpp — Hi-Z depth pyramid implementation

#include "renderer/HiZPass.h"
#include "utils/VulkanUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

    // HiZPass-internal push constant layout; must match shaders/hiz_downsample.comp
    struct HiZPushConstants
    {
        int32_t dstSizeX;
        int32_t dstSizeY;
        int32_t srcSizeX;
        int32_t srcSizeY;
        int32_t mode; // 0 = depth -> mip0; 1 = mip(i-1) -> mip i
        int32_t _pad0;
        int32_t _pad1;
        int32_t _pad2;
    };

    // floor(log2(maxDim)) + 1 levels, per the Vulkan spec.
    uint32_t computeMipLevels(VkExtent2D e, uint32_t cap)
    {
        uint32_t largest = std::max(e.width, e.height);
        if (largest == 0)
            return 0;
        uint32_t levels = static_cast<uint32_t>(std::floor(std::log2(largest))) + 1u;
        return std::min(levels, cap);
    }

} // anonymous namespace

// ============================================================================
// init()
// ============================================================================
void HiZPass::init(VkDevice device, VmaAllocator vma, VkExtent2D extent,
                   VkImageView depthView, VkSampler depthSampler)
{
    vma_ = vma;
    extent_ = extent;
    mipLevels_ = computeMipLevels(extent_, MAX_MIP_LEVELS);
    depthView_ = depthView;
    externalDepthSampler_ = depthSampler;

    if (mipLevels_ == 0 || extent_.width == 0 || extent_.height == 0)
    {
        std::cerr << "[HiZPass] init skipped (zero extent).\n";
        return;
    }

    createImage(device, vma);
    createViews(device);
    createSampler(device);
    createPipeline(device);
    createDescriptorSets(device, depthView, depthSampler);

    std::cout << "[HiZPass] Initialized: " << extent_.width << "x" << extent_.height
              << ", " << mipLevels_ << " mip levels.\n";
}

// ============================================================================
// resize()
// ============================================================================
void HiZPass::resize(VkDevice device, VmaAllocator vma, VkExtent2D extent,
                     VkImageView depthView, VkSampler depthSampler)
{
    cleanup(device, vma);
    init(device, vma, extent, depthView, depthSampler);
}

// ============================================================================
// createImage()
// ============================================================================
void HiZPass::createImage(VkDevice device, VmaAllocator vma)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format_;
    info.extent.width = extent_.width;
    info.extent.height = extent_.height;
    info.extent.depth = 1;
    info.mipLevels = mipLevels_;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE: compute writes; SAMPLED: compute reads (except mip0) + Lighting Pass debug sampling
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(vma, &info, &alloc, &image_, &allocation_, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: vmaCreateImage failed");
    }
    (void)device; // currently not needed here; kept for symmetry
}

// ============================================================================
// createViews()
// ============================================================================
void HiZPass::createViews(VkDevice device)
{
    mipViews_.assign(mipLevels_, VK_NULL_HANDLE);
    for (uint32_t m = 0; m < mipLevels_; ++m)
    {
        VkImageViewCreateInfo v{};
        v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image = image_;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D;
        v.format = format_;
        v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.baseMipLevel = m;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.baseArrayLayer = 0;
        v.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &v, nullptr, &mipViews_[m]) != VK_SUCCESS)
        {
            throw std::runtime_error("HiZPass: vkCreateImageView (per-mip) failed");
        }
    }

    // Extra view covering all mips, for Lighting Pass textureLod/sampling
    VkImageViewCreateInfo vAll{};
    vAll.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vAll.image = image_;
    vAll.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vAll.format = format_;
    vAll.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vAll.subresourceRange.baseMipLevel = 0;
    vAll.subresourceRange.levelCount = mipLevels_;
    vAll.subresourceRange.baseArrayLayer = 0;
    vAll.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &vAll, nullptr, &fullMipView_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: vkCreateImageView (full) failed");
    }
}

// ============================================================================
// createSampler()
// ============================================================================
void HiZPass::createSampler(VkDevice device)
{
    VkSamplerCreateInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // Compute shader uses normalized coords + linear clamp so out-of-bounds
    // samples clamp to the edge, simplifying bounds checks; also suits textureLod.
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.minLod = 0.0f;
    s.maxLod = static_cast<float>(mipLevels_);
    if (vkCreateSampler(device, &s, nullptr, &hizSampler_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: vkCreateSampler failed");
    }
}

// ============================================================================
// createPipeline()
// ============================================================================
void HiZPass::createPipeline(VkDevice device)
{
    // --- Descriptor Set Layout ---
    // binding 0: sampled image (src depth or src mip)
    // binding 1: storage image (dst mip)
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = static_cast<uint32_t>(bindings.size());
    li.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &dsLayout_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: create DS layout failed");
    }

    // --- Pipeline Layout ---
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(HiZPushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &pipelineLayout_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: create pipeline layout failed");
    }

    // --- Shader module ---
    std::string shaderDir = SHADER_DIR;
    auto code = VulkanUtils::readShaderFile(shaderDir + "/hiz_downsample.comp.spv");
    shaderModule_ = VulkanUtils::createShaderModule(device, code);

    // --- Compute Pipeline ---
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule_;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = stage;
    cpi.layout = pipelineLayout_;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: create compute pipeline failed");
    }
}

// ============================================================================
// createDescriptorSets()
// ============================================================================
void HiZPass::createDescriptorSets(VkDevice device,
                                   VkImageView depthView, VkSampler depthSampler)
{
    // One descriptor set per mip
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mipLevels_};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, mipLevels_};

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pi.pPoolSizes = sizes.data();
    pi.maxSets = mipLevels_;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &descPool_) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: create desc pool failed");
    }

    std::vector<VkDescriptorSetLayout> layouts(mipLevels_, dsLayout_);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = mipLevels_;
    ai.pSetLayouts = layouts.data();
    descSets_.assign(mipLevels_, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(device, &ai, descSets_.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("HiZPass: alloc desc sets failed");
    }

    // Descriptor writes: set i binding0 = src (depth for i=0, else mip i-1);
    // binding1 = dst mip i storage image.
    for (uint32_t i = 0; i < mipLevels_; ++i)
    {
        VkDescriptorImageInfo srcInfo{};
        if (i == 0)
        {
            srcInfo.sampler = depthSampler;
            srcInfo.imageView = depthView;
            srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else
        {
            srcInfo.sampler = hizSampler_;
            srcInfo.imageView = mipViews_[i - 1];
            srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VkDescriptorImageInfo dstInfo{};
        dstInfo.sampler = VK_NULL_HANDLE;
        dstInfo.imageView = mipViews_[i];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = descSets_[i];
        w[0].dstBinding = 0;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].descriptorCount = 1;
        w[0].pImageInfo = &srcInfo;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = descSets_[i];
        w[1].dstBinding = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].descriptorCount = 1;
        w[1].pImageInfo = &dstInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
    }
}

// ============================================================================
// dispatch()
// ============================================================================
void HiZPass::dispatch(VkCommandBuffer cmd)
{
    if (mipLevels_ == 0 || image_ == VK_NULL_HANDLE)
        return;

    // ------------------------------------------------------------------------
    // Step 1: transition whole Hi-Z image UNDEFINED -> GENERAL (compute writes)
    // ------------------------------------------------------------------------
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image_;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = mipLevels_;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask = 0;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // ------------------------------------------------------------------------
    // Step 2: one dispatch per mip; before mip>0, barrier mip-1 WRITE -> READ
    // ------------------------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    uint32_t mw = extent_.width;
    uint32_t mh = extent_.height;
    uint32_t prevW = mw, prevH = mh; // mip 0 sources from depth, same size as mip 0
    for (uint32_t m = 0; m < mipLevels_; ++m)
    {
        uint32_t dstW = std::max(1u, mw);
        uint32_t dstH = std::max(1u, mh);
        uint32_t srcW, srcH;
        if (m == 0)
        {
            // Build from depth: source size = extent_, dst = mip0 = extent_
            srcW = extent_.width;
            srcH = extent_.height;
        }
        else
        {
            // Build from previous mip
            srcW = prevW;
            srcH = prevH;

            // Barrier: mip (m-1) WRITE (from last dispatch) -> READ (for sampling)
            VkImageMemoryBarrier2 b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = image_;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel = m - 1;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount = 1;
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        HiZPushConstants pcData{};
        pcData.dstSizeX = static_cast<int32_t>(dstW);
        pcData.dstSizeY = static_cast<int32_t>(dstH);
        pcData.srcSizeX = static_cast<int32_t>(srcW);
        pcData.srcSizeY = static_cast<int32_t>(srcH);
        pcData.mode = (m == 0) ? 0 : 1;
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(HiZPushConstants), &pcData);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1, &descSets_[m], 0, nullptr);

        // Dispatch: one group per 16x16 threads
        uint32_t gx = (dstW + 15u) / 16u;
        uint32_t gy = (dstH + 15u) / 16u;
        vkCmdDispatch(cmd, gx, gy, 1);

        prevW = dstW;
        prevH = dstH;

        // Next level dst size: at least 1 per dimension
        mw = std::max(1u, mw / 2u);
        mh = std::max(1u, mh / 2u);
    }

    // ------------------------------------------------------------------------
    // Step 3: transition all mips to SHADER_READ_ONLY_OPTIMAL for Lighting sampling
    // ------------------------------------------------------------------------
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image_;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = mipLevels_;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

// ============================================================================
// cleanup()
// ============================================================================
void HiZPass::cleanup(VkDevice device, VmaAllocator vma)
{
    if (descPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
    }
    descSets_.clear();

    if (pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (shaderModule_ != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, shaderModule_, nullptr);
        shaderModule_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (dsLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, dsLayout_, nullptr);
        dsLayout_ = VK_NULL_HANDLE;
    }

    if (hizSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, hizSampler_, nullptr);
        hizSampler_ = VK_NULL_HANDLE;
    }

    for (auto &v : mipViews_)
    {
        if (v != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, v, nullptr);
            v = VK_NULL_HANDLE;
        }
    }
    mipViews_.clear();
    if (fullMipView_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, fullMipView_, nullptr);
        fullMipView_ = VK_NULL_HANDLE;
    }

    if (image_ != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
    mipLevels_ = 0;
    extent_ = {0, 0};
}
