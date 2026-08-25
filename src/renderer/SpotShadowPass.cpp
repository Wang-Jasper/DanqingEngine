// ============================================================================
// SpotShadowPass.cpp - spot light shadow pass implementation
// ============================================================================
#include "renderer/SpotShadowPass.h"
#include "renderer/PipelineBuilder.h"
#include "scene/Vertex.h"
#include "utils/VulkanUtils.h"

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cstring>

#ifndef SHADER_DIR
#define SHADER_DIR "./shaders"
#endif

namespace
{
    // Matches ShadowPass / shadow.vert exactly, so shadow.vert/frag can be reused as-is.
    struct SpotShadowPushConstants
    {
        uint32_t instanceOffset;
        uint32_t _pad0;
        uint32_t _pad1;
        uint32_t _pad2;
    };

    struct SpotShadowVPUBO
    {
        alignas(16) glm::mat4 lightViewProj;
    };

    // InstanceData layout shared with Geometry Pass / ShadowPass (96 bytes)
    constexpr VkDeviceSize kInstanceStride = 96;
} // namespace

// ============================================================================
// init()
// ============================================================================
void SpotShadowPass::init(VkDevice device, Allocator &allocator, uint32_t shadowMapResolution)
{
    resolution_ = shadowMapResolution;
    createImages(device, allocator);
    createSampler(device);
    createPipeline(device);
    createDescriptors(device, allocator);

    std::cout << "[SpotShadowPass] Initialized (" << MAX_SPOT_SHADOW_SLOTS
              << " slots, " << resolution_ << "x" << resolution_
              << " D32_SFLOAT each).\n";
}

// ============================================================================
// resize()
// ============================================================================
void SpotShadowPass::resize(VkDevice device, Allocator &allocator, uint32_t newResolution)
{
    if (newResolution == resolution_)
        return;
    cleanup(device, allocator);
    init(device, allocator, newResolution);
}

// ============================================================================
// cleanup()
// ============================================================================
void SpotShadowPass::cleanup(VkDevice device, Allocator &allocator)
{
    if (descriptorPool_)
    {
        vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }

    for (auto &slot : slots_)
    {
        for (auto &b : slot.uboBuffers)
        {
            if (b.buffer)
            {
                vmaUnmapMemory(allocator.getVma(), b.allocation);
                allocator.destroyBuffer(b);
            }
        }
        slot.uboBuffers.clear();
        slot.uboMapped.clear();

        for (auto &b : slot.instSSBOs)
        {
            if (b.buffer)
            {
                vmaUnmapMemory(allocator.getVma(), b.allocation);
                allocator.destroyBuffer(b);
            }
        }
        slot.instSSBOs.clear();
        slot.instMapped.clear();
        slot.descSets.clear();

        if (slot.view)
        {
            vkDestroyImageView(device, slot.view, nullptr);
            slot.view = VK_NULL_HANDLE;
        }
        if (slot.image.image)
        {
            vmaDestroyImage(allocator.getVma(), slot.image.image, slot.image.allocation);
            slot.image.image = VK_NULL_HANDLE;
            slot.image.allocation = VK_NULL_HANDLE;
        }
        slot.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        slot.lastLightVP = glm::mat4(1.0f);
    }

    if (pipeline_)
    {
        vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_)
    {
        vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (dsLayout_)
    {
        vkDestroyDescriptorSetLayout(device, dsLayout_, nullptr);
        dsLayout_ = VK_NULL_HANDLE;
    }
    if (shadowSampler_)
    {
        vkDestroySampler(device, shadowSampler_, nullptr);
        shadowSampler_ = VK_NULL_HANDLE;
    }
    resolution_ = 0;
}

// ============================================================================
// createImages() - 4 independent D32_SFLOAT depth images
// ============================================================================
void SpotShadowPass::createImages(VkDevice device, Allocator &allocator)
{
    for (uint32_t i = 0; i < MAX_SPOT_SHADOW_SLOTS; ++i)
    {
        auto &slot = slots_[i];

        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = format_;
        ii.extent.width = resolution_;
        ii.extent.height = resolution_;
        ii.extent.depth = 1;
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (vmaCreateImage(allocator.getVma(), &ii, &aci,
                           &slot.image.image, &slot.image.allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("[SpotShadowPass] Failed to create slot image!");
        slot.image.format = format_;
        slot.image.extent = {resolution_, resolution_};

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = slot.image.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &vi, nullptr, &slot.view) != VK_SUCCESS)
            throw std::runtime_error("[SpotShadowPass] Failed to create slot image view!");
        slot.image.imageView = slot.view;
        slot.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

// ============================================================================
// createSampler() - comparison sampler matching ShadowPass, shared by all 4 images
// ============================================================================
void SpotShadowPass::createSampler(VkDevice device)
{
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    si.maxLod = 0.0f;
    si.minLod = 0.0f;
    si.unnormalizedCoordinates = VK_FALSE;
    if (vkCreateSampler(device, &si, nullptr, &shadowSampler_) != VK_SUCCESS)
        throw std::runtime_error("[SpotShadowPass] Failed to create shadow sampler!");
}

// ============================================================================
// createPipeline() - depth-only pipeline (reuses shadow.vert / shadow.frag)
// ============================================================================
void SpotShadowPass::createPipeline(VkDevice device)
{
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = static_cast<uint32_t>(bindings.size());
    li.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &dsLayout_) != VK_SUCCESS)
        throw std::runtime_error("[SpotShadowPass] Failed to create descriptor set layout!");

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(SpotShadowPushConstants);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsLayout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("[SpotShadowPass] Failed to create pipeline layout!");

    std::string shaderDir = SHADER_DIR;
    auto bindingDesc = Vertex::getBindingDescription();
    auto attributeDescs = Vertex::getAttributeDescriptions();

    PipelineBuilder builder;
    pipeline_ = builder
                    .setShaders(device, shaderDir + "/shadow.vert.spv",
                                shaderDir + "/shadow.frag.spv")
                    .setVertexInput(bindingDesc, attributeDescs.data(),
                                    static_cast<uint32_t>(attributeDescs.size()))
                    .setInputAssembly()
                    .setViewportDynamic()
                    // Same scheme as directional shadow: back-face culling + slope-scaled bias
                    .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
                                   VK_FRONT_FACE_COUNTER_CLOCKWISE)
                    .setDynamicDepthBias()
                    .setMultisampling()
                    .setDepthStencil(true, true, VK_COMPARE_OP_LESS)
                    .setColorBlending(false)
                    .setDepthAttachmentFormat(format_)
                    .build(device, pipelineLayout_);
    builder.cleanupShaderModules(device);
}

// ============================================================================
// createDescriptors() - per-slot per-frame UBO + instance SSBO + descriptor set
// ============================================================================
void SpotShadowPass::createDescriptors(VkDevice device, Allocator &allocator)
{
    // Pool capacity = 4 slots x 2 frames = 8 sets, one UBO + SSBO per set
    const uint32_t totalSets = MAX_SPOT_SHADOW_SLOTS * MAX_FRAMES_IN_FLIGHT;
    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[0].descriptorCount = totalSets;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = totalSets;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = static_cast<uint32_t>(ps.size());
    pi.pPoolSizes = ps.data();
    pi.maxSets = totalSets;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("[SpotShadowPass] Failed to create descriptor pool!");

    const VkDeviceSize instSize = kInstanceStride * MAX_INSTANCES_PER_SLOT;

    for (uint32_t s = 0; s < MAX_SPOT_SHADOW_SLOTS; ++s)
    {
        auto &slot = slots_[s];
        slot.uboBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        slot.uboMapped.resize(MAX_FRAMES_IN_FLIGHT);
        slot.instSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
        slot.instMapped.resize(MAX_FRAMES_IN_FLIGHT);
        slot.descSets.resize(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
        {
            slot.uboBuffers[f] = allocator.createBuffer(
                sizeof(SpotShadowVPUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);
            if (vmaMapMemory(allocator.getVma(), slot.uboBuffers[f].allocation,
                             &slot.uboMapped[f]) != VK_SUCCESS)
                throw std::runtime_error("[SpotShadowPass] Failed to map UBO!");
            // Write identity as a safe initial value so lighting.frag never reads garbage before the first update.
            SpotShadowVPUBO init{};
            init.lightViewProj = glm::mat4(1.0f);
            std::memcpy(slot.uboMapped[f], &init, sizeof(init));

            slot.instSSBOs[f] = allocator.createBuffer(
                instSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);
            if (vmaMapMemory(allocator.getVma(), slot.instSSBOs[f].allocation,
                             &slot.instMapped[f]) != VK_SUCCESS)
                throw std::runtime_error("[SpotShadowPass] Failed to map instance SSBO!");
        }

        // Allocate descriptor sets (per frame)
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        ai.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(device, &ai, slot.descSets.data()) != VK_SUCCESS)
            throw std::runtime_error("[SpotShadowPass] Failed to allocate descriptor sets!");

        // Write descriptors
        for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
        {
            VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = slot.uboBuffers[f].buffer;
            uboInfo.offset = 0;
            uboInfo.range = sizeof(SpotShadowVPUBO);

            VkDescriptorBufferInfo instInfo{};
            instInfo.buffer = slot.instSSBOs[f].buffer;
            instInfo.offset = 0;
            instInfo.range = instSize;

            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = slot.descSets[f];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &uboInfo;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = slot.descSets[f];
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].descriptorCount = 1;
            writes[1].pBufferInfo = &instInfo;
            vkUpdateDescriptorSets(device,
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }
    }
}

// ============================================================================
// updateLightSpaceVP() - perspective projection + lookAt
// ============================================================================
void SpotShadowPass::updateLightSpaceVP(uint32_t frameIndex, uint32_t slot,
                                        const glm::vec3 &lightPos,
                                        const glm::vec3 &lightDir,
                                        float outerAngleDeg,
                                        float range)
{
    if (slot >= MAX_SPOT_SHADOW_SLOTS)
        return;
    auto &S = slots_[slot];
    if (frameIndex >= S.uboMapped.size() || !S.uboMapped[frameIndex])
        return;

    glm::vec3 dir = glm::normalize(lightDir);
    if (glm::length(dir) < 1e-4f)
        dir = glm::vec3(0.0f, -1.0f, 0.0f); // safe fallback

    glm::vec3 worldUp = (glm::abs(dir.y) > 0.99f)
                            ? glm::vec3(0.0f, 0.0f, 1.0f)
                            : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 view = glm::lookAt(lightPos, lightPos + dir, worldUp);

    // FOV = outerAngle * 2 (full diameter), matching SpotLight's conical falloff;
    // near=0.1 avoids precision issues, far=range aligns with SpotLight.radius.
    float fovYRad = glm::radians(glm::clamp(outerAngleDeg, 1.0f, 89.0f) * 2.0f);
    glm::mat4 proj = glm::perspective(fovYRad, 1.0f,
                                      0.1f, glm::max(range, 0.2f));
    proj[1][1] *= -1.0f; // Vulkan Y-flip

    SpotShadowVPUBO ubo{};
    ubo.lightViewProj = proj * view;
    S.lastLightVP = ubo.lightViewProj;
    std::memcpy(S.uboMapped[frameIndex], &ubo, sizeof(ubo));
}

// ============================================================================
// getInstanceSSBOMapped()
// ============================================================================
void *SpotShadowPass::getInstanceSSBOMapped(uint32_t frameIndex, uint32_t slot) const
{
    if (slot >= MAX_SPOT_SHADOW_SLOTS)
        return nullptr;
    const auto &S = slots_[slot];
    if (frameIndex >= S.instMapped.size())
        return nullptr;
    return S.instMapped[frameIndex];
}

VkDeviceSize SpotShadowPass::getInstanceSSBOSize() const
{
    return kInstanceStride * MAX_INSTANCES_PER_SLOT;
}

// ============================================================================
// recordSlot() - depth-only render to slot N's shadow map
// ============================================================================
void SpotShadowPass::recordSlot(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot,
                                const RecordDrawsFn &drawFn)
{
    if (!pipeline_ || slot >= MAX_SPOT_SHADOW_SLOTS)
        return;
    auto &S = slots_[slot];

    // (1) Layout transition -> DEPTH_ATTACHMENT_OPTIMAL
    VulkanUtils::transitionImageLayout(cmd, S.image.image,
                                       S.currentLayout,
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    S.currentLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = S.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {resolution_, resolution_}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 0;
    ri.pColorAttachments = nullptr;
    ri.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport vp{};
    vp.width = static_cast<float>(resolution_);
    vp.height = static_cast<float>(resolution_);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, {resolution_, resolution_}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Same bias formula as directional shadow
    vkCmdSetDepthBias(cmd, /*constant*/ 1.5f, /*clamp*/ 0.0f, /*slopeScale*/ 2.5f);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1,
                            &S.descSets[frameIndex], 0, nullptr);
    drawFn(cmd, pipelineLayout_);

    vkCmdEndRendering(cmd);

    // (2) Layout transition -> SHADER_READ_ONLY
    VulkanUtils::transitionImageLayout(cmd, S.image.image,
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    S.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// ============================================================================
// transitionAllToReadOnly() / transitionSlotToReadOnly()
// ============================================================================
void SpotShadowPass::transitionAllToReadOnly(VkCommandBuffer cmd)
{
    for (uint32_t s = 0; s < MAX_SPOT_SHADOW_SLOTS; ++s)
        transitionSlotToReadOnly(cmd, s);
}

void SpotShadowPass::transitionSlotToReadOnly(VkCommandBuffer cmd, uint32_t slot)
{
    if (slot >= MAX_SPOT_SHADOW_SLOTS)
        return;
    auto &S = slots_[slot];
    if (!S.image.image)
        return;
    if (S.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return;
    VulkanUtils::transitionImageLayout(cmd, S.image.image,
                                       S.currentLayout,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    S.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
