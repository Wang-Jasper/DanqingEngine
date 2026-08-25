// ============================================================================
// ShadowPass.cpp - directional light shadow pass implementation
// ============================================================================
#include "renderer/ShadowPass.h"
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
    // Push constant layout matches Geometry Pass (16 bytes / 4 x uint).
    // Only `instanceOffset` (offset 0) is read by shadow.vert; rest is padding.
    struct ShadowPushConstants
    {
        uint32_t instanceOffset;
        uint32_t _pad0;
        uint32_t _pad1;
        uint32_t _pad2;
    };

    struct ShadowVPUBO
    {
        alignas(16) glm::mat4 lightViewProj;
    };
} // namespace

// ============================================================================
// init()
// ============================================================================
void ShadowPass::init(VkDevice device, Allocator &allocator, uint32_t shadowMapResolution)
{
    resolution_ = shadowMapResolution;

    createImage(device, allocator);
    createSampler(device);
    createPipeline(device);
    createDescriptors(device, allocator);

    std::cout << "[ShadowPass] Initialized (directional, " << resolution_
              << "x" << resolution_ << " D32_SFLOAT).\n";
}

// ============================================================================
// resize()
// ============================================================================
void ShadowPass::resize(VkDevice device, Allocator &allocator, uint32_t newResolution)
{
    if (newResolution == resolution_)
        return;
    cleanup(device, allocator);
    init(device, allocator, newResolution);
}

// ============================================================================
// cleanup()
// ============================================================================
void ShadowPass::cleanup(VkDevice device, Allocator &allocator)
{
    if (descriptorPool_)
    {
        vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }
    descriptorSets_.clear();

    for (auto &b : uboBuffers_)
    {
        if (b.buffer)
        {
            vmaUnmapMemory(allocator.getVma(), b.allocation);
            allocator.destroyBuffer(b);
        }
    }
    uboBuffers_.clear();
    uboMapped_.clear();

    for (auto &b : instanceSSBOs_)
    {
        if (b.buffer)
        {
            vmaUnmapMemory(allocator.getVma(), b.allocation);
            allocator.destroyBuffer(b);
        }
    }
    instanceSSBOs_.clear();
    instanceSSBOsMapped_.clear();

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
    if (shadowMapView_)
    {
        vkDestroyImageView(device, shadowMapView_, nullptr);
        shadowMapView_ = VK_NULL_HANDLE;
    }
    if (shadowMap_.image)
    {
        vmaDestroyImage(allocator.getVma(), shadowMap_.image, shadowMap_.allocation);
        shadowMap_.image = VK_NULL_HANDLE;
        shadowMap_.allocation = VK_NULL_HANDLE;
    }

    instanceBuffersBound_ = false;
    currentLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    resolution_ = 0;
}

// ============================================================================
// createImage() - D32_SFLOAT depth image with SAMPLED+DEPTH_ATTACHMENT usage
// ============================================================================
void ShadowPass::createImage(VkDevice device, Allocator &allocator)
{
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
    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(allocator.getVma(), &ii, &aci,
                       &shadowMap_.image, &shadowMap_.allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create shadow map image!");

    shadowMap_.format = format_;
    shadowMap_.extent = {resolution_, resolution_};

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = shadowMap_.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format_;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &vi, nullptr, &shadowMapView_) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create shadow map view!");

    // Cache the view inside the AllocatedImage struct as well, for symmetry.
    shadowMap_.imageView = shadowMapView_;
}

// ============================================================================
// createSampler() - comparison sampler for sampler2DShadow / textureProj()
// ============================================================================
void ShadowPass::createSampler(VkDevice device)
{
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; // hardware PCF (4-tap bilinear comparison)
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Outside-the-frustum samples -> treat as fully lit (depth = 1.0 = far).
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // depth=1 outside frustum = no shadow
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // returns 1.0 if fragmentDepth <= sampledDepth (lit)
    si.maxLod = 0.0f;
    si.minLod = 0.0f;
    si.unnormalizedCoordinates = VK_FALSE;
    if (vkCreateSampler(device, &si, nullptr, &shadowSampler_) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create shadow sampler!");
}

// ============================================================================
// createPipeline() - depth-only graphics pipeline with no fragment shader
// ============================================================================
void ShadowPass::createPipeline(VkDevice device)
{
    // Descriptor set layout: binding 0 = UBO (lightVP), binding 1 = instance SSBO.
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
        throw std::runtime_error("[ShadowPass] Failed to create descriptor set layout!");

    // Pipeline layout: 16-byte push constant (vertex stage only).
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(ShadowPushConstants);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsLayout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create pipeline layout!");

    // Build the depth-only pipeline. We still need a fragment shader that
    // simply discards / does nothing - but the simpler path on Vulkan is to
    // provide a dummy frag (PipelineBuilder requires both). We borrow the
    // existing geometry.frag is overkill (it expects MRT outputs); use a
    // lightweight `shadow.frag` that writes nothing.
    //
    // Note: PipelineBuilder.setShaders requires a fragment shader path, so we
    // ship a minimal `shadow.frag`. With color attachment count = 0 in the
    // pipeline rendering info (no setColorAttachmentFormat), the frag stage
    // is trivially optimized away by the driver.
    std::string shaderDir = SHADER_DIR;
    auto bindingDesc = Vertex::getBindingDescription();
    auto attributeDescs = Vertex::getAttributeDescriptions();

    PipelineBuilder builder;
    pipeline_ = builder
                    .setShaders(device, shaderDir + "/shadow.vert.spv", shaderDir + "/shadow.frag.spv")
                    .setVertexInput(bindingDesc, attributeDescs.data(), static_cast<uint32_t>(attributeDescs.size()))
                    .setInputAssembly()
                    .setViewportDynamic()
                    // Back-face culling + slope-scaled depth bias
                    // ----------------------------------------------------------------
                    // Front-face culling makes a cube's bottom face share depth with the
                    // plane when they touch; with normal bias this guarantees a Peter
                    // Panning bright band. Back-face culling (shadow map stores
                    // light-facing depth) + slope-scaled bias is the classic UE5 / OpenGL
                    // shadow map scheme: coplanar front/back faces still resolve as occluded.
                    .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                    .setDynamicDepthBias()
                    .setMultisampling()
                    .setDepthStencil(true, true, VK_COMPARE_OP_LESS)
                    .setColorBlending(false)
                    .setDepthAttachmentFormat(format_)
                    .build(device, pipelineLayout_);
    builder.cleanupShaderModules(device);
}

// ============================================================================
// createDescriptors()
// ============================================================================
void ShadowPass::createDescriptors(VkDevice device, Allocator &allocator)
{
    // Per-frame UBO (lightViewProj).
    uboBuffers_.resize(MAX_FRAMES_IN_FLIGHT);
    uboMapped_.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        uboBuffers_[i] = allocator.createBuffer(
            sizeof(ShadowVPUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (vmaMapMemory(allocator.getVma(), uboBuffers_[i].allocation, &uboMapped_[i]) != VK_SUCCESS)
            throw std::runtime_error("[ShadowPass] Failed to map UBO!");
    }

    // Dedicated per-frame instance SSBO. Layout matches the
    // Geometry Pass InstanceData exactly (mat4 + vec4 + vec4 = 96 bytes), so
    // shadow.vert can always read InstanceData[] starting at offset zero.
    instanceSSBOs_.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOsMapped_.resize(MAX_FRAMES_IN_FLIGHT);
    const VkDeviceSize instanceSize = 96 * MAX_SHADOW_INSTANCES; // 96 = sizeof(InstanceData)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        instanceSSBOs_[i] = allocator.createBuffer(
            instanceSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (vmaMapMemory(allocator.getVma(), instanceSSBOs_[i].allocation, &instanceSSBOsMapped_[i]) != VK_SUCCESS)
            throw std::runtime_error("[ShadowPass] Failed to map instance SSBO!");
    }

    // Descriptor pool.
    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = static_cast<uint32_t>(ps.size());
    pi.pPoolSizes = ps.data();
    pi.maxSets = MAX_FRAMES_IN_FLIGHT;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create descriptor pool!");

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, dsLayout_);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool_;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts = layouts.data();
    descriptorSets_.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, descriptorSets_.data()) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to allocate descriptor sets!");

    // Write both UBO (binding 0) and instance SSBO (binding 1).
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uboBuffers_[i].buffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(ShadowVPUBO);

        VkDescriptorBufferInfo instInfo{};
        instInfo.buffer = instanceSSBOs_[i].buffer;
        instInfo.offset = 0;
        instInfo.range = instanceSize;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &uboInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &instInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // The SSBO is already bound in createDescriptors;
    // no longer depends on an external bindInstanceBuffers() call.
    instanceBuffersBound_ = true;
}

// ============================================================================
// bindInstanceBuffers() - deprecated: ShadowPass owns its instance SSBO since
// that change, so this function ignores the external buffer and
// exists only so existing callers (Renderer) don't break.
// ============================================================================
void ShadowPass::bindInstanceBuffers(VkDevice /*device*/,
                                     const std::vector<AllocatedBuffer> & /*instanceSSBOs*/,
                                     VkDeviceSize /*bufferSize*/)
{
    // No-op: ShadowPass owns its instance SSBO, bound in createDescriptors.
    // Interface kept so callers don't need changes.
}

// ============================================================================
// getInstanceSSBOMapped() / getInstanceSSBOSize()
// ============================================================================
void *ShadowPass::getInstanceSSBOMapped(uint32_t frameIndex) const
{
    if (frameIndex >= instanceSSBOsMapped_.size())
        return nullptr;
    return instanceSSBOsMapped_[frameIndex];
}

VkDeviceSize ShadowPass::getInstanceSSBOSize() const
{
    return 96 * MAX_SHADOW_INSTANCES; // sizeof(InstanceData) * MAX_SHADOW_INSTANCES
}

// ============================================================================
// updateLightSpaceVP()
// ----------------------------------------------------------------------------
// Light-space view: lookAt from a virtual position offset along -lightDir from
// the camera, looking toward the camera. Orthographic projection covers a
// shadowDistance x shadowDistance square centered on the camera.
//
// Note: this is a single, non-cascaded "main light" shadow (Unity URP default).
// Cascade splits live in the high-end feature tier and are out of scope here.
// ============================================================================
void ShadowPass::updateLightSpaceVP(uint32_t frameIndex,
                                    const glm::vec3 &lightDirection,
                                    const glm::vec3 &cameraPosition,
                                    float shadowDistance)
{
    if (frameIndex >= uboMapped_.size() || !uboMapped_[frameIndex])
        return;

    glm::vec3 dir = glm::normalize(lightDirection);
    if (glm::length(dir) < 1e-4f)
        dir = glm::vec3(0.0f, -1.0f, 0.0f); // safe fallback

    // Stand the light far enough back along -dir so the orthographic frustum
    // brackets the relevant scene volume.
    glm::vec3 lightPos = cameraPosition - dir * shadowDistance;

    // Pick an "up" vector that's not colinear with `dir` (handle vertical light).
    glm::vec3 worldUp = (glm::abs(dir.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    glm::mat4 view = glm::lookAt(lightPos, cameraPosition, worldUp);

    // Symmetric orthographic frustum.
    float half = shadowDistance;
    glm::mat4 proj = glm::ortho(-half, half, -half, half,
                                0.0f, shadowDistance * 2.0f);
    // Vulkan clip-space Y flip (Vulkan: +Y down in NDC, OpenGL/glm: +Y up).
    proj[1][1] *= -1.0f;

    ShadowVPUBO ubo{};
    ubo.lightViewProj = proj * view;
    lastLightVP_ = ubo.lightViewProj;
    std::memcpy(uboMapped_[frameIndex], &ubo, sizeof(ubo));
}

// ============================================================================
// record() - depth-only render pass, calls back into caller for actual draws
// ============================================================================
void ShadowPass::record(VkCommandBuffer cmd, uint32_t frameIndex,
                        const RecordDrawsFn &drawFn)
{
    if (!pipeline_ || !instanceBuffersBound_)
        return; // Not initialized or no instance buffer bound yet - skip silently.

    // (1) Layout transition: current layout -> DEPTH_ATTACHMENT_OPTIMAL.
    VulkanUtils::transitionImageLayout(cmd, shadowMap_.image,
                                       currentLayout_,
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    currentLayout_ = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = shadowMapView_;
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

    // Back-face culling + slope-scaled depth bias
    // ------------------------------------------------------------------
    // Back-face culling stores light-facing depth, so slope-scaled bias is
    // needed to prevent acne on surfaces nearly parallel to the light.
    // Values follow the classic UE5 / OpenGL Cookbook recommendations.
    vkCmdSetDepthBias(cmd, /*constant*/ 1.5f, /*clamp*/ 0.0f, /*slopeScale*/ 2.5f);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1,
                            &descriptorSets_[frameIndex], 0, nullptr);

    // Caller records vkCmdPushConstants + vkCmdBindVertex/IndexBuffer + vkCmdDrawIndexed.
    drawFn(cmd, pipelineLayout_);

    vkCmdEndRendering(cmd);

    // (2) Layout transition: DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY (for sampling
    // in the lighting fragment shader).
    VulkanUtils::transitionImageLayout(cmd, shadowMap_.image,
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    currentLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// ============================================================================
// transitionToReadOnly() - used when no directional caster this frame
// ============================================================================
void ShadowPass::transitionToReadOnly(VkCommandBuffer cmd, bool firstUse)
{
    (void)firstUse;
    if (!shadowMap_.image)
        return;
    if (currentLayout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return; // Already in target layout
    VulkanUtils::transitionImageLayout(cmd, shadowMap_.image,
                                       currentLayout_,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
    currentLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
