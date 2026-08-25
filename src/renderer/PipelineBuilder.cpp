// PipelineBuilder.cpp — Builder-pattern implementation: each setXxx() configures
// one pipeline state and returns *this; build() assembles VkGraphicsPipelineCreateInfo.

#include "PipelineBuilder.h"
#include "utils/VulkanUtils.h"
#include <stdexcept>
#include <iostream>

// setShaders(): load SPIR-V, create shader modules, configure stages
PipelineBuilder &PipelineBuilder::setShaders(
    VkDevice device,
    const std::string &vertPath,
    const std::string &fragPath)
{
    auto vertCode = VulkanUtils::readShaderFile(vertPath);
    auto fragCode = VulkanUtils::readShaderFile(fragPath);

    vertModule = VulkanUtils::createShaderModule(device, vertCode);
    fragModule = VulkanUtils::createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    // Entry point name; a module may contain several entry points.
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    shaderStages = {vertStage, fragStage};
    return *this;
}

// setVertexInput(): describe vertex buffer layout (binding stride/per-instance, attribute formats/offsets)
PipelineBuilder &PipelineBuilder::setVertexInput(
    const VkVertexInputBindingDescription &binding,
    const VkVertexInputAttributeDescription *attributes,
    uint32_t attributeCount)
{
    // Copy into builder storage so the pointers stay valid until build().
    storedBinding = binding;
    storedAttributes.assign(attributes, attributes + attributeCount);

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    // Point at internal storage, not caller temporaries.
    vertexInputInfo.pVertexBindingDescriptions = &storedBinding;
    vertexInputInfo.vertexAttributeDescriptionCount = attributeCount;
    vertexInputInfo.pVertexAttributeDescriptions = storedAttributes.data();
    return *this;
}

// setInputAssembly(): how vertices form primitives
PipelineBuilder &PipelineBuilder::setInputAssembly(VkPrimitiveTopology topology)
{
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    // Restart only matters for STRIP topologies; we use LIST, so disabled.
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    return *this;
}

// setViewportDynamic(): viewport/scissor become dynamic state so window resizes
// need no pipeline rebuild; values are set per frame via vkCmdSetViewport/vkCmdSetScissor.
PipelineBuilder &PipelineBuilder::setViewportDynamic()
{
    // Counts only; actual values come from dynamic commands.
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    // pViewports/pScissors are nullptr since these are dynamic states.

    dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicStateInfo.pDynamicStates = dynamicStates.data();
    return *this;
}

// setRasterizer(): configure rasterization state
PipelineBuilder &PipelineBuilder::setRasterizer(
    VkPolygonMode polygonMode,
    VkCullModeFlags cullMode,
    VkFrontFace frontFace)
{
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // depthClampEnable needs a special GPU feature; usually off.
    rasterizer.depthClampEnable = VK_FALSE;
    // Discards all primitives if enabled; only for geometry-only passes (e.g. transform feedback).
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = polygonMode;
    // Line width; >1.0f requires the wideLines device feature.
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cullMode;
    // Front face = CCW triangles; depends on index winding and projection handedness.
    rasterizer.frontFace = frontFace;
    // Depth bias prevents shadow acne; no shadow maps in use, so off.
    rasterizer.depthBiasEnable = VK_FALSE;
    return *this;
}

PipelineBuilder &PipelineBuilder::setLineWidth(float width)
{
    rasterizer.lineWidth = width;
    return *this;
}

// setMultisampling(): MSAA sample count
PipelineBuilder &PipelineBuilder::setMultisampling(VkSampleCountFlagBits samples)
{
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    // Sample shading off (finer-grained MSAA, more expensive).
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = samples;
    return *this;
}

// setDepthStencil(): depth test/write and stencil configuration
PipelineBuilder &PipelineBuilder::setDepthStencil(
    bool depthTest, bool depthWrite, VkCompareOp compareOp)
{
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    // Some transparent passes read depth without writing it.
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    // VK_COMPARE_OP_LESS: closer fragment passes (depth range [0,1], 0 = nearest).
    depthStencil.depthCompareOp = compareOp;
    // Depth bounds test: advanced, usually off.
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    // Stencil test off (used for mirrors/outlines, not here).
    depthStencil.stencilTestEnable = VK_FALSE;
    return *this;
}

// setColorBlending(): how new fragments combine with the framebuffer (opaque or alpha)
PipelineBuilder &PipelineBuilder::setColorBlending(bool enableBlend)
{
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = enableBlend ? VK_TRUE : VK_FALSE;

    if (enableBlend)
    {
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    // logicOp is mutually exclusive with blending; off.
    colorBlending.logicOpEnable = VK_FALSE;
    // Single attachment (the swapchain image).
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    return *this;
}

// setAdditiveBlending(): RGB sums (src*1 + dst*1) so overlapping light proxies
// accumulate; alpha = src*1 + dst*0 (kept, unused).
PipelineBuilder &PipelineBuilder::setAdditiveBlending()
{
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    // Preserve dst alpha: the fullscreen pass writes alpha=1; if Light Volume
    // zeroed it, ImGui would render as transparent with a gray background.
    // src * 0 + dst * 1 = dst.a
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    return *this;
}

// setDynamicDepthBias(): enable depth bias and add DEPTH_BIAS to the dynamic
// states; actual constant/slope are set via vkCmdSetDepthBias() when recording.
// Call AFTER setRasterizer() and setViewportDynamic(), which reset those fields.
PipelineBuilder &PipelineBuilder::setDynamicDepthBias()
{
    rasterizer.depthBiasEnable = VK_TRUE;
    // depthBiasConstantFactor / depthBiasClamp / depthBiasSlopeFactor are
    // ignored when DEPTH_BIAS is dynamic (recorded via vkCmdSetDepthBias).
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // Avoid duplicate insertion if user calls this twice.
    bool already = false;
    for (auto s : dynamicStates)
        if (s == VK_DYNAMIC_STATE_DEPTH_BIAS)
        {
            already = true;
            break;
        }
    if (!already)
        dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);

    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicStateInfo.pDynamicStates = dynamicStates.data();
    return *this;
}
// — Set Dynamic Rendering attachment formats. These are passed via
// VkPipelineRenderingCreateInfo in the pNext chain, replacing VkRenderPass
// attachment descriptions.
PipelineBuilder &PipelineBuilder::setColorAttachmentFormat(VkFormat format)
{
    colorAttachmentFormat = format;
    colorAttachmentFormats = {format};
    return *this;
}

PipelineBuilder &PipelineBuilder::setColorAttachmentFormats(const std::vector<VkFormat> &formats)
{
    colorAttachmentFormats = formats;
    if (!formats.empty())
        colorAttachmentFormat = formats[0];
    return *this;
}

PipelineBuilder &PipelineBuilder::setDepthAttachmentFormat(VkFormat format)
{
    depthAttachmentFormat = format;
    return *this;
}

PipelineBuilder &PipelineBuilder::setNoVertexInput()
{
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    return *this;
}

// build(): assemble all configured state into VkGraphicsPipelineCreateInfo and create the pipeline
VkPipeline PipelineBuilder::build(VkDevice device, VkPipelineLayout layout)
{
    // Ensure we have color formats
    if (colorAttachmentFormats.empty() && colorAttachmentFormat != VK_FORMAT_UNDEFINED)
    {
        colorAttachmentFormats = {colorAttachmentFormat};
    }

    // Setup MRT blend attachments if needed
    if (colorBlendAttachments.empty())
    {
        colorBlendAttachments.resize(colorAttachmentFormats.size(), colorBlendAttachment);
    }
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

    // Dynamic Rendering info
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size());
    renderingInfo.pColorAttachmentFormats = colorAttachmentFormats.data();
    renderingInfo.depthAttachmentFormat = depthAttachmentFormat;

    // --- Graphics pipeline create info: assemble all configured state ---
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // pNext chains the Dynamic Rendering info — this is the key part.
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout = layout;
    // renderPass = VK_NULL_HANDLE: Dynamic Rendering needs no VkRenderPass;
    // attachment formats come from the pNext chain.
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    // No subpasses in Dynamic Rendering; subpass = 0.
    pipelineInfo.subpass = 0;

    // Pipeline cache (VK_NULL_HANDLE) can speed up later pipelines; not used here.
    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create graphics pipeline!");
    }

    std::cout << "[PipelineBuilder] Graphics pipeline created (Dynamic Rendering).\n";
    return pipeline;
}

// cleanupShaderModules(): shader modules aren't needed after the pipeline is built; free them early
void PipelineBuilder::cleanupShaderModules(VkDevice device)
{
    if (vertModule != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, vertModule, nullptr);
        // Null out to prevent double destruction.
        vertModule = VK_NULL_HANDLE;
    }
    if (fragModule != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, fragModule, nullptr);
        fragModule = VK_NULL_HANDLE;
    }
}
