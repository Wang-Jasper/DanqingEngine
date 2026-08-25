// PipelineBuilder.h — Builder-pattern helper for Vulkan graphics pipelines:
// chained setters replace ~10 hand-filled CreateInfo structs. Uses Dynamic
// Rendering (Vulkan 1.3): no VkRenderPass; formats go in VkPipelineRenderingCreateInfo.

#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

// All setXxx() methods return PipelineBuilder& for chaining; build() creates the pipeline.
class PipelineBuilder
{
public:
    // Public API — chained configuration

    // Load and compile the vertex/fragment shaders.
    // vertPath/fragPath: SPIR-V binary paths (built by glslc).
    // Creates VkShaderModules and configures VkPipelineShaderStageCreateInfo.
    PipelineBuilder &setShaders(VkDevice device,
                                const std::string &vertPath,
                                const std::string &fragPath);

    // Vertex input: memory layout of vertex data.
    // binding: buffer stride (per-vertex or per-instance).
    // attributes: format and offset of each attribute (position, normal, color, uv).
    // attributeCount: number of attributes.
    // Copies binding/attributes internally so pointers stay valid until build().
    PipelineBuilder &setVertexInput(
        const VkVertexInputBindingDescription &binding,
        const VkVertexInputAttributeDescription *attributes,
        uint32_t attributeCount);

    // Input assembly: how vertices form primitives. Default: TRIANGLE_LIST (3 vertices per triangle).
    PipelineBuilder &setInputAssembly(VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // Dynamic viewport/scissor: only counts are set here; real values are applied
    // per frame via vkCmdSetViewport/vkCmdSetScissor, so window resizes need no
    // pipeline rebuild.
    PipelineBuilder &setViewportDynamic();

    // Rasterizer: converts primitives to fragments.
    // polygonMode: FILL / LINE / POINT.
    // cullMode: BACK_BIT / FRONT_BIT / NONE.
    // frontFace: which winding is front (COUNTER_CLOCKWISE).
    PipelineBuilder &setRasterizer(VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL,
                                   VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT,
                                   VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE);

    // Set line width (requires the wideLines device feature).
    PipelineBuilder &setLineWidth(float width);

    // MSAA: default VK_SAMPLE_COUNT_1_BIT (off).
    PipelineBuilder &setMultisampling(VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

    // Depth/stencil: depthTest on/off, depthWrite on/off,
    // compareOp (LESS = closer fragment passes).
    PipelineBuilder &setDepthStencil(bool depthTest = true, bool depthWrite = true,
                                     VkCompareOp compareOp = VK_COMPARE_OP_LESS);

    // Color blending: false = opaque overwrite, true = alpha blend
    // (srcAlpha * src + (1-srcAlpha) * dst).
    PipelineBuilder &setColorBlending(bool enableBlend = false);

    // Additive blend (src*1 + dst*1) for Light Volume rendering: contributions of
    // many lights accumulate. srcAlpha/dstAlpha stay ONE/ZERO so alpha is unaffected.
    PipelineBuilder &setAdditiveBlending();

    // Enable depth bias (shadow map acne avoidance): sets depthBiasEnable and
    // adds DEPTH_BIAS to the dynamic states; set via vkCmdSetDepthBias when recording.
    PipelineBuilder &setDynamicDepthBias();

    // Dynamic Rendering attachment formats (replaces VkRenderPass): passed via
    // VkPipelineRenderingCreateInfo in the pNext chain.
    // Color formats must match the swapchain image format.
    PipelineBuilder &setColorAttachmentFormat(VkFormat format);
    // Multiple color attachment formats (MRT, for the G-Buffer Geometry Pass).
    PipelineBuilder &setColorAttachmentFormats(const std::vector<VkFormat> &formats);
    // Depth attachment format (must match the depth image format).
    PipelineBuilder &setDepthAttachmentFormat(VkFormat format);
    // No vertex input (fullscreen triangle; vertices generated in the shader).
    PipelineBuilder &setNoVertexInput();

    // Build the final pipeline.
    // layout: pipeline layout (descriptor sets and push constants).
    // Returns: the created VkPipeline handle.
    VkPipeline build(VkDevice device, VkPipelineLayout layout);

    // Destroy shader modules after build(); they are no longer needed.
    void cleanupShaderModules(VkDevice device);

private:
    // Private members — state needed to create the pipeline

    // Shader stage array (vertex + fragment = 2 stages).
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    // Shader module handles (destroyed manually after build).
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    // Vertex input state: memory layout of vertex data (attributes, formats, offsets).
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    // Input assembly state: how vertices form primitives.
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    // Viewport state: viewport/scissor counts (values set via dynamic state).
    VkPipelineViewportStateCreateInfo viewportState{};
    // Rasterization state: fill mode, culling, front face, etc.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    // Multisample state: MSAA configuration.
    VkPipelineMultisampleStateCreateInfo multisampling{};
    // Depth/stencil test state.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    // Color blend attachment state (one per color attachment).
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments; // for MRT
    // Global color blend state (covers all attachments).
    VkPipelineColorBlendStateCreateInfo colorBlending{};

    // Dynamic state list (viewport + scissor).
    std::vector<VkDynamicState> dynamicStates;
    // Dynamic state create info.
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};

    // Dynamic Rendering attachment formats (replaces VkRenderPass attachment descriptions).
    VkFormat colorAttachmentFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkFormat> colorAttachmentFormats; // for MRT
    VkFormat depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    // Stored vertex input descriptions so the pointers in
    // VkPipelineVertexInputStateCreateInfo stay valid until build() — the
    // caller's temporaries may be gone by then.
    VkVertexInputBindingDescription storedBinding{};
    std::vector<VkVertexInputAttributeDescription> storedAttributes;
};
