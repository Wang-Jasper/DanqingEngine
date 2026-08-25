#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include "core/Allocator.h"

class VulkanContext;
class GBuffer;
class ECSScene;

// Light-source gizmos and collider wireframes: line pipeline + draw recording.
class GizmoRenderer
{
public:
    void createLinePipeline(VkDevice device, VulkanContext &context, Allocator &allocator,
                            VkDescriptorSetLayout geomDSLayout, const std::vector<VkFormat> &colorFormats,
                            VkFormat depthFormat);
    void cleanup(VkDevice device, Allocator &allocator);

    // Records gizmo draws (called inside the geometry pass). Returns the number of
    // vkCmdDraw calls issued, for the Performance Overlay stats.
    uint32_t recordGizmoCommands(VkCommandBuffer cmd,
                                 VkDescriptorSet geomDS, VkViewport &viewport, VkRect2D &scissor,
                                 ECSScene &scene, bool physicsPlaying);

    VkPipelineLayout getPipelineLayout() const { return linePipelineLayout; }

private:
    VkPipeline linePipeline = VK_NULL_HANDLE;
    VkPipelineLayout linePipelineLayout = VK_NULL_HANDLE;
    AllocatedBuffer lineVertexBuffer{};

    uint32_t lineCrossVertexCount = 0;
    uint32_t lineCircleStartVertex = 0;
    uint32_t lineCircleVertexCount = 0;
    uint32_t lineArrowStartVertex = 0;
    uint32_t lineArrowVertexCount = 0;
    uint32_t lineConeStartVertex = 0;
    uint32_t lineConeVertexCount = 0;
    uint32_t lineBoxStartVertex = 0;
    uint32_t lineBoxVertexCount = 0;
    // Unit capsule wireframe: hemisphere ends (r=1) separated by halfHeight=1 along Y.
    // Sphere colliders reuse the lineCircle* segments (unit sphere = 3 rings).
    uint32_t lineCapsuleStartVertex = 0;
    uint32_t lineCapsuleVertexCount = 0;
    uint32_t lineTotalVertexCount = 0;
};
