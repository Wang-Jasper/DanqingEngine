#include "renderer/GizmoRenderer.h"
#include "renderer/PipelineBuilder.h"
#include "core/VulkanContext.h"
#include "ecs/ECSScene.h"
#include "ecs/Components.h"
#include "physics/PhysicsComponents.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cmath>

// Strip scale from a TRS matrix (keep rotation + translation) so directional/spot
// gizmos don't deform from entity scale. Normalizes the first 3 columns (no shear assumed).
static glm::mat4 stripScale(const glm::mat4 &m)
{
    glm::mat4 r = m;
    for (int c = 0; c < 3; c++)
    {
        float len = glm::length(glm::vec3(r[c]));
        if (len > 0.0f)
            r[c] /= len;
    }
    return r;
}
void GizmoRenderer::createLinePipeline(VkDevice device, VulkanContext &context, Allocator &allocator,
                                       VkDescriptorSetLayout geomDSLayout,
                                       const std::vector<VkFormat> &colorFormats,
                                       VkFormat depthFormat)
{
    struct LineVertex
    {
        glm::vec3 pos;
        glm::vec3 color;
    };
    std::vector<LineVertex> vertices;
    const int segments = 64;
    const float radius = 1.0f;
    glm::vec3 warmYellow = {1.0f, 1.0f, 0.6f};
    glm::vec3 skyBlue = {0.4f, 0.7f, 1.0f};
    glm::vec3 orange = {1.0f, 0.6f, 0.2f};

    // Shared crosshair (6 vertices)
    float crossLen = 0.15f;
    vertices.push_back({{-crossLen, 0, 0}, warmYellow});
    vertices.push_back({{crossLen, 0, 0}, warmYellow});
    vertices.push_back({{0, -crossLen, 0}, warmYellow});
    vertices.push_back({{0, crossLen, 0}, warmYellow});
    vertices.push_back({{0, 0, -crossLen}, warmYellow});
    vertices.push_back({{0, 0, crossLen}, warmYellow});
    lineCrossVertexCount = 6;

    lineCircleStartVertex = static_cast<uint32_t>(vertices.size());
    auto addCircle = [&](int axis0, int axis1, glm::vec3 col)
    {
        for (int i = 0; i < segments; i++)
        {
            float a0 = 2.0f * 3.14159265f * i / segments;
            float a1 = 2.0f * 3.14159265f * (i + 1) / segments;
            glm::vec3 p0(0), p1(0);
            p0[axis0] = cosf(a0) * radius;
            p0[axis1] = sinf(a0) * radius;
            p1[axis0] = cosf(a1) * radius;
            p1[axis1] = sinf(a1) * radius;
            vertices.push_back({p0, col});
            vertices.push_back({p1, col});
        }
    };
    addCircle(0, 1, warmYellow);
    addCircle(0, 2, warmYellow);
    addCircle(1, 2, warmYellow);
    lineCircleVertexCount = segments * 2;

    // Directional light: top disc + 8 parallel rays
    lineArrowStartVertex = static_cast<uint32_t>(vertices.size());
    float discR = 0.4f;
    float rayLen = 1.2f;
    float tipLen = 0.15f;
    float tipW = 0.06f;
    int discSegs = 32;

    for (int i = 0; i < discSegs; i++)
    {
        float a0 = 2.0f * 3.14159265f * i / discSegs;
        float a1 = 2.0f * 3.14159265f * (i + 1) / discSegs;
        vertices.push_back({{cosf(a0) * discR, sinf(a0) * discR, 0}, skyBlue});
        vertices.push_back({{cosf(a1) * discR, sinf(a1) * discR, 0}, skyBlue});
    }

    int numRays = 8;
    for (int i = 0; i < numRays; i++)
    {
        float angle = 2.0f * 3.14159265f * i / numRays;
        float cx = cosf(angle) * discR;
        float cy = sinf(angle) * discR;
        glm::vec3 start = {cx, cy, 0.0f};
        glm::vec3 end = {cx, cy, -rayLen};

        vertices.push_back({start, skyBlue});
        vertices.push_back({end, skyBlue});

        float nx = cosf(angle) * tipW;
        float ny = sinf(angle) * tipW;
        glm::vec3 tipA = end + glm::vec3(-nx, -ny, tipLen);
        glm::vec3 tipB = end + glm::vec3(nx, ny, tipLen);
        vertices.push_back({end, skyBlue});
        vertices.push_back({tipA, skyBlue});
        vertices.push_back({end, skyBlue});
        vertices.push_back({tipB, skyBlue});
    }
    lineArrowVertexCount = static_cast<uint32_t>(vertices.size()) - lineArrowStartVertex;

    // Spot light: 4 cone lines + bottom ring
    lineConeStartVertex = static_cast<uint32_t>(vertices.size());
    float coneH = 1.0f;
    float coneR = 1.0f;
    glm::vec3 coneCorners[] = {
        {coneR, 0, -coneH}, {-coneR, 0, -coneH}, {0, coneR, -coneH}, {0, -coneR, -coneH}};
    for (auto &c : coneCorners)
    {
        vertices.push_back({{0, 0, 0}, orange});
        vertices.push_back({c, orange});
    }
    for (int i = 0; i < segments; i++)
    {
        float a0 = 2.0f * 3.14159265f * i / segments;
        float a1 = 2.0f * 3.14159265f * (i + 1) / segments;
        vertices.push_back({{cosf(a0) * coneR, sinf(a0) * coneR, -coneH}, orange});
        vertices.push_back({{cosf(a1) * coneR, sinf(a1) * coneR, -coneH}, orange});
    }
    lineConeVertexCount = static_cast<uint32_t>(vertices.size()) - lineConeStartVertex;

    // Box wireframe (unit cube [-0.5, 0.5]^3, 12 edges = 24 vertices)
    lineBoxStartVertex = static_cast<uint32_t>(vertices.size());
    glm::vec3 boxGreen = {0.0f, 1.0f, 0.0f};
    glm::vec3 bv[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};
    int boxEdges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Back face
        {4, 5},
        {5, 6},
        {6, 7},
        {7, 4}, // Front face
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7} // Connecting front and back
    };
    for (auto &e : boxEdges)
    {
        vertices.push_back({bv[e[0]], boxGreen});
        vertices.push_back({bv[e[1]], boxGreen});
    }
    lineBoxVertexCount = static_cast<uint32_t>(vertices.size()) - lineBoxStartVertex;

    // Capsule wireframe (unit convention: long axis Y, r=1, halfHeight=1): 3 rings at
    // each sphere end plus 4 side lines along ±X/±Z. Rendered as three draws — top
    // sphere, bottom sphere, side lines — because a non-uniform scale(r, hh, r) would
    // deform the rings; the side lines alone are stretched with scale(r, hh, r).
    lineCapsuleStartVertex = static_cast<uint32_t>(vertices.size());
    glm::vec3 capsuleGreen = {0.0f, 1.0f, 0.0f};
    glm::vec3 capsuleSideDirs[4] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1}};
    for (auto &d : capsuleSideDirs)
    {
        glm::vec3 pTop = d;
        pTop.y = 1.0f;
        glm::vec3 pBot = d;
        pBot.y = -1.0f;
        vertices.push_back({pTop, capsuleGreen});
        vertices.push_back({pBot, capsuleGreen});
    }
    lineCapsuleVertexCount = static_cast<uint32_t>(vertices.size()) - lineCapsuleStartVertex;

    lineTotalVertexCount = static_cast<uint32_t>(vertices.size());

    if (lineVertexBuffer.buffer == VK_NULL_HANDLE)
    {
        lineVertexBuffer = allocator.createBufferWithStaging(
            context, vertices.data(), sizeof(LineVertex) * vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &geomDSLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &linePipelineLayout);

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(LineVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].binding = 0;
    attrs[0].location = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(LineVertex, pos);
    attrs[1].binding = 0;
    attrs[1].location = 1;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(LineVertex, color);

    std::string shaderDir = SHADER_DIR;
    PipelineBuilder builder;
    linePipeline = builder
                       .setShaders(device, shaderDir + "/line.vert.spv", shaderDir + "/line.frag.spv")
                       .setVertexInput(binding, attrs, 2)
                       .setInputAssembly(VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
                       .setViewportDynamic()
                       .setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                       .setLineWidth(2.5f)
                       .setMultisampling()
                       .setDepthStencil(true, false, VK_COMPARE_OP_LESS_OR_EQUAL)
                       .setColorBlending(false)
                       .setColorAttachmentFormats(colorFormats)
                       .setDepthAttachmentFormat(depthFormat)
                       .build(device, linePipelineLayout);
    builder.cleanupShaderModules(device);

    std::cout << "[GizmoRenderer] Line pipeline created (" << lineTotalVertexCount << " vertices).\n";
}

void GizmoRenderer::cleanup(VkDevice device, Allocator &allocator)
{
    if (linePipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, linePipeline, nullptr);
        vkDestroyPipelineLayout(device, linePipelineLayout, nullptr);
        allocator.destroyBuffer(lineVertexBuffer);
        linePipeline = VK_NULL_HANDLE;
        linePipelineLayout = VK_NULL_HANDLE;
    }
}

uint32_t GizmoRenderer::recordGizmoCommands(VkCommandBuffer cmd,
                                            VkDescriptorSet geomDS, VkViewport &viewport, VkRect2D &scissor,
                                            ECSScene &scene, bool physicsPlaying)
{
    if (!linePipeline)
        return 0;

    // Don't draw gizmos in Play mode (matches Unity)
    if (physicsPlaying)
        return 0;

    auto &reg = scene.registry;
    bool pipelineBound = false;
    uint32_t drawCount = 0; // vkCmdDraw calls issued this frame

    // Bind pipeline and shared state only once
    auto ensureBound = [&]()
    {
        if (!pipelineBound)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    linePipelineLayout, 0, 1, &geomDS, 0, nullptr);
            VkBuffer vbufs[] = {lineVertexBuffer.buffer};
            VkDeviceSize voffsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, voffsets);
            pipelineBound = true;
        }
    };

    // === Collider wireframes: drawn only for the selected entity (like Unity) ===
    // Prefer CollidersComponent (Box/Sphere/Capsule/ConvexHull); fall back to the
    // legacy BoxColliderComponent single-box path.
    if (scene.selectedEntity != entt::null && reg.valid(scene.selectedEntity) && reg.all_of<TransformComponent>(scene.selectedEntity))
    {
        auto &tc = reg.get<TransformComponent>(scene.selectedEntity);

        bool hasMulti = reg.all_of<CollidersComponent>(scene.selectedEntity) && !reg.get<CollidersComponent>(scene.selectedEntity).list.empty();

        if (hasMulti)
        {
            ensureBound();
            auto &cc = reg.get<CollidersComponent>(scene.selectedEntity);
            for (auto &cd : cc.list)
            {
                glm::mat4 localM = glm::translate(glm::mat4(1.0f), cd.localPosition) * glm::mat4_cast(cd.localRotation);
                glm::mat4 base = tc.worldMatrix * localM;

                switch (cd.shape.type)
                {
                case ShapeType::Box:
                {
                    glm::mat4 m = glm::scale(base, cd.shape.halfExtents * 2.0f);
                    vkCmdPushConstants(cmd, linePipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &m);
                    vkCmdDraw(cmd, lineBoxVertexCount, 1, lineBoxStartVertex, 0);
                    ++drawCount;
                    break;
                }
                case ShapeType::Sphere:
                {
                    // Reuse the 3 lineCircle rings (unit sphere)
                    glm::mat4 m = glm::scale(base, glm::vec3(cd.shape.radius));
                    vkCmdPushConstants(cmd, linePipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &m);
                    vkCmdDraw(cmd, lineCircleVertexCount * 3, 1, lineCircleStartVertex, 0);
                    ++drawCount;
                    break;
                }
                case ShapeType::Capsule:
                {
                    float r = cd.shape.radius;
                    float hh = cd.shape.halfHeight;
                    glm::mat4 mTop = glm::translate(base, glm::vec3(0.0f, hh, 0.0f));
                    mTop = glm::scale(mTop, glm::vec3(r));
                    vkCmdPushConstants(cmd, linePipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mTop);
                    vkCmdDraw(cmd, lineCircleVertexCount * 3, 1, lineCircleStartVertex, 0);
                    ++drawCount;
                    glm::mat4 mBot = glm::translate(base, glm::vec3(0.0f, -hh, 0.0f));
                    mBot = glm::scale(mBot, glm::vec3(r));
                    vkCmdPushConstants(cmd, linePipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mBot);
                    vkCmdDraw(cmd, lineCircleVertexCount * 3, 1, lineCircleStartVertex, 0);
                    ++drawCount;
                    // scale(r, hh, r) stretches the unit side lines to the real length
                    glm::mat4 mSide = glm::scale(base, glm::vec3(r, hh, r));
                    vkCmdPushConstants(cmd, linePipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mSide);
                    vkCmdDraw(cmd, lineCapsuleVertexCount, 1, lineCapsuleStartVertex, 0);
                    ++drawCount;
                    break;
                }
                case ShapeType::ConvexHull:
                {
                    // Simplification: no per-edge hull wireframe (would need a dynamic
                    // buffer from hullCache); draw the hull's local AABB instead to
                    // indicate a hull is present.
                    if (cd.shape.hullCache && !cd.shape.hullCache->vertices.empty())
                    {
                        glm::vec3 mn = cd.shape.hullCache->localAABBMin;
                        glm::vec3 mx = cd.shape.hullCache->localAABBMax;
                        glm::vec3 center = 0.5f * (mn + mx);
                        glm::vec3 extent = 0.5f * (mx - mn); // Half extents
                        glm::mat4 m = glm::translate(base, center);
                        m = glm::scale(m, extent * 2.0f);
                        vkCmdPushConstants(cmd, linePipelineLayout,
                                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &m);
                        vkCmdDraw(cmd, lineBoxVertexCount, 1, lineBoxStartVertex, 0);
                        ++drawCount;
                    }
                    break;
                }
                case ShapeType::Compound:
                default:
                    // Compound isn't drawn here; the ECS layer expands it into multiple ColliderDescs
                    break;
                }
            }
        }
        else if (reg.all_of<BoxColliderComponent>(scene.selectedEntity))
        {
            auto &bc = reg.get<BoxColliderComponent>(scene.selectedEntity);
            ensureBound();
            glm::mat4 boxModel = tc.worldMatrix;
            boxModel = glm::translate(boxModel, bc.center);
            boxModel = glm::scale(boxModel, bc.halfExtents * 2.0f);
            vkCmdPushConstants(cmd, linePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &boxModel);
            vkCmdDraw(cmd, lineBoxVertexCount, 1, lineBoxStartVertex, 0);
            ++drawCount;
        }
    }
    // === Selected light gizmo ===
    if (scene.selectedEntity == entt::null)
        return drawCount;
    if (!reg.valid(scene.selectedEntity))
        return drawCount;
    if (!reg.all_of<TransformComponent>(scene.selectedEntity))
        return drawCount;

    auto sel = scene.selectedEntity;
    auto &tc = reg.get<TransformComponent>(sel);
    bool isAnyLight = reg.all_of<PointLightComponent>(sel) || reg.all_of<DirectionalLightComponent>(sel) || reg.all_of<SpotLightComponent>(sel);

    if (!isAnyLight)
        return drawCount;

    glm::vec3 pos = glm::vec3(tc.worldMatrix[3]);

    ensureBound();

    glm::mat4 crossModel = glm::translate(glm::mat4(1.0f), pos);
    vkCmdPushConstants(cmd, linePipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &crossModel);
    vkCmdDraw(cmd, lineCrossVertexCount, 1, 0, 0);
    ++drawCount;

    if (reg.all_of<PointLightComponent>(sel))
    {
        float r = reg.get<PointLightComponent>(sel).radius;
        glm::mat4 circleModel = glm::translate(glm::mat4(1.0f), pos);
        circleModel = glm::scale(circleModel, glm::vec3(r));
        vkCmdPushConstants(cmd, linePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &circleModel);
        vkCmdDraw(cmd, lineCircleVertexCount * 3, 1, lineCircleStartVertex, 0);
        ++drawCount;
    }

    if (reg.all_of<DirectionalLightComponent>(sel))
    {
        glm::mat4 arrowModel = stripScale(tc.worldMatrix);
        vkCmdPushConstants(cmd, linePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &arrowModel);
        vkCmdDraw(cmd, lineArrowVertexCount, 1, lineArrowStartVertex, 0);
        ++drawCount;
    }

    if (reg.all_of<SpotLightComponent>(sel))
    {
        auto &slc = reg.get<SpotLightComponent>(sel);
        float coneRadius = slc.radius * glm::tan(glm::radians(slc.outerAngle));
        glm::mat4 coneModel = stripScale(tc.worldMatrix);
        coneModel = glm::scale(coneModel, glm::vec3(coneRadius, coneRadius, slc.radius));
        vkCmdPushConstants(cmd, linePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &coneModel);
        vkCmdDraw(cmd, lineConeVertexCount, 1, lineConeStartVertex, 0);
        ++drawCount;
    }

    return drawCount;
}
