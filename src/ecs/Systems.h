#pragma once

#include <entt/entt.hpp>
#include "ecs/Components.h"
#include "scene/Light.h"

namespace Systems
{

    // Computes world matrices for all entities: roots first, then children recursively.
    // A plain inline recursive function is used instead of a std::function lambda to
    // avoid type-erasure / heap allocation in deep hierarchies; interface and numeric
    // behavior are identical.
    inline void updateTransformRecursive(entt::registry &reg,
                                         entt::entity e,
                                         const glm::mat4 &parentWorld)
    {
        auto &tc = reg.get<TransformComponent>(e);
        tc.worldMatrix = parentWorld * tc.getLocalMatrix();

        if (reg.all_of<HierarchyComponent>(e))
        {
            auto &hc = reg.get<HierarchyComponent>(e);
            for (auto child : hc.children)
            {
                if (reg.valid(child) && reg.all_of<TransformComponent>(child))
                {
                    updateTransformRecursive(reg, child, tc.worldMatrix);
                }
            }
        }
    }

    inline void updateTransforms(entt::registry &reg)
    {
        // Find all root entities (no parent or parent == null).
        auto view = reg.view<TransformComponent>();
        for (auto e : view)
        {
            bool isRoot = true;
            if (reg.all_of<HierarchyComponent>(e))
            {
                auto &hc = reg.get<HierarchyComponent>(e);
                if (hc.parent != entt::null && reg.valid(hc.parent))
                {
                    isRoot = false;
                }
            }
            if (isRoot)
            {
                updateTransformRecursive(reg, e, glm::mat4(1.0f));
            }
        }
    }

    // Gathers all lights into their respective vectors.
    inline void gatherLights(entt::registry &reg, std::vector<PointLight> &outLights)
    {
        outLights.clear();
        auto view = reg.view<PointLightComponent, TransformComponent>();
        for (auto e : view)
        {
            auto &plc = view.get<PointLightComponent>(e);
            auto &tc = view.get<TransformComponent>(e);
            PointLight pl;
            pl.position = glm::vec3(tc.worldMatrix[3]);
            pl.radius = plc.radius;
            pl.color = plc.color;
            pl.intensity = plc.intensity;
            pl.shadowSlot = -1; // No shadow by default; PointShadowSelectionSystem fills the slot by top-N.
            pl._pad[0] = pl._pad[1] = pl._pad[2] = 0.0f;
            outLights.push_back(pl);
        }
    }

    inline void gatherDirLights(entt::registry &reg, std::vector<DirectionalLight> &outLights)
    {
        outLights.clear();
        auto view = reg.view<DirectionalLightComponent, TransformComponent>();
        for (auto e : view)
        {
            auto &dlc = view.get<DirectionalLightComponent>(e);
            auto &tc = view.get<TransformComponent>(e);
            DirectionalLight dl;
            // Direction derived from the rotation's -Z axis (forward).
            glm::vec3 forward = glm::normalize(glm::vec3(tc.worldMatrix * glm::vec4(0, 0, -1, 0)));
            dl.direction = forward;
            dl.color = dlc.color;
            dl.intensity = dlc.intensity;
            dl._pad = 0.0f;
            outLights.push_back(dl);
        }
    }

    inline void gatherSpotLights(entt::registry &reg, std::vector<SpotLight> &outLights)
    {
        outLights.clear();
        auto view = reg.view<SpotLightComponent, TransformComponent>();
        for (auto e : view)
        {
            auto &slc = view.get<SpotLightComponent>(e);
            auto &tc = view.get<TransformComponent>(e);
            SpotLight sl;
            sl.position = glm::vec3(tc.worldMatrix[3]);
            sl.radius = slc.radius;
            glm::vec3 forward = glm::normalize(glm::vec3(tc.worldMatrix * glm::vec4(0, 0, -1, 0)));
            sl.direction = forward;
            sl.intensity = slc.intensity;
            sl.color = slc.color;
            sl.innerCos = glm::cos(glm::radians(slc.innerAngle));
            sl.outerCos = glm::cos(glm::radians(slc.outerAngle));
            sl.shadowSlot = -1; // No shadow by default; SpotShadowSelectionSystem fills the slot by top-N.
            sl._pad[0] = sl._pad[1] = 0.0f;
            outLights.push_back(sl);
        }
    }

} // namespace Systems
