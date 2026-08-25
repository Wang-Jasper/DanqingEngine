#pragma once

#include <entt/entt.hpp>
#include <functional>
#include <algorithm> // std::remove
#include <vector>
#include "ecs/Components.h"

// entt::registry wrapper plus editor selection state.
class ECSScene
{
public:
    entt::registry registry;
    entt::entity selectedEntity = entt::null;

    // Called once before an entity is removed from the registry so upper layers
    // (e.g. Renderer) can release non-ECS resources (e.g. PhysicsWorld bodies).
    // Must not call destroyEntity() from within.
    std::function<void(entt::entity)> onBeforeDestroyEntity;

    // Erases the first occurrence of target from the vector (no-op if absent).
    // Extracted to avoid duplicating the erase-remove idiom in setParent / destroyEntity.
    static void eraseEntity(std::vector<entt::entity> &vec, entt::entity target)
    {
        vec.erase(std::remove(vec.begin(), vec.end(), target), vec.end());
    }

    entt::entity createEntity(const std::string &name)
    {
        auto e = registry.create();
        registry.emplace<NameComponent>(e, NameComponent{name});
        registry.emplace<TransformComponent>(e);
        registry.emplace<HierarchyComponent>(e);
        return e;
    }

    void setParent(entt::entity child, entt::entity parent)
    {
        auto &childHC = registry.get<HierarchyComponent>(child);
        if (childHC.parent != entt::null && registry.valid(childHC.parent))
        {
            auto &oldParentHC = registry.get<HierarchyComponent>(childHC.parent);
            eraseEntity(oldParentHC.children, child);
        }
        childHC.parent = parent;
        if (parent != entt::null && registry.valid(parent))
        {
            auto &parentHC = registry.get<HierarchyComponent>(parent);
            parentHC.children.push_back(child);
        }
    }

    // Destroys an entity and all its children recursively.
    void destroyEntity(entt::entity e)
    {
        if (!registry.valid(e))
            return;
        if (selectedEntity == e)
            selectedEntity = entt::null;

        if (registry.all_of<HierarchyComponent>(e))
        {
            auto children = registry.get<HierarchyComponent>(e).children; // copy
            for (auto child : children)
            {
                destroyEntity(child);
            }
        }
        if (registry.all_of<HierarchyComponent>(e))
        {
            auto &hc = registry.get<HierarchyComponent>(e);
            if (hc.parent != entt::null && registry.valid(hc.parent))
            {
                auto &parentHC = registry.get<HierarchyComponent>(hc.parent);
                eraseEntity(parentHC.children, e);
            }
        }

        if (onBeforeDestroyEntity)
        {
            onBeforeDestroyEntity(e);
        }

        registry.destroy(e);
    }

    // Clears the whole scene; fires onBeforeDestroyEntity for every entity so
    // external resources (physics world, etc.) are released in sync. Do not
    // bypass this with registry.clear().
    void destroyAll()
    {
        selectedEntity = entt::null;
        if (onBeforeDestroyEntity)
        {
            auto all = registry.view<entt::entity>();
            for (auto e : all)
            {
                onBeforeDestroyEntity(e);
            }
        }
        registry.clear();
    }

    std::vector<entt::entity> getRootEntities()
    {
        std::vector<entt::entity> roots;
        auto view = registry.view<HierarchyComponent>();
        for (auto e : view)
        {
            auto &hc = view.get<HierarchyComponent>(e);
            if (hc.parent == entt::null)
            {
                roots.push_back(e);
            }
        }
        return roots;
    }
};
