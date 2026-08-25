#pragma once

// RenderBVH — render-side dynamic AABB tree making CPU frustum culling O(log N)
// (shares the physics impl, separate instances). Rebuilt fully each frame:
// world AABBs change almost every frame; rebuild cost is negligible at N < 10k.

#include <entt/entt.hpp>
#include <vector>

#include "ecs/Components.h"
#include "physics/DynamicAABBTree.h"
#include "utils/Frustum.h"

class RenderBVH
{
public:
    // Rebuild from the scene registry: for each <MeshComponent, TransformComponent>,
    // compute the loose world AABB from mesh local AABB + worldMatrix as a leaf.
    // userData = index into leafEntities_ (mapped back to entt::entity on query).
    void rebuildFromScene(entt::registry &reg);

    // Frustum query: calls visitor(entity) for every leaf conservatively
    // intersecting the frustum. queryFrustum prunes by p-vertices, then leaves
    // are re-checked with FrustumCulling::testAABB (equivalent to brute-force).
    template <typename Visitor>
    void queryFrustum(const FrustumCulling::Frustum &frustum, Visitor &&visitor) const
    {
        DynamicAABBTree::QueryPlane planes[6];
        for (int i = 0; i < 6; ++i)
        {
            planes[i].n = frustum.planes[i].n;
            planes[i].d = frustum.planes[i].d;
        }
        tree_.queryFrustum(planes, [&](int userData)
                           {
            if (userData < 0 || userData >= static_cast<int>(leafEntities_.size()))
                return;
            visitor(leafEntities_[userData]); });
    }

    // Leaf / node counts (for the overlay).
    int leafCount() const { return static_cast<int>(leafEntities_.size()); }
    int nodeCount() const { return tree_.nodeCount(); }

private:
    DynamicAABBTree tree_;
    std::vector<entt::entity> leafEntities_; // userData -> entity mapping
};

// Defined at the end of the header to stay inline; short enough, can move to a .cpp later.
inline void RenderBVH::rebuildFromScene(entt::registry &reg)
{
    tree_.clear();
    leafEntities_.clear();

    auto view = reg.view<MeshComponent, TransformComponent>();
    leafEntities_.reserve(view.size_hint());

    for (auto e : view)
    {
        auto &mc = view.template get<MeshComponent>(e);
        auto &tc = view.template get<TransformComponent>(e);
        if (!mc.mesh)
            continue;

        glm::vec3 wmin, wmax;
        FrustumCulling::transformLocalAABBToWorld(
            tc.worldMatrix, mc.mesh->boundsCenter, mc.mesh->boundsExtents,
            wmin, wmax);

        AABB leafAABB;
        leafAABB.min = wmin;
        leafAABB.max = wmax;
        int userData = static_cast<int>(leafEntities_.size());
        leafEntities_.push_back(e);
        tree_.insert(leafAABB, userData);
    }
}
