// ============================================================================
// Contact constraint: normal plus two-tangent friction for one manifold.
// Holds only the manifold index and per-point solver caches; point data is not
// copied.
// ============================================================================
#pragma once

#include "physics/Constraint.h"
#include <glm/glm.hpp>

class ContactConstraint : public Constraint
{
public:
    // Per-point solver constants, recomputed each frame in prepare.
    // Field order mirrors the old resolveCollisions PointCtx to keep numerics identical.
    struct PointCache
    {
        bool skip = false;
        float denomN = 0.0f;
        float denomT1 = 0.0f;
        float denomT2 = 0.0f;
        float restitutionTargetVel = 0.0f;
    };

    // Interact via the manifold index; a raw manifold pointer would dangle when
    // manifolds_ reallocates.
    explicit ContactConstraint(int manifoldIndex) : manifoldIndex_(manifoldIndex) {}

    // Access to the point cache (used by PhysicsWorld's SI loop; removable once
    // the migration is complete).
    PointCache &point(int k) { return pointCache_[k]; }
    const PointCache &point(int k) const { return pointCache_[k]; }

    int manifoldIndex() const { return manifoldIndex_; }

    // Max initial contact impact speed across this manifold's points, computed in
    // prepare; PhysicsWorld uses it for neighbor wake-up (body-level side effects
    // stay out of Constraint).
    float maxInitialImpact() const { return maxInitialImpact_; }

    // Constraint interface.
    void prepare(PhysicsWorld &world, float dt) override;
    void warmStart(PhysicsWorld &world) override;
    void solveVelocity(PhysicsWorld &world) override;
    float solvePosition(PhysicsWorld &world) override;

private:
    int manifoldIndex_ = -1;
    PointCache pointCache_[4];
    float maxInitialImpact_ = 0.0f;
};
