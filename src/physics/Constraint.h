// ============================================================================
// Polymorphic base shared by contact and joint constraints. The island-scoped
// solver loop consumes `Constraint*` without knowing the concrete type.
// Built each frame at the end of detectCollisions; kept after resolveCollisions
// so warm-start impulses (stored in the manifold) carry into the next frame.
// ============================================================================
#pragma once

class PhysicsWorld;

class Constraint
{
public:
    virtual ~Constraint() = default;

    // Called once per step before solving: compute invariants (rA/rB, denominator,
    // restitutionTarget).
    virtual void prepare(PhysicsWorld &world, float dt) = 0;

    // Apply the accumulated impulse inherited from the previous frame (zero if none).
    virtual void warmStart(PhysicsWorld &world) = 0;

    // One velocity-solving pass of the SI iteration; the outer iteration loop lives
    // in PhysicsWorld.
    virtual void solveVelocity(PhysicsWorld &world) = 0;

    // One split-impulse position-correction pass; returns the max penetration
    // correction produced this round (the outer loop exits early below slop).
    virtual float solvePosition(PhysicsWorld &world) = 0;
};
