// BallJoint.h — spherical joint skeleton: a point on each body stays aligned
// with free relative rotation. Uses: character skeletons, rope anchors, gimbals.
#pragma once

#include "physics/Constraint.h"
#include <cassert>

class BallJoint : public Constraint
{
public:
    void prepare(PhysicsWorld & /*world*/, float /*dt*/) override
    {
        assert(false && "BallJoint::prepare not implemented (Phase 5.5 skeleton)");
    }
    void warmStart(PhysicsWorld & /*world*/) override
    {
        assert(false && "BallJoint::warmStart not implemented (Phase 5.5 skeleton)");
    }
    void solveVelocity(PhysicsWorld & /*world*/) override
    {
        assert(false && "BallJoint::solveVelocity not implemented (Phase 5.5 skeleton)");
    }
    float solvePosition(PhysicsWorld & /*world*/) override
    {
        assert(false && "BallJoint::solvePosition not implemented (Phase 5.5 skeleton)");
        return 0.0f;
    }
};
