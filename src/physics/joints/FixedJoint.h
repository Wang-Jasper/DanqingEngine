// FixedJoint.h — weld joint skeleton: locks the relative position and
// orientation of the two bodies. Compiles only; all solve* assert(false) until
// the joint system plan lands.
#pragma once

#include "physics/Constraint.h"
#include <cassert>

class FixedJoint : public Constraint
{
public:
    void prepare(PhysicsWorld & /*world*/, float /*dt*/) override
    {
        assert(false && "FixedJoint::prepare not implemented (Phase 5.5 skeleton)");
    }
    void warmStart(PhysicsWorld & /*world*/) override
    {
        assert(false && "FixedJoint::warmStart not implemented (Phase 5.5 skeleton)");
    }
    void solveVelocity(PhysicsWorld & /*world*/) override
    {
        assert(false && "FixedJoint::solveVelocity not implemented (Phase 5.5 skeleton)");
    }
    float solvePosition(PhysicsWorld & /*world*/) override
    {
        assert(false && "FixedJoint::solvePosition not implemented (Phase 5.5 skeleton)");
        return 0.0f;
    }
};
