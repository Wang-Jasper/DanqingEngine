// HingeJoint.h — hinge joint skeleton: two bodies rotate freely about a fixed
// world axis, all other DOFs locked. Uses: door hinges, rocker arms, simple wheels.
#pragma once

#include "physics/Constraint.h"
#include <cassert>

class HingeJoint : public Constraint
{
public:
    void prepare(PhysicsWorld & /*world*/, float /*dt*/) override
    {
        assert(false && "HingeJoint::prepare not implemented (Phase 5.5 skeleton)");
    }
    void warmStart(PhysicsWorld & /*world*/) override
    {
        assert(false && "HingeJoint::warmStart not implemented (Phase 5.5 skeleton)");
    }
    void solveVelocity(PhysicsWorld & /*world*/) override
    {
        assert(false && "HingeJoint::solveVelocity not implemented (Phase 5.5 skeleton)");
    }
    float solvePosition(PhysicsWorld & /*world*/) override
    {
        assert(false && "HingeJoint::solvePosition not implemented (Phase 5.5 skeleton)");
        return 0.0f;
    }
};
