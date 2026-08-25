// SliderJoint.h — prismatic joint skeleton: two bodies translate freely along
// a fixed axis with orientation locked. Uses: pistons, drawers, elevator rails.
#pragma once

#include "physics/Constraint.h"
#include <cassert>

class SliderJoint : public Constraint
{
public:
    void prepare(PhysicsWorld & /*world*/, float /*dt*/) override
    {
        assert(false && "SliderJoint::prepare not implemented (Phase 5.5 skeleton)");
    }
    void warmStart(PhysicsWorld & /*world*/) override
    {
        assert(false && "SliderJoint::warmStart not implemented (Phase 5.5 skeleton)");
    }
    void solveVelocity(PhysicsWorld & /*world*/) override
    {
        assert(false && "SliderJoint::solveVelocity not implemented (Phase 5.5 skeleton)");
    }
    float solvePosition(PhysicsWorld & /*world*/) override
    {
        assert(false && "SliderJoint::solvePosition not implemented (Phase 5.5 skeleton)");
        return 0.0f;
    }
};
