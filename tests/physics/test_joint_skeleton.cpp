// ============================================================================
// test_joint_skeleton.cpp — joint skeleton headers compile check
// ----------------------------------------------------------------------------
// Parses all four joint headers once so external users can include them
// unchanged; no joint is instantiated, so the assert(false) stubs never fire.
// ============================================================================
#include "test_framework.h"
#include "physics/joints/FixedJoint.h"
#include "physics/joints/HingeJoint.h"
#include "physics/joints/SliderJoint.h"
#include "physics/joints/BallJoint.h"

PHYS_TEST(JointSkeleton, AllHeadersCompile)
{
    // Size checks only, no instantiation (avoids the assert(false) stubs).
    PHYS_CHECK(sizeof(FixedJoint) > 0, "FixedJoint class is defined");
    PHYS_CHECK(sizeof(HingeJoint) > 0, "HingeJoint class is defined");
    PHYS_CHECK(sizeof(SliderJoint) > 0, "SliderJoint class is defined");
    PHYS_CHECK(sizeof(BallJoint) > 0, "BallJoint class is defined");
}
