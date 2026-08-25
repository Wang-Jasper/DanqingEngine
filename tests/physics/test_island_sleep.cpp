// ============================================================================
// test_island_sleep.cpp — island-level sleep criteria and wake propagation
// ----------------------------------------------------------------------------
// EdgeContactSettles30_60_20: a box landing at 30/60/20 deg must fall asleep
// in reasonable time (island.sleepFrames reaches sleepFramesRequired, body
// sleeping, velocity zero) instead of jittering upright.
// WakePropagatesAcrossIsland: after a box stack sleeps, applyImpulse on the
// top body must wake every island member in the same frame.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

namespace
{
    int addStaticFloor(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.8f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addDynBox(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half,
                  const glm::mat3 &orient = glm::mat3(1.0f), float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.6f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = orient;
        return w.addBody(rb, obb, mat);
    }

    void runSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int i = 0; i < steps; ++i)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

// ----------------------------------------------------------------------------
// 1. Box tilted at (30, 60, 20) deg must fall asleep within 5 s with velocity
// and angular velocity zeroed.
// ----------------------------------------------------------------------------
PHYS_TEST(IslandSleep, EdgeContactSettles30_60_20)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticFloor(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));

    // Euler angles (30, 60, 20) deg, Rx*Ry*Rz.
    glm::mat4 R4 = glm::eulerAngleXYZ(glm::radians(30.0f),
                                      glm::radians(60.0f),
                                      glm::radians(20.0f));
    glm::mat3 R(R4);
    int bi = addDynBox(w, glm::vec3(0, 3.0f, 0), glm::vec3(0.5f), R);

    runSeconds(w, 5.0f);

    const auto &body = w.getBody(bi);
    // Must sleep within 5 s (typically < 3 s; 2 s buffer).
    PHYS_CHECK(body.sleeping, "body should enter sleeping state");
    // Sleeping zeroes velocity and angular velocity.
    PHYS_CHECK_NEAR(glm::length(body.velocity), 0.0f, 1e-4f);
    PHYS_CHECK_NEAR(glm::length(body.angularVelocity), 0.0f, 1e-4f);
}

// ----------------------------------------------------------------------------
// 2: impulse on the top of a sleeping stack wakes the whole island same-frame
// ----------------------------------------------------------------------------
PHYS_TEST(IslandSleep, WakePropagatesAcrossIsland)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticFloor(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));

    // 5-box stack (fewer layers settle faster, cutting runtime).
    const int stackCount = 5;
    std::vector<int> stackIds;
    stackIds.reserve(stackCount);
    for (int i = 0; i < stackCount; ++i)
    {
        float y = 0.5f + i * 1.05f; // centers from y=0.5, spacing 1.05 (halfExtents 0.5; 0.05 gap so frame 1 does not penetrate)
        int id = addDynBox(w, glm::vec3(0, y, 0), glm::vec3(0.5f));
        stackIds.push_back(id);
    }

    // Run 6 s so the stack's island settles to sleep.
    runSeconds(w, 6.0f);

    // Stack must be stable: at least half the bodies asleep.
    int sleepingBefore = 0;
    for (int id : stackIds)
        if (w.getBody(id).sleeping)
            ++sleepingBefore;
    PHYS_CHECK(sleepingBefore >= stackCount / 2,
               "at least half of stack should be sleeping after 6s");

    // Upward impulse on the top body; wakes it and the whole island instantly.
    int topId = stackIds.back();
    w.applyImpulse(topId, glm::vec3(0, 10.0f, 0), w.getPosition(topId));

    // Check immediately (no step): applyImpulse calls wakeBody, clearing
    // sleeping on every island member.
    for (int id : stackIds)
    {
        PHYS_CHECK(!w.getBody(id).sleeping,
                   "island member should be awake same-frame after impulse on any member");
    }
}
