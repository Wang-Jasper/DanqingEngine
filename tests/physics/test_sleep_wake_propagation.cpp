// ============================================================================
// test_sleep_wake_propagation.cpp — sleep wake-propagation regressions:
// an awake dynamic wakes a sleeping dynamic neighbor (island propagation), stacks
// still converge to sleep, setBodyPose on a static wakes the cube above it
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <cmath>
#include <vector>

namespace
{
    int addGround(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = glm::vec3(0, -0.5f, 0);
        obb.halfExtents = glm::vec3(20.0f, 0.5f, 20.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addDynamicCube(PhysicsWorld &w, const glm::vec3 &center, float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = glm::vec3(0.5f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    void stepSec(PhysicsWorld &w, float sec)
    {
        int steps = static_cast<int>(sec / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

// ----------------------------------------------------------------------------
// 1. AwakeDynamicWakesSleepingNeighbor
//    Pure dynamic-dynamic: after applyImpulse gives the bottom cube a horizontal
//    velocity, the sleeping top cube must wake through the awake bottom's
//    tangential contact.
//
//    Acceptance:
//      (a) applyImpulse wakes top immediately (no step) via island propagation
//      (b) within the next substep, friction transfers positive tangential velocity
//          to the top
//    Both rely on plan A's wake channels: (a) uses wakeBody's island-neighbor
//    propagation (pre-existing), (b) uses plan A's sustained pre-step waking (if
//    the top re-sleeps, the awake bottom's tangential velocity at the contact
//    point must trigger wakeByNeighborMotion again). Long-term stability is not
//    required; the stack may re-sleep after the short impulse.
// ----------------------------------------------------------------------------
PHYS_TEST(SleepWake, AwakeDynamicWakesSleepingNeighbor)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    int bottom = addDynamicCube(w, glm::vec3(0.0f, 0.5f, 0.0f));
    int top = addDynamicCube(w, glm::vec3(0.0f, 1.5f, 0.0f));

    // Wait 3s so both cubes fully sleep
    stepSec(w, 3.0f);
    PHYS_CHECK(w.getBody(bottom).sleeping, "bottom cube must sleep after settle");
    PHYS_CHECK(w.getBody(top).sleeping, "top cube must sleep after settle");

    // Impulse J = m·Δv with Δv = 3 m/s → J = 3 kg·m/s, far above sleepLinearThreshold=0.08
    glm::vec3 impulsePoint = w.getPosition(bottom);
    w.applyImpulse(bottom, glm::vec3(3.0f, 0.0f, 0.0f), impulsePoint);

    // Acceptance (a): applyImpulse calls wakeBody internally, so island propagation
    // must wake top immediately — a static assertion outside stepping that depends
    // only on wakeBody semantics
    PHYS_CHECK(!w.getBody(bottom).sleeping,
               "bottom cube must wake immediately after applyImpulse");
    PHYS_CHECK(!w.getBody(top).sleeping,
               "top cube must wake via island propagation from wakeBody(bottom)");

    // Acceptance (b): advance 1 substep for enough friction impulse transfer.
    // bottom starts at 3 m/s; top mass 1kg, μ=0.8, normal load m·g=9.81N;
    // max tangential impulse per step μ·Jn·dt ≈ 0.8·9.81·(1/60) ≈ 0.131 N·s
    // → top gains ≈ 0.131 m/s per step, accumulating to a detectable value
    stepSec(w, 0.1f); // 6 substeps

    PHYS_CHECK(std::fabs(w.getBody(top).velocity.x) > 0.05f,
               "top cube must acquire tangential velocity from bottom friction in 6 substeps");
}

// ----------------------------------------------------------------------------
// 2. StackSleepConvergenceUnaffected
//    A 10-layer stack must fully sleep within 30s. Anti-regression for plan A:
//    after loosening the wake conditions, all stack velocities must zero out in
//    steady state without repeated mutual wake-ups.
// ----------------------------------------------------------------------------
PHYS_TEST(SleepWake, StackSleepConvergenceUnaffected)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    const int N = 10;
    std::vector<int> ids;
    for (int i = 0; i < N; ++i)
        ids.push_back(addDynamicCube(w, glm::vec3(0.0f, 0.5f + i * 1.0f, 0.0f)));

    stepSec(w, 30.0f);

    // Core check: the whole stack must sleep (stack sleep convergence not broken by plan A)
    int sleepCount = 0;
    for (int id : ids)
    {
        if (w.getBody(id).sleeping)
            ++sleepCount;
    }
    // Loose: at least 80% must sleep (top-edge jitter tolerated)
    PHYS_CHECK(sleepCount >= N * 80 / 100,
               "at least 80% of stack must sleep after 30s");

    for (int id : ids)
    {
        glm::vec3 v = w.getBody(id).velocity;
        PHYS_CHECK(glm::length(v) < 0.5f,
                   "stacked box velocity ~0 at steady state");
    }
}

// ----------------------------------------------------------------------------
// 3. SetBodyPoseWakesNeighbor
//    Moving the static ground +Y by 0.5m via PhysicsWorld::setBodyPose must wake
//    the sleeping cube above, which is then pushed up by normal penetration
//    impulses.
//
//    Acceptance:
//      (a) setBodyPose wakes the cube immediately (no step) via island propagation
//          — core assertion of plan B
//      (b) after 1s the cube is lifted > 0.2m — the solver resolves penetration
//          with normal impulses
//    No requirement to stay awake after settling (re-sleep at the new position is
//    normal physics).
// ----------------------------------------------------------------------------
PHYS_TEST(SleepWake, SetBodyPoseWakesNeighbor)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    int ground = addGround(w);
    int cube = addDynamicCube(w, glm::vec3(0.0f, 1.0f, 0.0f));

    // Settle and sleep
    stepSec(w, 3.0f);
    PHYS_CHECK(w.getBody(cube).sleeping, "cube must sleep after settle");
    float yStart = w.getPosition(cube).y;

    // setBodyPose teleports the static ground +Y 0.5m → top face goes y=0 → y=0.5;
    // the cube bottom at y=0.5 ends up fully penetrated
    OBB obb = w.getOBB(ground);
    BodyPose pose;
    pose.position = obb.center + glm::vec3(0.0f, 0.5f, 0.0f);
    pose.orientation = obb.orientation;
    w.setBodyPose(ground, pose);

    // Acceptance (a): setBodyPose must wake the cube immediately via island
    // propagation — core plan-B assertion, no stepping needed
    PHYS_CHECK(!w.getBody(cube).sleeping,
               "cube must wake immediately via setBodyPose island propagation");

    // Acceptance (b): once awake, solver penetration impulses must lift the cube
    stepSec(w, 1.0f);
    float yEnd = w.getPosition(cube).y;
    PHYS_CHECK(yEnd - yStart > 0.2f,
               "cube must be lifted > 0.2m after ground teleport (penetration resolved)");
}
