// ============================================================================
// test_static_stability.cpp — sleep-criterion static-stability checks (TDD):
// tilted cubes must NOT sleep in unstable postures (edge/corner standing), while
// flat boxes and upright stacks must still sleep
// ============================================================================

#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
    // Floor matching the SceneSetup parameters exactly
    int addDefaultFloor(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.3f;
        mat.friction = 0.8f;
        OBB obb;
        obb.center = glm::vec3(0.0f, -1.5f, 0.0f);     // Transform(0,-1) + BoxCollider.center(0,-0.5) = (0,-1.5)
        obb.halfExtents = glm::vec3(5.0f, 0.5f, 5.0f); // scale(10,1,10) * halfExtents(0.5,0.5,0.5)
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    // Cube matching the SceneSetup parameters exactly
    int addDefaultCube(PhysicsWorld &w, const glm::vec3 &pos, const glm::mat3 &orient)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.3f;
        mat.friction = 0.5f;
        OBB obb;
        obb.center = pos;
        obb.halfExtents = glm::vec3(0.5f);
        obb.orientation = orient;
        return w.addBody(rb, obb, mat);
    }

    float bestAxisAlignWithY(const glm::mat3 &orient)
    {
        float ax = std::fabs(orient[0][1]);
        float ay = std::fabs(orient[1][1]);
        float az = std::fabs(orient[2][1]);
        return std::max(ax, std::max(ay, az));
    }

    void runSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int i = 0; i < steps; ++i)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

// ----------------------------------------------------------------------------
// Case 1: 30°/60°/20° edge standing — must not be allowed. Currently sleeps stably
// at bestAlign ≈ 0.93 (edge tilted 21.6°) — that is the bug
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, EdgeTiltedBoxMustNotSleep)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addDefaultFloor(w);

    glm::mat4 R4 = glm::eulerAngleXYZ(glm::radians(30.0f),
                                      glm::radians(60.0f),
                                      glm::radians(20.0f));
    int bi = addDefaultCube(w, glm::vec3(0, 3.0f, 0), glm::mat3(R4));

    runSeconds(w, 10.0f);

    const auto &body = w.getBody(bi);
    const OBB obb = w.getOBB(bi);
    float align = bestAxisAlignWithY(obb.orientation);

    // Legal final states: face down (high alignment) or still moving (engine still
    // seeking a stable pose). Illegal: sleeping at 0.577 ~ 0.95 alignment (edge/corner balance)
    bool settledFlat = align > 0.95f;
    bool stillMoving = !body.sleeping;
    PHYS_CHECK(settledFlat || stillMoving,
               "Edge-tilted cube must not sleep in unstable posture");
}

// ----------------------------------------------------------------------------
// Case 2: corner pointing down (bestAlign ≈ 0.577) — must not be allowed
// (45°, 35.264°, 0°) Euler XYZ puts local Y at world Y = 0.577 (standard corner-down pose)
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, CornerStandingMustNotSleep)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addDefaultFloor(w);

    // Standard corner-down pose: rotate X=45° then Z=45° so the local (1,1,1)/√3 axis points -Y
    glm::mat4 R4 = glm::eulerAngleXYZ(glm::radians(45.0f),
                                      glm::radians(0.0f),
                                      glm::radians(45.0f));
    int bi = addDefaultCube(w, glm::vec3(0, 3.0f, 0), glm::mat3(R4));

    runSeconds(w, 10.0f);

    const auto &body = w.getBody(bi);
    const OBB obb = w.getOBB(bi);
    float align = bestAxisAlignWithY(obb.orientation);

    bool settledFlat = align > 0.95f;
    bool stillMoving = !body.sleeping;
    PHYS_CHECK(settledFlat || stillMoving,
               "Corner-standing cube must not sleep in unstable posture");
}

// ----------------------------------------------------------------------------
// Case 3 (regression guard): upright flat cube must still sleep (new criterion
// must not break normal scenes)
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, FlatBoxStillSleeps)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addDefaultFloor(w);
    int bi = addDefaultCube(w, glm::vec3(0, 2.0f, 0), glm::mat3(1.0f));

    runSeconds(w, 5.0f);

    const auto &body = w.getBody(bi);
    PHYS_CHECK(body.sleeping,
               "Flat box must enter sleep within 5 seconds");
    PHYS_CHECK_NEAR(glm::length(body.velocity), 0.0f, 1e-4f);
    PHYS_CHECK_NEAR(glm::length(body.angularVelocity), 0.0f, 1e-4f);
}

// ----------------------------------------------------------------------------
// Case 4 (regression guard): 3-layer upright stack — all must sleep within 10s
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, StackStillSleeps)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addDefaultFloor(w);

    // 3 layers, centers y = 0/1.0/2.0 (half edge 0.5, zero initial gap so contact
    // starts on the first frame)
    std::vector<int> ids;
    for (int i = 0; i < 3; ++i)
    {
        float y = 0.0f + 1.0f * i;
        ids.push_back(addDefaultCube(w, glm::vec3(0, y, 0), glm::mat3(1.0f)));
    }

    runSeconds(w, 10.0f);

    int sleepingCount = 0;
    for (int id : ids)
        if (w.getBody(id).sleeping)
            ++sleepingCount;
    PHYS_CHECK(sleepingCount == 3,
               "All 3 stacked boxes must enter sleep within 10 seconds");
}

// ----------------------------------------------------------------------------
// Case 5 (extended regression): batch of rotations must not stand on edge —
// PLAN 5.3.6.5 angles plus boundary angles; all must be face-down or still moving
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, MultipleRotationsNoEdgeStandings)
{
    struct Sample
    {
        float ex, ey, ez;
    };
    // Selected angles (degrees):
    //   30/60/20 — the exact user-reported angles
    //   45/0/0   — single-axis 45° edge-contact boundary
    //   45/45/0  — two-axis 45° corner-contact boundary
    //   20/70/35 — PLAN-required random sample
    //   60/30/10, 15/75/40, 50/20/55, 35/65/25 — broader irregular-angle coverage
    Sample samples[] = {
        {30.0f, 60.0f, 20.0f},
        {45.0f, 0.0f, 0.0f},
        {45.0f, 45.0f, 0.0f},
        {20.0f, 70.0f, 35.0f},
        {60.0f, 30.0f, 10.0f},
        {15.0f, 75.0f, 40.0f},
        {50.0f, 20.0f, 55.0f},
        {35.0f, 65.0f, 25.0f},
    };

    int bad = 0;
    for (const auto &s : samples)
    {
        PhysicsWorld w;
        w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
        w.fixedTimeStep = 1.0f / 60.0f;

        addDefaultFloor(w);
        glm::mat4 R4 = glm::eulerAngleXYZ(glm::radians(s.ex),
                                          glm::radians(s.ey),
                                          glm::radians(s.ez));
        int bi = addDefaultCube(w, glm::vec3(0, 3.0f, 0), glm::mat3(R4));

        runSeconds(w, 10.0f);

        const auto &body = w.getBody(bi);
        const OBB obb = w.getOBB(bi);
        float align = bestAxisAlignWithY(obb.orientation);
        bool settledFlat = align > 0.95f;
        bool stillMoving = !body.sleeping;
        if (!(settledFlat || stillMoving))
        {
            ++bad;
            std::printf("  [MultiRot BAD] rot=(%.0f,%.0f,%.0f) align=%.3f cy=%.3f sleep=%d |v|=%.3f |w|=%.3f\n",
                        s.ex, s.ey, s.ez, align, obb.center.y,
                        body.sleeping ? 1 : 0,
                        glm::length(body.velocity), glm::length(body.angularVelocity));
        }
    }
    PHYS_CHECK(bad == 0,
               "no rotation sample should result in unstable sleep posture");
}

// ----------------------------------------------------------------------------
// Case 6 (fall-time assertion): edge-contact cube must tip to face-down within a
// reasonable time — guards the second zeroing-path fix of Phase 5.3.6.
//
// Pre-fix timing: the 30/60/20 cube first edge-contacts at t≈0.9s, bounces up to
// ~1.5s, then spends 1.9-3.6s (1.7s!) slowly tipping, face-down at t≈3.7s. The
// slow tip was caused by applyRestingStaticFriction zeroing angular velocity
// every frame.
// Post-fix: edge contact at 1.9s, face-down at 2.3s (tips within 0.4s).
//
// Assert t=2.3s gives bestAlign > 0.95 — 1.4s tighter than the pre-fix 3.7s, with
// margin over the fixed real fall time.
// ----------------------------------------------------------------------------
PHYS_TEST(StaticStability, EdgeTiltedMustFallFast)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    w.fixedTimeStep = 1.0f / 60.0f;

    addDefaultFloor(w);

    glm::mat4 R4 = glm::eulerAngleXYZ(glm::radians(30.0f),
                                      glm::radians(60.0f),
                                      glm::radians(20.0f));
    int bi = addDefaultCube(w, glm::vec3(0, 3.0f, 0), glm::mat3(R4));

    runSeconds(w, 2.3f);

    const OBB obb = w.getOBB(bi);
    float align = bestAxisAlignWithY(obb.orientation);
    PHYS_CHECK(align > 0.95f,
               "edge-tilted cube must reach face-down posture within 2.3s");
}
