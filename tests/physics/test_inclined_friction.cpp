// ============================================================================
// test_inclined_friction.cpp — inclined-plane friction benchmark
// ----------------------------------------------------------------------------
// Static and kinetic inclined-plane cases (as in Box2D Testbed / Jolt / PEEL):
// mu > tan(theta) must hold the box still, mu < tan(theta) must let it slide
// at a = g(sinθ - mu*cosθ). Public API only, no per-scene special cases; the
// slope is a rotated Static box and the box free-falls onto it.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    // Rotate theta about +Z: +Y tilts toward +X, giving the slope angle.
    glm::mat3 rotZ(float theta)
    {
        return glm::mat3(glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0, 0, 1)));
    }

    // Static inclined slab, geometric center at origin, rotated theta about Z.
    // Must be thick enough (halfY >= 1.0) to avoid tunneling while the box
    // slides, and wide enough (halfX >= 15) to keep the box on for 2 s.
    int addInclinedPlane(PhysicsWorld &w, float theta, float friction,
                         const glm::vec3 &origin = glm::vec3(0, 0, 0),
                         const glm::vec3 &half = glm::vec3(15.0f, 1.0f, 10.0f))
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = friction;
        mat.restitution = 0.0f;
        OBB obb;
        obb.center = origin;
        obb.halfExtents = half;
        obb.orientation = rotZ(theta);
        return w.addBody(rb, obb, mat);
    }

    // Place the box just above the slope so it free-falls onto the slope center.
    int addBoxOnSlope(PhysicsWorld &w, float theta, float friction,
                      float planeHalfY = 1.0f)
    {
        // Slope normal in world space is R * (0,1,0)
        glm::mat3 R = rotZ(theta);
        glm::vec3 n = R * glm::vec3(0, 1, 0);

        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.friction = friction;
        mat.restitution = 0.0f;
        OBB obb;
        // Box center just above the slope surface: origin + n*planeHalfY plus
        // box half-height (0.25) and a 0.1 buffer.
        const float boxHalf = 0.25f;
        obb.center = n * (planeHalfY + boxHalf + 0.1f);
        obb.halfExtents = glm::vec3(boxHalf);
        obb.orientation = R;
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
// 1. mu > tan(theta): static friction must hold the box (theta=20 deg,
//    tan≈0.364, mu=0.8 comfortably above).
// ----------------------------------------------------------------------------
PHYS_TEST(InclinedFriction, StaticHoldsOnSteepSlope)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    const float theta = glm::radians(20.0f);
    const float mu = 0.8f;

    addInclinedPlane(w, theta, mu);
    int box = addBoxOnSlope(w, theta, mu);

    // Settle onto the slope first (2 s is enough).
    stepSec(w, 2.0f);
    glm::vec3 pSettle = w.getPosition(box);

    // Observe 5 s more; mu > tan(theta) must prevent sliding.
    stepSec(w, 5.0f);
    glm::vec3 pFinal = w.getPosition(box);
    glm::vec3 v = w.getBody(box).velocity;

    // Position drift must stay small (< 0.15 m).
    glm::vec3 drift = pFinal - pSettle;
    PHYS_CHECK(glm::length(drift) < 0.15f,
               "box with mu>tan(theta) should not slide noticeably");
    // Velocity must be near zero.
    PHYS_CHECK(glm::length(v) < 0.3f,
               "box at rest on high-friction slope");
}

// ----------------------------------------------------------------------------
// 2. mu < tan(theta): box must accelerate down the slope. Theory:
//    a = 9.81*(sin30 - 0.1*cos30) ≈ 4.055 m/s^2. Assert only "clearly slides"
//    plus a plausible speed range, not the exact acceleration.
// ----------------------------------------------------------------------------
PHYS_TEST(InclinedFriction, KineticSlidesDownShallowSlope)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    const float theta = glm::radians(30.0f);
    const float mu = 0.1f;

    addInclinedPlane(w, theta, mu);
    int box = addBoxOnSlope(w, theta, mu);

    // Settle 0.5 s (0.1 m free fall takes ~0.14 s; leaves sleep-to-wake margin).
    stepSec(w, 0.5f);
    glm::vec3 pStart = w.getPosition(box);

    // Then slide 1.0 s (1.5 s total observation).
    stepSec(w, 1.0f);
    glm::vec3 pEnd = w.getPosition(box);
    glm::vec3 v = w.getBody(box).velocity;

    glm::vec3 disp = pEnd - pStart;

    // Displacement must be noticeable (> 0.3 m); theory gives ~2 m over 1 s
    // but the 0.5 s settle phase dissipates energy, so keep a loose lower bound.
    PHYS_CHECK(glm::length(disp) > 0.3f,
               "box should slide noticeably down slope");
    // Slide direction projects to -X in world space.
    PHYS_CHECK(disp.x < -0.2f,
               "slide direction is toward -X (down the slope)");

    // Speed magnitude plausible: theory gives |v| ≈ 4 m/s after 1 s of sliding;
    // assert a generous engineering range so an impact spike cannot pass.
    PHYS_CHECK(glm::length(v) > 1.0f,
               "box velocity should grow as it slides");
    PHYS_CHECK(glm::length(v) < 10.0f,
               "velocity not absurdly large (friction still acts, no tunneling)");

    // Anti-regression: the box must stay above the slab (no tunneling). Slab is
    // 2 m thick at 30 deg; "above" means n*pEnd > -planeHalfY.
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0, 0, 1)));
    glm::vec3 n = R * glm::vec3(0, 1, 0);
    float signedDist = glm::dot(n, pEnd);
    PHYS_CHECK(signedDist > 0.9f,
               "box stays on top of the slope (no tunneling through slab)");
}
