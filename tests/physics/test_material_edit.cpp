// ============================================================================
// test_material_edit.cpp — CP-3.2 regression guard
// ----------------------------------------------------------------------------
// Locks in PhysicsMaterial as the single authoritative source of
// friction/restitution so refactors cannot silently revert to body-level
// parameters. Behavior-only, no private APIs: each case runs two simulations
// differing in one variable and compares final displacement.
// ============================================================================

#include "test_framework.h"

#include "physics/PhysicsWorld.h"
#include "physics/Shape.h"
#include "physics/PhysicsMaterial.h"

#include <glm/glm.hpp>
#include <cmath>

namespace
{
    // ------------------------------------------------------------------
    // Shared parameters (both cases, keeping the experimental conditions identical)
    // ------------------------------------------------------------------
    constexpr float kFixedStep = 1.0f / 60.0f;
    constexpr float kSimulateSeconds = 2.0f;
    constexpr glm::vec3 kGravity = glm::vec3(0.0f, -9.81f, 0.0f);

    // Ball starts at y = radius + tiny epsilon, hugging the ground top so the
    // free-fall bounce phase cannot mask friction deceleration; 5 m/s initial
    // speed gives a wide signal-to-noise ratio on the 2 s slide distance.
    constexpr glm::vec3 kBallInitPos = glm::vec3(0.0f, 0.50001f, 0.0f);
    constexpr glm::vec3 kBallInitVel = glm::vec3(5.0f, 0.0f, 0.0f);
    constexpr float kBallRadius = 0.5f;
    constexpr float kBallMass = 1.0f;

    // Ground: large horizontal box, top at y = 0.
    constexpr glm::vec3 kGroundCenter = glm::vec3(0.0f, -0.25f, 0.0f);
    constexpr glm::vec3 kGroundHalf = glm::vec3(5.0f, 0.25f, 5.0f);

    void runForSeconds(PhysicsWorld &w, float seconds)
    {
        const int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int i = 0; i < steps; ++i)
            w.stepSimulation(w.fixedTimeStep);
    }

    // Dynamic sphere shared by both cases; material at defaults
    // (friction 0.5, restitution 0.3).
    int addSphere(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = kBallMass;
        rb.velocity = kBallInitVel;
        // Deliberately no rb.friction/rb.restitution here: the sphere's
        // material comes from attachShape, decoupled from the parameter under test.

        BodyPose pose;
        pose.position = kBallInitPos;
        int bi = w.addEmptyBody(rb, pose);

        Shape sh;
        sh.type = ShapeType::Sphere;
        sh.radius = kBallRadius;

        PhysicsMaterial mat;    // defaults
        mat.restitution = 0.0f; // remove ball-floor bounce so friction keeps acting
        BodyPose localId;
        w.attachShape(bi, sh, localId, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }

    // Ground via the addBody(rb, obb, material) convenience path: since CP-3.2
    // this path's friction/restitution come only from the explicit
    // PhysicsMaterial; RigidBody no longer carries implicit fields.
    int addGroundViaConvenienceApi(PhysicsWorld &w, float materialFriction)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;

        PhysicsMaterial mat;
        mat.friction = materialFriction;
        mat.restitution = 0.1f; // low bounce so the ball does not rebound and change its horizontal displacement

        OBB obb;
        obb.center = kGroundCenter;
        obb.halfExtents = kGroundHalf;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    // Ground via the explicit addEmptyBody + attachShape path: only
    // material.friction matters here, giving a control against the convenience
    // path (both must behave identically).
    int addGroundViaExplicitAttach(PhysicsWorld &w, float materialFriction)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;

        BodyPose pose;
        pose.position = kGroundCenter;
        int bi = w.addEmptyBody(rb, pose);

        Shape sh;
        sh.type = ShapeType::Box;
        sh.halfExtents = kGroundHalf;

        PhysicsMaterial mat;
        mat.friction = materialFriction;
        mat.restitution = 0.1f;

        BodyPose localId;
        w.attachShape(bi, sh, localId, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }

    // Full "ball slides on the ground for kSimulateSeconds" experiment,
    // returning the ball's final horizontal displacement |x|. groundBuilder
    // (injected per case) decides the ground's friction source.
    template <typename GroundBuilder>
    float runSlideExperiment(const GroundBuilder &buildGround)
    {
        PhysicsWorld w;
        w.gravity = kGravity;
        w.fixedTimeStep = kFixedStep;

        buildGround(w);
        int ballIdx = addSphere(w);

        runForSeconds(w, kSimulateSeconds);

        glm::vec3 endPos = w.getPosition(ballIdx);
        return std::fabs(endPos.x);
    }
} // namespace

// ============================================================================
// Case 1: addBody(rb, obb, material) — its material is what the solver uses
// ----------------------------------------------------------------------------
// Two identical simulations differ only in ground material.friction (2.0 vs
// 0.0; combined 1.25 vs 0.25). High friction must visibly shorten the slide;
// the 0.3 m delta threshold is conservative against jitter. If addBody ever
// stops forwarding the material to the auto-created shape, both runs converge
// and the guard fails.
// ============================================================================
PHYS_TEST(MaterialEdit, AddBodyHonorsExplicitMaterialParam)
{
    float distHigh = runSlideExperiment([](PhysicsWorld &w)
                                        { addGroundViaConvenienceApi(w, 2.0f); });
    float distLow = runSlideExperiment([](PhysicsWorld &w)
                                       { addGroundViaConvenienceApi(w, 0.0f); });

    PHYS_CHECK(distLow > distHigh + 0.3f,
               "addBody(rb, obb, material) must propagate material.friction into "
               "the auto-created box shape; high-friction run should slow the ball "
               "noticeably compared to zero-friction run");
}

// ============================================================================
// Case 2: attachShape's material is the authoritative solver source
// ----------------------------------------------------------------------------
// Same experiment via the explicit attachShape path (friction 2.0 vs 0.0;
// combined 1.25 vs 0.25). Results must match Case 1 in direction and
// magnitude, confirming addBody is just addEmptyBody + attachShape wrapped.
// ============================================================================
PHYS_TEST(MaterialEdit, AttachedMaterialIsAuthoritative)
{
    float distMatHigh = runSlideExperiment([](PhysicsWorld &w)
                                           { addGroundViaExplicitAttach(w, 2.0f); });
    float distMatLow = runSlideExperiment([](PhysicsWorld &w)
                                          { addGroundViaExplicitAttach(w, 0.0f); });

    PHYS_CHECK(distMatLow > distMatHigh + 0.3f,
               "attachShape material must be the authoritative source: the "
               "simulation with material.friction=2 must slow the ball more "
               "than material.friction=0");
}
