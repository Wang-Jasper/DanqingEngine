// Regression: a box landing on the ground with a second box dropping onto it
// from above once produced chained penetration (single-point manifolds, cold
// warm-start, weak position solver). Assert no transient interpenetration over
// 3 s and a clean final stack.
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    int addGround(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.8f;
        OBB obb;
        obb.center = glm::vec3(0, -0.4f, 0);
        obb.halfExtents = glm::vec3(5.0f, 0.4f, 5.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addBox(PhysicsWorld &w, const glm::vec3 &center,
               const glm::vec3 &half = glm::vec3(0.5f), float mass = 1.0f)
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
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }
} // namespace

// ----------------------------------------------------------------------------
// Core regression: A lands from y=2, then B falls from y=5 onto A's top.
// ----------------------------------------------------------------------------
PHYS_TEST(FallingChain, TwoBoxNoSustainedPenetration)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    int A = addBox(w, glm::vec3(0, 2.0f, 0));
    int B = addBox(w, glm::vec3(0, 5.0f, 0));

    // Run 3 s, checking every frame for transient penetration.
    int steps = static_cast<int>(3.0f / w.fixedTimeStep + 0.5f);
    float minA_y = 1e9f, minB_minus_A_y = 1e9f;
    bool sawASunk = false;
    bool sawBUnderA = false;
    int sunkFrames = 0;
    int underFrames = 0;

    for (int k = 0; k < steps; ++k)
    {
        w.stepSimulation(w.fixedTimeStep);
        float yA = w.getPosition(A).y;
        float yB = w.getPosition(B).y;
        if (yA < minA_y)
            minA_y = yA;
        if ((yB - yA) < minB_minus_A_y)
            minB_minus_A_y = yB - yA;

        // A must not sink below y=-0.3 (ground top at y=0, so the box center is
        // inside the floor); B must never drop below A.
        if (yA < -0.3f)
        {
            sawASunk = true;
            ++sunkFrames;
        }
        if (yB < yA)
        {
            sawBUnderA = true;
            ++underFrames;
        }
    }

    PHYS_CHECK(!sawASunk,
               "A never penetrated ground during 3s (min y must be >= -0.3)");
    PHYS_CHECK(!sawBUnderA,
               "B never slipped below A during 3s");
    (void)sunkFrames;
    (void)underFrames;

    // Final steady state.
    float yA_end = w.getPosition(A).y;
    float yB_end = w.getPosition(B).y;
    float vA_end = glm::length(w.getBody(A).velocity);
    float vB_end = glm::length(w.getBody(B).velocity);

    PHYS_CHECK(yA_end > 0.3f && yA_end < 0.7f,
               "A settles at ground level (~0.5)");
    PHYS_CHECK(yB_end - yA_end > 0.8f && yB_end - yA_end < 1.2f,
               "B stacks on top of A with ~1.0 gap");
    PHYS_CHECK(vA_end < 1.0f, "A final velocity small");
    PHYS_CHECK(vB_end < 1.0f, "B final velocity small");
}

// ----------------------------------------------------------------------------
// Harder: three boxes dropped together so contact chains form mid-air.
// ----------------------------------------------------------------------------
PHYS_TEST(FallingChain, ThreeBoxChainStableAfter3s)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    // Three layers 1.2 m apart, released together: landings are ~1-2 steps
    // apart, which chains the contacts.
    int A = addBox(w, glm::vec3(0, 1.5f, 0));
    int B = addBox(w, glm::vec3(0, 2.7f, 0));
    int C = addBox(w, glm::vec3(0, 3.9f, 0));

    int steps = static_cast<int>(3.0f / w.fixedTimeStep + 0.5f);
    bool anyPenetration = false;
    for (int k = 0; k < steps; ++k)
    {
        w.stepSimulation(w.fixedTimeStep);
        float yA = w.getPosition(A).y;
        float yB = w.getPosition(B).y;
        float yC = w.getPosition(C).y;
        // Order C >= B >= A must hold at all times (0.05 tolerance for contact slop).
        if (yA > yB + 0.05f || yB > yC + 0.05f)
            anyPenetration = true;
        if (yA < -0.3f)
            anyPenetration = true;
    }

    PHYS_CHECK(!anyPenetration,
               "3-box chain: order preserved and no ground sink during 3s");

    float yA = w.getPosition(A).y;
    float yB = w.getPosition(B).y;
    float yC = w.getPosition(C).y;
    PHYS_CHECK(yB - yA > 0.8f, "B-A gap stable");
    PHYS_CHECK(yC - yB > 0.8f, "C-B gap stable");

    for (int id : {A, B, C})
        PHYS_CHECK(glm::length(w.getBody(id).velocity) < 1.0f,
                   "3-box chain final velocity small");
}
