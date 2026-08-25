// ============================================================================
// test_manifold_persist.cpp — persistent-manifold and warm-start basics
// ----------------------------------------------------------------------------
// First-contact and post-separation fresh-manifold Jn magnitude (multi-point
// warm start lives in test_manifold_persist_4point.cpp). CP-3.1 reads the
// solver-written accumNormalImpulse, so a fresh manifold is never 0 after
// stepSimulation; assert gravity-support magnitude instead.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    // Stable scene: small box just penetrating a large floor.
    // - ground: static, (0, -1.5, 0), half (2.75, 0.4, 2.75), top at y=-1.1
    // - box: dynamic mass 1, half (0.5, 0.5, 0.5)
    // - box position puts its bottom slightly below y=-1.1, in direct contact
    void setupFloorBoxScene(PhysicsWorld &w,
                            float boxY,
                            int &groundIdx, int &boxIdx)
    {
        w.gravity = glm::vec3(0, -9.81f, 0);
        w.fixedTimeStep = 1.0f / 60.0f;

        // Ground
        {
            RigidBody rb;
            rb.bodyType = BodyType::Static;
            rb.mass = 0.0f;
            PhysicsMaterial mat;
            mat.restitution = 0.3f;
            mat.friction = 0.8f;
            OBB obb;
            obb.center = glm::vec3(0, -1.5f, 0);
            obb.halfExtents = glm::vec3(2.75f, 0.4f, 2.75f);
            obb.orientation = glm::mat3(1.0f);
            groundIdx = w.addBody(rb, obb, mat);
        }
        // Box
        {
            RigidBody rb;
            rb.bodyType = BodyType::Dynamic;
            rb.mass = 1.0f;
            PhysicsMaterial mat;
            mat.restitution = 0.3f;
            mat.friction = 0.5f;
            OBB obb;
            obb.center = glm::vec3(0, boxY, 0);
            obb.halfExtents = glm::vec3(0.5f);
            obb.orientation = glm::mat3(1.0f);
            boxIdx = w.addBody(rb, obb, mat);
        }
    }
} // namespace

PHYS_TEST(ManifoldPersist, FirstFrameJnSupportsGravity)
{
    PhysicsWorld w;
    int gi, bi;
    // Box bottom at y=-1.1 touches the floor top; 0.01 lower guarantees a
    // shallow penetration.
    setupFloorBoxScene(w, /*boxY=*/-0.61f, gi, bi);

    // Step 1: detect creates the manifold (Jn starts 0) and resolve runs SI,
    // accumulating Jn. External reads see the post-resolve value: gravity-support magnitude.
    w.stepSimulation(w.fixedTimeStep);

    const auto &ms = w.getManifolds();
    PHYS_CHECK(!ms.empty(), "first-frame contact must be detected");

    // Each point's accumNormalImpulse must be non-negative (solver clamps
    // normal impulses to max(0,·)).
    float sumJn = 0.0f;
    int pts = 0;
    for (const auto &m : ms)
    {
        for (int k = 0; k < m.pointCount; ++k)
        {
            PHYS_CHECK(m.points[k].accumNormalImpulse >= 0.0f,
                       "normal impulse must be non-negative after SI clamp");
            sumJn += m.points[k].accumNormalImpulse;
            ++pts;
        }
    }
    PHYS_CHECK(pts > 0, "at least one contact point");
    // Magnitude check: m*g*dt = 1*9.81*(1/60) ≈ 0.1635; with restitution and
    // incomplete SI convergence the first frame lands in [0.05, 1.0].
    PHYS_CHECK(sumJn > 0.02f,
               "first-frame sum(Jn) must reach gravity-support magnitude");
    PHYS_CHECK(sumJn < 2.0f,
               "first-frame sum(Jn) must not diverge");
}

PHYS_TEST(ManifoldPersist, SeparationThenContactYieldsFreshJn)
{
    PhysicsWorld w;
    int gi, bi;
    // Box starts far above the floor (y=2 vs floor top y=-1.1).
    setupFloorBoxScene(w, 2.0f, gi, bi);

    // The first frames must have no contact.
    for (int k = 0; k < 3; ++k)
        w.stepSimulation(w.fixedTimeStep);
    PHYS_CHECK(w.getManifolds().empty(),
               "no contact while box falls freely above ground");

    // Teleport the box onto the floor, then run one frame.
    BodyPose bp;
    bp.position = glm::vec3(0, -0.6f, 0);
    bp.orientation = glm::mat3(1.0f);
    w.setBodyPose(bi, bp);

    // The manifold is brand new (the previous one was fully erased), so it
    // inherits no history; after the first resolve Jn must reach gravity
    // support plus penetration restitution from zero.
    w.stepSimulation(w.fixedTimeStep);
    const auto &ms = w.getManifolds();
    PHYS_CHECK(!ms.empty(), "contact re-established after warp");

    float sumJn = 0.0f;
    for (const auto &m : ms)
    {
        for (int k = 0; k < m.pointCount; ++k)
        {
            PHYS_CHECK(m.points[k].accumNormalImpulse >= 0.0f,
                       "Jn non-negative after SI clamp");
            sumJn += m.points[k].accumNormalImpulse;
        }
    }
    // Deeper penetration after the teleport raises Jn above a normal first
    // frame; still bounded.
    PHYS_CHECK(sumJn > 0.02f,
               "fresh manifold after teleport yields non-trivial support Jn");
    PHYS_CHECK(sumJn < 3.0f,
               "fresh manifold Jn must not diverge even after teleport");
}
