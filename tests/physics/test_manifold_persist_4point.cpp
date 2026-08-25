// ============================================================================
// test_manifold_persist_4point.cpp — multi-point manifold warm-start regression
// ----------------------------------------------------------------------------
// Single-point manifolds jittered the bottom corners across frames, so only
// non-negative/non-diverging Jn could be asserted. Multi-point manifolds keep
// the 4 corners stable, making true warm-start inheritance testable: frame-1
// sum(Jn) at gravity support, frame-2 in the same order of magnitude (no
// collapse or divergence), steady state ≈ m*g*dt before sleep (sleep zeroing
// Jn is correct physics).
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    void setupFloorBoxScene(PhysicsWorld &w,
                            float boxY,
                            int &groundIdx, int &boxIdx)
    {
        w.gravity = glm::vec3(0, -9.81f, 0);
        w.fixedTimeStep = 1.0f / 60.0f;

        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial groundMat;
        groundMat.restitution = 0.3f;
        groundMat.friction = 0.8f;
        OBB gobb;
        gobb.center = glm::vec3(0, -1.5f, 0);
        gobb.halfExtents = glm::vec3(2.75f, 0.4f, 2.75f);
        gobb.orientation = glm::mat3(1.0f);
        groundIdx = w.addBody(rb, gobb, groundMat);

        RigidBody rb2;
        rb2.bodyType = BodyType::Dynamic;
        rb2.mass = 1.0f;
        PhysicsMaterial boxMat;
        boxMat.restitution = 0.3f;
        boxMat.friction = 0.5f;
        OBB bobb;
        bobb.center = glm::vec3(0, boxY, 0);
        bobb.halfExtents = glm::vec3(0.5f);
        bobb.orientation = glm::mat3(1.0f);
        boxIdx = w.addBody(rb2, bobb, boxMat);
    }
} // namespace

PHYS_TEST(ManifoldPersist4Point, FourContactsPerFrame)
{
    // With multi-point manifolds, a box flat on the floor must yield 4
    // contacts every frame.
    PhysicsWorld w;
    int gi, bi;
    setupFloorBoxScene(w, /*boxY=*/-0.61f, gi, bi);

    for (int frame = 0; frame < 5; ++frame)
    {
        w.stepSimulation(w.fixedTimeStep);
        const auto &cs = w.getContacts();
        PHYS_CHECK_EQ((int)cs.size(), 4);
    }
}

PHYS_TEST(ManifoldPersist4Point, WarmStartInheritance)
{
    // Warm start means frame-N Jn inherits from frame N-1, not "frame 1 is 0":
    // via getManifolds() the first frame already shows resolved, non-zero Jn.
    // Assert the inheritance instead:
    //   (a) frame 1: 4 points exist, sum(Jn) at gravity-support magnitude
    //   (b) frame 2: sum(Jn) in the same order of magnitude (stable
    //       inheritance, no collapse/divergence), each point still non-negative
    //   (c) some point Jn > 0 (warm start seeded the next frame's SI solve)
    PhysicsWorld w;
    int gi, bi;
    setupFloorBoxScene(w, -0.61f, gi, bi);

    w.stepSimulation(w.fixedTimeStep);
    const auto &ms1 = w.getManifolds();
    int pts1 = 0;
    float sumJn1 = 0.0f;
    for (const auto &m : ms1)
    {
        pts1 += m.pointCount;
        for (int k = 0; k < m.pointCount; ++k)
            sumJn1 += m.points[k].accumNormalImpulse;
    }
    PHYS_CHECK_EQ(pts1, 4);
    PHYS_CHECK(sumJn1 > 0.02f,
               "frame-1 sum(Jn) must reach gravity-support magnitude");

    w.stepSimulation(w.fixedTimeStep);
    const auto &ms2 = w.getManifolds();
    int pts2 = 0;
    float sumJn2 = 0.0f;
    float maxJn2 = 0.0f;
    for (const auto &m : ms2)
    {
        pts2 += m.pointCount;
        for (int k = 0; k < m.pointCount; ++k)
        {
            PHYS_CHECK(m.points[k].accumNormalImpulse >= 0.0f,
                       "Jn non-negative after SI clamp");
            sumJn2 += m.points[k].accumNormalImpulse;
            maxJn2 = std::max(maxJn2, m.points[k].accumNormalImpulse);
        }
    }
    PHYS_CHECK_EQ(pts2, 4);
    PHYS_CHECK(maxJn2 > 0.001f,
               "warm-start: at least one contact inherits non-zero Jn");
    // Stable inheritance: frame-2 sum(Jn) in the same order of magnitude as
    // frame-1 (0.25x-4x allowed). This is the key no-collapse/no-divergence
    // check, truly exercising warm-start continuity.
    PHYS_CHECK(sumJn2 > 0.25f * sumJn1,
               "warm-start: frame-2 Jn must not collapse to ~0");
    PHYS_CHECK(sumJn2 < 4.0f * sumJn1,
               "warm-start: frame-2 Jn must not diverge");
}

PHYS_TEST(ManifoldPersist4Point, SteadyStateSupportsGravityBeforeSleep)
{
    // With multi-point manifolds + warm start the system briefly balances
    // impulse-supported gravity before sleeping. sleepFramesRequired is 12
    // frames (PhysicsWorld constant), so 8 frames stay before sleep and Jn
    // should be at steady magnitude. Once asleep Jn zeroes — correct physics
    // (sleep skips integrate + resolve) — so no steady-Jn assertion there.
    PhysicsWorld w;
    int gi, bi;
    setupFloorBoxScene(w, -0.61f, gi, bi);

    // Run 8 frames (< sleepFramesRequired=12), tracking the peak per-frame
    // sum(Jn); sum(Jn) aggregates accumNormalImpulse from getManifolds().
    float peakSumJn = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        w.stepSimulation(w.fixedTimeStep);
        const auto &ms = w.getManifolds();
        float sumJn = 0.0f;
        for (const auto &m : ms)
            for (int kp = 0; kp < m.pointCount; ++kp)
                sumJn += m.points[kp].accumNormalImpulse;
        if (sumJn > peakSumJn)
            peakSumJn = sumJn;
    }
    // Peak sum(Jn) must reach "support 1 kg against gravity" magnitude,
    // ~m*g*dt = 0.163.
    PHYS_CHECK(peakSumJn > 0.05f,
               "peak sum Jn must reach gravity-support magnitude");
    PHYS_CHECK(peakSumJn < 1.0f,
               "peak sum Jn must not diverge");
}

PHYS_TEST(ManifoldPersist4Point, EnterSleepAfterStable)
{
    // After 20 frames the box must be asleep — expected PhysicsWorld
    // behavior; sleeping zeroes accumJn and velocity and skips work beyond
    // broadphase.
    PhysicsWorld w;
    int gi, bi;
    setupFloorBoxScene(w, -0.61f, gi, bi);

    for (int k = 0; k < 20; ++k)
        w.stepSimulation(w.fixedTimeStep);

    const auto &b = w.getBody(bi);
    PHYS_CHECK(b.sleeping, "box should enter sleep after stable rest");
    PHYS_CHECK(glm::length(b.velocity) < 0.01f,
               "sleeping body velocity should be ~0");
}

PHYS_TEST(ManifoldPersist4Point, BoxStaysAtRest)
{
    // Core physics: box flat on the floor for 20 frames must stay ~still,
    // no drift or sinking.
    PhysicsWorld w;
    int gi, bi;
    setupFloorBoxScene(w, -0.61f, gi, bi);

    for (int k = 0; k < 20; ++k)
        w.stepSimulation(w.fixedTimeStep);

    glm::vec3 p = w.getPosition(bi);
    glm::vec3 v = w.getBody(bi).velocity;

    // Box bottom y near -1.1 (ground top); small penetration slop acceptable.
    float bottomY = p.y - 0.5f;
    PHYS_CHECK(bottomY >= -1.15f,
               "box not sunk below ground");
    PHYS_CHECK(bottomY <= -1.08f,
               "box resting on ground surface");

    // No horizontal drift (friction cone locks the axis-aligned case).
    PHYS_CHECK(std::fabs(p.x) < 0.1f, "no x drift");
    PHYS_CHECK(std::fabs(p.z) < 0.1f, "no z drift");

    // Velocity near zero.
    PHYS_CHECK(glm::length(v) < 0.5f,
               "box velocity near zero at rest");
}
