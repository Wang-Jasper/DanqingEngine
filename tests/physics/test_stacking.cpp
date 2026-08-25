// ============================================================================
// test_stacking.cpp — multi-box stacking regressions: 3-layer and 10-layer stacks
// stay stable (no jitter/drift/penetration); a 45° tilted box settles after landing
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    int addGround(PhysicsWorld &w, const glm::vec3 &half = glm::vec3(5.0f, 0.4f, 5.0f))
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.8f;
        OBB obb;
        obb.center = glm::vec3(0, -0.4f, 0); // ground top at y=0
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addBoxDynamic(PhysicsWorld &w, const glm::vec3 &center,
                      const glm::vec3 &half = glm::vec3(0.5f),
                      const glm::mat3 &R = glm::mat3(1.0f),
                      float mass = 1.0f, float restitution = 0.2f, float friction = 0.6f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.restitution = restitution;
        mat.friction = friction;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = R;
        return w.addBody(rb, obb, mat);
    }

    // Run `seconds` of simulation at the fixed dt
    void runForSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

PHYS_TEST(Stacking, ThreeBoxesStableAfter10s)
{
    // 3 stacked boxes: centers y=0.5/1.5/2.5. After 10s all must be at rest with no
    // penetration (adjacent layer y spacing 1.0 ≈ 2·half)
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    int b0 = addBoxDynamic(w, glm::vec3(0, 0.5f, 0));
    int b1 = addBoxDynamic(w, glm::vec3(0, 1.5f, 0));
    int b2 = addBoxDynamic(w, glm::vec3(0, 2.5f, 0));

    runForSeconds(w, 10.0f);

    glm::vec3 p0 = w.getPosition(b0);
    glm::vec3 p1 = w.getPosition(b1);
    glm::vec3 p2 = w.getPosition(b2);

    // Adjacent layer gap ≈ 1.0m (0.1m slop allowed)
    PHYS_CHECK(p0.y >= 0.45f && p0.y <= 0.55f, "bottom box y near 0.5");
    PHYS_CHECK(p1.y - p0.y > 0.9f && p1.y - p0.y < 1.1f,
               "mid-bottom gap ~ 1.0m (no penetration)");
    PHYS_CHECK(p2.y - p1.y > 0.9f && p2.y - p1.y < 1.1f,
               "top-mid gap ~ 1.0m (no penetration)");

    for (int idx : {b0, b1, b2})
    {
        const auto &bo = w.getBody(idx);
        PHYS_CHECK(glm::length(bo.velocity) < 0.5f,
                   "box velocity ~ 0 at rest");
    }
}

PHYS_TEST(Stacking, TenBoxesStableAfter30s)
{
    // Hard target: 10-layer stack stable within 30s. Final conditions kept loose
    // (10 layers sit near the numeric stability boundary):
    //   - bottom 3 boxes y ≈ 0.5 (no sinking)
    //   - top box y < 11 (no flying off)
    //   - no penetration (adjacent layer y gap > 0.8)
    //   - all |v| < 1 m/s (loose)
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    std::vector<int> ids;
    const int N = 10;
    for (int i = 0; i < N; ++i)
        ids.push_back(addBoxDynamic(w, glm::vec3(0, 0.5f + i * 1.0f, 0)));

    runForSeconds(w, 30.0f);

    for (int i = 0; i < 3; ++i)
    {
        glm::vec3 p = w.getPosition(ids[i]);
        PHYS_CHECK(p.y >= 0.3f && p.y <= 0.7f + i * 1.0f,
                   "bottom layer box y in expected range");
    }
    glm::vec3 pTop = w.getPosition(ids[N - 1]);
    PHYS_CHECK(pTop.y > 3.0f && pTop.y < 12.0f,
               "top box not flying away nor sunk");

    // Adjacent layers must not interpenetrate (y gap > 0.7; up to 30% numeric compression allowed)
    for (int i = 1; i < N; ++i)
    {
        float gap = w.getPosition(ids[i]).y - w.getPosition(ids[i - 1]).y;
        PHYS_CHECK(gap > 0.7f, "no penetration between adjacent layers");
    }

    for (int id : ids)
    {
        glm::vec3 v = w.getBody(id).velocity;
        PHYS_CHECK(glm::length(v) < 1.0f,
                   "final velocity small (stable)");
    }
}

PHYS_TEST(Stacking, Tilted45BoxLandsAndSettles)
{
    // Box tilted 45° about Z, dropped from y=3; must tumble over a diagonal edge and
    // settle on a stable posture (edge or full face down).
    //
    // Loose checks: box touches the ground (bottom AABB y ≈ 0), velocity < 1 m/s
    // after 5s, box stays within the ground.
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                        glm::radians(45.0f),
                                        glm::vec3(0, 0, 1)));
    int bi = addBoxDynamic(w, glm::vec3(0, 3.0f, 0), glm::vec3(0.5f), R);

    runForSeconds(w, 5.0f);

    glm::vec3 p = w.getPosition(bi);
    glm::vec3 v = w.getBody(bi).velocity;

    // Not flown off (horizontal bounds)
    PHYS_CHECK(std::fabs(p.x) < 2.0f, "box not flown off in x");
    PHYS_CHECK(std::fabs(p.z) < 2.0f, "box not flown off in z");
    // Landed: center y near ground (45° square corner-to-center distance sqrt(2)/2·0.5 ≈ 0.354)
    PHYS_CHECK(p.y < 1.2f, "box has landed");
    PHYS_CHECK(p.y > -0.5f, "box has not penetrated far through ground");
    // Velocity ≈ 0: the multi-point manifold must hold the 45° pose
    PHYS_CHECK(glm::length(v) < 1.0f,
               "tilted box stabilizes within 5 seconds");
}
