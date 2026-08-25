// ============================================================================
// test_phase6_verification.cpp — Phase 6 overall verification
// ----------------------------------------------------------------------------
// #1: box/sphere/capsule/hull fall onto one ground and settle independently.
// #2: a >50 m/s bullet must not tunnel a thin wall (also covered by
// CcdIntegration.HighSpeedSphere; bullet-like scene added). #3: high-speed
// bodies must not explode after maxLinearSpeed removal (also covered by
// CcdIntegration.HighSpeedNumerical; long-run stability added here).
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    int addStaticBox(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half)
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

    int addDynShape(PhysicsWorld &w, ShapeType type, const glm::vec3 &pos)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;

        BodyPose pose;
        pose.position = pos;
        pose.orientation = glm::mat3(1.0f);
        int bi = w.addEmptyBody(rb, pose);

        Shape sh;
        sh.type = type;
        switch (type)
        {
        case ShapeType::Box:
            sh.halfExtents = glm::vec3(0.3f);
            break;
        case ShapeType::Sphere:
            sh.radius = 0.3f;
            break;
        case ShapeType::Capsule:
            sh.radius = 0.2f;
            sh.halfHeight = 0.3f;
            break;
        case ShapeType::ConvexHull:
            // Simple tetrahedron hull.
            {
                std::vector<glm::vec3> verts = {
                    glm::vec3(-0.3f, -0.3f, -0.3f),
                    glm::vec3(0.3f, -0.3f, -0.3f),
                    glm::vec3(0.0f, 0.3f, -0.3f),
                    glm::vec3(0.0f, 0.0f, 0.3f),
                };
                sh.hullIndex = w.registerConvexHull(verts);
            }
            break;
        default:
            break;
        }

        BodyPose localPose;
        PhysicsMaterial mat;
        mat.restitution = 0.1f;
        mat.friction = 0.6f;
        w.attachShape(bi, sh, localPose, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }
} // namespace

// ----------------------------------------------------------------------------
// #1: all four shape types fall onto the same box ground
// ----------------------------------------------------------------------------
PHYS_TEST(Phase6Verification, FourShapeTypesCoexistOnGround)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    // Static box ground.
    addStaticBox(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));

    // Four dynamic bodies, spaced apart so they never touch.
    int bBox = addDynShape(w, ShapeType::Box, glm::vec3(-2.0f, 2.0f, 0));
    int bSphere = addDynShape(w, ShapeType::Sphere, glm::vec3(-0.7f, 2.0f, 0));
    int bCapsule = addDynShape(w, ShapeType::Capsule, glm::vec3(0.7f, 2.0f, 0));
    int bHull = addDynShape(w, ShapeType::ConvexHull, glm::vec3(2.0f, 2.0f, 0));

    // 4 s is enough for everything to land and settle.
    int steps = static_cast<int>(4.0f / w.fixedTimeStep + 0.5f);
    for (int i = 0; i < steps; ++i)
        w.stepSimulation(w.fixedTimeStep);

    // All four must neither sink (y > -0.3) nor fly (|v| < 2).
    for (int id : {bBox, bSphere, bCapsule, bHull})
    {
        glm::vec3 p = w.getPosition(id);
        const auto &b = w.getBody(id);
        PHYS_CHECK(p.y > -0.3f && p.y < 2.5f,
                   "shape settles in reasonable y range");
        PHYS_CHECK(glm::length(b.velocity) < 2.0f,
                   "shape final velocity small");
    }
}

// ----------------------------------------------------------------------------
// #2: high-speed bullet (50 m/s+) must not tunnel a thin wall
// ----------------------------------------------------------------------------
PHYS_TEST(Phase6Verification, HighSpeedBulletNoPenetrationBoxBox)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0);
    w.fixedTimeStep = 1.0f / 60.0f;
    w.ccdEnabled = true;

    // Thin wall (0.05 m thick in y).
    addStaticBox(w, glm::vec3(0, 0, 0), glm::vec3(3.0f, 0.025f, 3.0f));

    // Box bullet 2.5 m above, 100 m/s downward: one discrete step covers
    // 100/60 ≈ 1.67 m, far beyond the 0.05 m wall — it would tunnel.
    int bullet = addDynShape(w, ShapeType::Box, glm::vec3(0, 2.5f, 0));
    w.setLinearVelocity(bullet, glm::vec3(0, -100.0f, 0));

    for (int i = 0; i < 60; ++i) // 1 s
        w.stepSimulation(w.fixedTimeStep);

    glm::vec3 p = w.getPosition(bullet);
    // Bullet bottom = p.y - 0.3; wall top = 0.025; dropping below (p.y < -0.3)
    // means tunneling.
    PHYS_CHECK(p.y > -0.2f,
               "Box bullet did NOT tunnel through thin Box wall");
}

// ----------------------------------------------------------------------------
// #3: long-run stability after removing maxLinearSpeed — several high-speed
// bodies over a static ground for 2 s, all positions/velocities finite
// ----------------------------------------------------------------------------
PHYS_TEST(Phase6Verification, NoExplosionAfterRemoveMaxSpeed)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticBox(w, glm::vec3(0, -0.4f, 0), glm::vec3(10.0f, 0.4f, 10.0f));

    int sBox = addDynShape(w, ShapeType::Box, glm::vec3(-3.0f, 5.0f, 0));
    int sSphere = addDynShape(w, ShapeType::Sphere, glm::vec3(-1.0f, 5.0f, 0));
    int sCapsule = addDynShape(w, ShapeType::Capsule, glm::vec3(1.0f, 5.0f, 0));
    int sHull = addDynShape(w, ShapeType::ConvexHull, glm::vec3(3.0f, 5.0f, 0));

    // Box/Hull get extreme vertical speed (far beyond the old
    // maxLinearSpeed=20); horizontal speed stays small so bodies cannot leave
    // the 10 m ground and free-fall off it.
    w.setLinearVelocity(sBox, glm::vec3(1.0f, -80.0f, 0));
    w.setLinearVelocity(sHull, glm::vec3(-1.0f, -80.0f, 0));

    int steps = static_cast<int>(2.0f / w.fixedTimeStep + 0.5f);
    for (int i = 0; i < steps; ++i)
        w.stepSimulation(w.fixedTimeStep);

    for (int id : {sBox, sSphere, sCapsule, sHull})
    {
        glm::vec3 p = w.getPosition(id);
        const auto &b = w.getBody(id);
        PHYS_CHECK(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z),
                   "position finite after 2s high-speed mixed simulation");
        PHYS_CHECK(std::isfinite(b.velocity.x) && std::isfinite(b.velocity.y) && std::isfinite(b.velocity.z),
                   "velocity finite after 2s high-speed mixed simulation");
        // Only numerical stability is checked here — finite positions and
        // velocity < 300 m/s (no runaway). Tunneling is
        // HighSpeedBulletNoPenetration's job.
        PHYS_CHECK(std::fabs(p.x) < 1000.0f && std::fabs(p.z) < 1000.0f,
                   "position finite-ish within scene");
        PHYS_CHECK(std::fabs(p.y) < 1000.0f, "y position finite-ish");
        PHYS_CHECK(glm::length(b.velocity) < 300.0f,
                   "velocity bounded (no numerical explosion)");
    }
}
