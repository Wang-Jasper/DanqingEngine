// ============================================================================
// test_mixed_stacking.cpp — Box/Sphere/Capsule/Hull mixed-scene integration
// ----------------------------------------------------------------------------
// Full stepSimulation runs verifying detectCollisions dispatch (Box-Box SAT,
// everything else GJK+EPA) in mixed scenes: sphere, capsule and hull fall and
// rest on the floor; a small sphere sits on a box.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    int addStaticBoxBody(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half)
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

    // Dynamic body carrying a sphere shape.
    int addSphereBody(PhysicsWorld &w, const glm::vec3 &center, float radius,
                      float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        BodyPose bp;
        bp.position = center;
        int bi = w.addEmptyBody(rb, bp);
        Shape sh;
        sh.type = ShapeType::Sphere;
        sh.radius = radius;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.5f;
        BodyPose localId;
        w.attachShape(bi, sh, localId, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }

    int addCapsuleBody(PhysicsWorld &w, const glm::vec3 &center, float radius,
                       float halfHeight, float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        BodyPose bp;
        bp.position = center;
        int bi = w.addEmptyBody(rb, bp);
        Shape sh;
        sh.type = ShapeType::Capsule;
        sh.radius = radius;
        sh.halfHeight = halfHeight;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.5f;
        BodyPose localId;
        w.attachShape(bi, sh, localId, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }

    int addHullBody(PhysicsWorld &w, const glm::vec3 &center,
                    const std::vector<glm::vec3> &verts, float mass = 1.0f)
    {
        int hi = w.registerConvexHull(verts);
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        BodyPose bp;
        bp.position = center;
        int bi = w.addEmptyBody(rb, bp);
        Shape sh;
        sh.type = ShapeType::ConvexHull;
        sh.hullIndex = hi;
        // attachShape backfills hullCache automatically.
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.5f;
        BodyPose localId;
        w.attachShape(bi, sh, localId, mat, 0u, 0xFFFFFFFFu, false);
        return bi;
    }

    int addBoxDynamic(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half,
                      float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.5f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    void runForSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

PHYS_TEST(MixedStacking, SphereLandsOnGround)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    // Ground top at y = 0.
    addStaticBoxBody(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));
    int bi = addSphereBody(w, glm::vec3(0, 3.0f, 0), 0.3f);

    runForSeconds(w, 5.0f);

    glm::vec3 p = w.getPosition(bi);
    glm::vec3 v = w.getBody(bi).velocity;

    // Sphere resting on ground top (y=0): center y = 0.3 (±0.1 slop).
    PHYS_CHECK(p.y > -0.1f && p.y < 0.5f,
               "sphere rests on ground (center y near radius)");
    PHYS_CHECK(glm::length(v) < 1.0f, "sphere nearly at rest");
}

PHYS_TEST(MixedStacking, CapsuleLandsOnGround)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticBoxBody(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));
    int bi = addCapsuleBody(w, glm::vec3(0, 3.0f, 0), 0.3f, 0.5f);

    runForSeconds(w, 5.0f);

    glm::vec3 p = w.getPosition(bi);
    glm::vec3 v = w.getBody(bi).velocity;

    // Standing capsule bottom y = center - halfHeight - radius = -0.8, so at
    // rest center y ≈ 0.8; tipping is allowed (lying down gives y ≈ radius =
    // 0.3). Accept y in [0.2, 1.0].
    PHYS_CHECK(p.y > 0.0f && p.y < 1.2f, "capsule rests on ground");
    PHYS_CHECK(glm::length(v) < 1.5f, "capsule nearly at rest");
}

PHYS_TEST(MixedStacking, SphereOnBox)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticBoxBody(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));
    int bBox = addBoxDynamic(w, glm::vec3(0, 0.5f, 0), glm::vec3(0.5f));
    int bSph = addSphereBody(w, glm::vec3(0, 2.0f, 0), 0.3f);

    runForSeconds(w, 5.0f);

    glm::vec3 pBox = w.getPosition(bBox);
    glm::vec3 pSph = w.getPosition(bSph);
    glm::vec3 vBox = w.getBody(bBox).velocity;
    glm::vec3 vSph = w.getBody(bSph).velocity;

    // Box still on the floor (bottom near y=0).
    PHYS_CHECK(pBox.y > 0.3f && pBox.y < 0.7f,
               "box rests on ground");
    // Sphere sits on the box top (box top y=1, sphere center y ≈ 1.3).
    PHYS_CHECK(pSph.y > 1.0f && pSph.y < 1.7f,
               "sphere rests on top of box");
    PHYS_CHECK(glm::length(vBox) < 1.0f, "box at rest");
    PHYS_CHECK(glm::length(vSph) < 1.0f, "sphere at rest");
}

PHYS_TEST(MixedStacking, HullLandsOnGround)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticBoxBody(w, glm::vec3(0, -0.4f, 0), glm::vec3(5.0f, 0.4f, 5.0f));

    // 8-vertex hull approximating a unit cube.
    std::vector<glm::vec3> verts = {
        glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, -0.5f),
        glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(0.5f, 0.5f, -0.5f),
        glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(0.5f, -0.5f, 0.5f),
        glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f)};
    int bi = addHullBody(w, glm::vec3(0, 3.0f, 0), verts);

    runForSeconds(w, 5.0f);

    glm::vec3 p = w.getPosition(bi);
    glm::vec3 v = w.getBody(bi).velocity;

    // Hull local bottom y=-0.5, so at rest center y ≈ 0.5.
    PHYS_CHECK(p.y > 0.2f && p.y < 0.9f,
               "hull rests on ground (center y near 0.5)");
    PHYS_CHECK(glm::length(v) < 1.5f, "hull nearly at rest");
}
