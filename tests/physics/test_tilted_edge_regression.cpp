// ============================================================================
// test_tilted_edge_regression.cpp — historical regressions: edge/corner-landed
// cubes must settle within seconds (final posture is free, per generality principle)
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

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

    int addBox(PhysicsWorld &w, const glm::vec3 &center, const glm::mat3 &R,
               const glm::vec3 &half = glm::vec3(0.5f))
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.6f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = R;
        return w.addBody(rb, obb, mat);
    }
    glm::mat3 eulerXYZ(float degX, float degY, float degZ)
    {
        glm::mat4 M(1.0f);
        M = glm::rotate(M, glm::radians(degX), glm::vec3(1, 0, 0));
        M = glm::rotate(M, glm::radians(degY), glm::vec3(0, 1, 0));
        M = glm::rotate(M, glm::radians(degZ), glm::vec3(0, 0, 1));
        return glm::mat3(M);
    }

    void runForSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

// ----------------------------------------------------------------------------
// Historical regression 1: the 30°/60°/20° cube must not balance on an edge after
// landing (root cause was a single-point manifold concentrating normal impulses at
// one cone point with abnormal lever arm; fixed by multi-point manifold + dual-tangent friction)
// ----------------------------------------------------------------------------
PHYS_TEST(TiltedEdgeRegression, Cube_30_60_20_MustSettle)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    int bi = addBox(w, glm::vec3(0, 3.0f, 0), eulerXYZ(30.0f, 60.0f, 20.0f));

    runForSeconds(w, 6.0f);

    glm::vec3 p = w.getPosition(bi);
    const auto &b = w.getBody(bi);
    float linVel = glm::length(b.velocity);
    float angVel = glm::length(b.angularVelocity);

    PHYS_CHECK(linVel < 0.5f, "linear velocity stopped within 6s");
    PHYS_CHECK(angVel < 0.5f, "angular velocity stopped within 6s");
    // Not flown off
    PHYS_CHECK(std::fabs(p.x) < 2.0f && std::fabs(p.z) < 2.0f, "box not flying off");
    // Landed: center below the corner radius sqrt(3)/2·0.5 ≈ 0.433 (plus slop)
    PHYS_CHECK(p.y < 0.85f, "box has landed");
    PHYS_CHECK(p.y > -0.5f, "box not deeply penetrated");
}

// ----------------------------------------------------------------------------
// Historical regression 2: corner-contact cube (local (1,1,1) pointing down) must
// not jitter indefinitely
// ----------------------------------------------------------------------------
PHYS_TEST(TiltedEdgeRegression, Cube_CornerContact_MustSettle)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    // Point the (1,1,1) corner down: rotate about (-1,0,1) by arctan(sqrt(2)) ≈ 54.7356°
    glm::mat4 M = glm::rotate(glm::mat4(1.0f),
                              glm::radians(54.7356f),
                              glm::normalize(glm::vec3(-1, 0, 1)));
    int bi = addBox(w, glm::vec3(0, 3.0f, 0), glm::mat3(M));

    runForSeconds(w, 8.0f);

    const auto &b = w.getBody(bi);
    float linVel = glm::length(b.velocity);
    float angVel = glm::length(b.angularVelocity);

    PHYS_CHECK(linVel < 0.5f, "corner-contact cube: linear velocity settled");
    PHYS_CHECK(angVel < 0.5f, "corner-contact cube: angular velocity settled");
}

// ----------------------------------------------------------------------------
// Historical regression 3: cubes landing at various angles at once must all settle
// (batch regression anchor)
// ----------------------------------------------------------------------------
PHYS_TEST(TiltedEdgeRegression, MultiAnglesBatchSettle)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);

    // 5 representative angle cases covering edge / corner / face contact
    struct AngleCase
    {
        float rx, ry, rz;
        float px, pz;
    };
    AngleCase cases[] = {
        {0, 0, 0, -2.0f, -2.0f},           // face contact
        {30, 60, 20, -1.0f, -2.0f},        // historical bug: edge contact
        {45, 0, 0, 0.0f, -2.0f},           // edge contact (Z axis)
        {54.7356f, 0, 45.0f, 1.0f, -2.0f}, // near corner contact
        {15, 45, 75, 2.0f, -2.0f},         // mixed twist
    };

    std::vector<int> ids;
    for (auto &c : cases)
        ids.push_back(addBox(w, glm::vec3(c.px, 2.5f, c.pz), eulerXYZ(c.rx, c.ry, c.rz)));

    runForSeconds(w, 8.0f);

    for (size_t i = 0; i < ids.size(); ++i)
    {
        const auto &b = w.getBody(ids[i]);
        float linVel = glm::length(b.velocity);
        float angVel = glm::length(b.angularVelocity);
        glm::vec3 p = w.getPosition(ids[i]);

        PHYS_CHECK(linVel < 0.6f,
                   "multi-angle case: linear velocity settled");
        PHYS_CHECK(angVel < 0.6f,
                   "multi-angle case: angular velocity settled");
        PHYS_CHECK(p.y > -0.5f && p.y < 1.2f,
                   "multi-angle case: box position reasonable");
    }
}
