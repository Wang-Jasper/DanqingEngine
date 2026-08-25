// End-to-end CCD regressions: high-speed bodies must not tunnel with CCD on,
// medium-speed scenes must be unaffected, and high-speed solving must stay finite.
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    int addGroundThin(PhysicsWorld &w, float thickness = 0.1f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.8f;
        OBB obb;
        obb.center = glm::vec3(0, -thickness * 0.5f, 0);
        obb.halfExtents = glm::vec3(5.0f, thickness * 0.5f, 5.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addSphereDyn(PhysicsWorld &w, const glm::vec3 &pos, float r = 0.2f, float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.3f;
        OBB obb;
        obb.center = pos;
        obb.halfExtents = glm::vec3(r);
        obb.orientation = glm::mat3(1.0f);
        int bi = w.addBody(rb, obb, mat);
        return bi;
    }
} // namespace

// ----------------------------------------------------------------------------
// High-speed sphere falling onto a thin floor: no tunneling with CCD on.
// ----------------------------------------------------------------------------
PHYS_TEST(CcdIntegration, HighSpeedSphereNoPenetration_CcdOn)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0);
    w.fixedTimeStep = 1.0f / 60.0f;
    w.ccdEnabled = true;

    addGroundThin(w, 0.1f);
    int s = addSphereDyn(w, glm::vec3(0, 3.0f, 0), 0.2f);
    // ~180 m/s covers the 3 m drop in one 1/60 s step.
    w.setLinearVelocity(s, glm::vec3(0, -500.0f, 0));

    // 5 steps at 1/60 s: enough to cross the floor.
    for (int i = 0; i < 5; ++i)
        w.stepSimulation(w.fixedTimeStep);

    glm::vec3 p = w.getPosition(s);
    // Sphere bottom = p.y - 0.2, floor top = 0; CCD keeps the bottom near 0.
    PHYS_CHECK(p.y - 0.2f > -0.2f,
               "CCD on: high-speed sphere did NOT tunnel through thin floor");
}

// ----------------------------------------------------------------------------
// Same scene with CCD off: the sphere tunnels, confirming the toggle matters.
// ----------------------------------------------------------------------------
PHYS_TEST(CcdIntegration, HighSpeedSpherePenetrates_CcdOff)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0);
    w.fixedTimeStep = 1.0f / 60.0f;
    w.ccdEnabled = false;

    addGroundThin(w, 0.1f);
    int s = addSphereDyn(w, glm::vec3(0, 3.0f, 0), 0.2f);
    w.setLinearVelocity(s, glm::vec3(0, -500.0f, 0));

    for (int i = 0; i < 5; ++i)
        w.stepSimulation(w.fixedTimeStep);

    glm::vec3 p = w.getPosition(s);
    // No CCD and no maxLinearSpeed clamp: the sphere tunnels; 5 steps move it
    // 500 * 5/60 ≈ 41.7 m, far below the floor.
    PHYS_CHECK(p.y < -5.0f,
               "CCD off: high-speed sphere tunneled through (confirms CCD toggle matters)");
}

// ----------------------------------------------------------------------------
// Medium-speed scene must end in nearly the same state with CCD on or off.
// ----------------------------------------------------------------------------
PHYS_TEST(CcdIntegration, NormalSpeedUnaffectedByCcd)
{
    auto runScene = [](bool ccdOn) -> float
    {
        PhysicsWorld w;
        w.gravity = glm::vec3(0, -9.81f, 0);
        w.fixedTimeStep = 1.0f / 60.0f;
        w.ccdEnabled = ccdOn;
        addGroundThin(w, 0.4f);
        int s = addSphereDyn(w, glm::vec3(0, 2.0f, 0), 0.3f);
        // 2 s of free fall, bounce, and settling.
        int steps = static_cast<int>(2.0f / w.fixedTimeStep + 0.5f);
        for (int i = 0; i < steps; ++i)
            w.stepSimulation(w.fixedTimeStep);
        return w.getPosition(s).y;
    };

    float yOn = runScene(true);
    float yOff = runScene(false);

    // Low-speed scenes never trigger the CCD threshold, so results should be
    // identical; 5 cm is a conservative bound.
    float diff = std::fabs(yOn - yOff);
    PHYS_CHECK(diff < 0.05f,
               "normal-speed scene: CCD on/off final position delta < 5cm");
    // Both should rest on the floor: top at y=0, sphere r=0.3, center ~0.3.
    PHYS_CHECK(yOn > 0.1f && yOn < 0.5f, "CCD on: sphere settles above ground");
    PHYS_CHECK(yOff > 0.1f && yOff < 0.5f, "CCD off: sphere settles above ground");
}

// ----------------------------------------------------------------------------
// No NaN/inf in high-speed scenes after the maxLinearSpeed clamp was removed.
// ----------------------------------------------------------------------------
PHYS_TEST(CcdIntegration, HighSpeedNumericalStability)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;
    w.ccdEnabled = true;

    addGroundThin(w, 0.4f);
    // Far above the removed maxLinearSpeed=20 clamp.
    int s = addSphereDyn(w, glm::vec3(0, 5.0f, 0), 0.3f);
    w.setLinearVelocity(s, glm::vec3(100.0f, -100.0f, 50.0f));

    for (int i = 0; i < 60; ++i) // 1 s of simulation
        w.stepSimulation(w.fixedTimeStep);

    const auto &b = w.getBody(s);
    glm::vec3 p = w.getPosition(s);
    PHYS_CHECK(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z),
               "position finite after high-speed 1s");
    PHYS_CHECK(std::isfinite(b.velocity.x) && std::isfinite(b.velocity.y) && std::isfinite(b.velocity.z),
               "velocity finite after high-speed 1s");
}
