// ============================================================================
// test_kinematic_friction.cpp — friction driven by contact-tangential velocity
// ----------------------------------------------------------------------------
// Covers the 7 Phase 17.2 cases: static-ground baseline, kinematic base
// dragging cubes (single and stacked), low-mu slip, rest on stop, no ghost
// force from teleports, and kinematic rotation driving angular velocity.
// Public API only; helpers stay in this file's anonymous namespace; thresholds
// are literals or world members, no magic constants.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    // Static ground; default top aligned to y=0, sized to cover the tests.
    int addStaticGround(PhysicsWorld &w,
                        const glm::vec3 &half = glm::vec3(20.0f, 0.4f, 20.0f),
                        float friction = 0.8f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = friction;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = glm::vec3(0.0f, -0.4f, 0.0f);
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addKinematicBox(PhysicsWorld &w,
                        const glm::vec3 &center,
                        const glm::vec3 &half,
                        float friction = 0.8f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Kinematic;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = friction;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addDynamicBox(PhysicsWorld &w,
                      const glm::vec3 &center,
                      const glm::vec3 &half = glm::vec3(0.5f),
                      float mass = 1.0f,
                      float friction = 0.8f,
                      float restitution = 0.05f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.friction = friction;
        mat.restitution = restitution;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    // Step `seconds` at the fixed dt.
    void stepForSeconds(PhysicsWorld &w, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }

    // Advance a Kinematic body one frame along +X via setBodyPose.
    void advanceKinematicLinear(PhysicsWorld &w, int bodyIdx,
                                const glm::vec3 &vel, float dt)
    {
        OBB obb = w.getOBB(bodyIdx);
        BodyPose pose;
        pose.position = obb.center + vel * dt;
        pose.orientation = obb.orientation;
        w.setBodyPose(bodyIdx, pose);
    }

    // Rotate a Kinematic body about +Y by omega*dt via setBodyPose.
    void advanceKinematicSpinY(PhysicsWorld &w, int bodyIdx, float omega, float dt)
    {
        OBB obb = w.getOBB(bodyIdx);
        glm::mat3 R = obb.orientation;
        glm::mat3 dR = glm::mat3(glm::rotate(glm::mat4(1.0f), omega * dt, glm::vec3(0, 1, 0)));
        BodyPose pose;
        pose.position = obb.center;
        pose.orientation = dR * R;
        w.setBodyPose(bodyIdx, pose);
    }

    // Run N seconds, moving the Kinematic body via the driver callback each frame.
    void runWithKinematicDriver(PhysicsWorld &w, float seconds,
                                const std::function<void(float dt)> &driver)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
        {
            driver(w.fixedTimeStep);
            w.stepSimulation(w.fixedTimeStep);
        }
    }
} // namespace

// ----------------------------------------------------------------------------
// 1. StaticGroundBaseline — static behavior must not regress after Kinematic
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, StaticGroundBaseline)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    int cube = addDynamicBox(w, glm::vec3(0, 1.0f, 0), glm::vec3(0.5f), 1.0f, 0.8f);

    stepForSeconds(w, 2.0f);

    glm::vec3 p = w.getPosition(cube);
    glm::vec3 v = w.getBody(cube).velocity;
    PHYS_CHECK(std::fabs(p.x) < 0.01f, "no lateral drift on static ground");
    PHYS_CHECK(std::fabs(p.z) < 0.01f, "no lateral drift on static ground");
    PHYS_CHECK(glm::length(v) < w.sleepLinearThreshold + 0.05f,
               "settled velocity below sleep threshold");
}

// ----------------------------------------------------------------------------
// 2. MovingGroundDragsStackedCube — kinematic base drags the cube above (high mu)
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, MovingGroundDragsStackedCube)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    // Static floor kept below the kinematic base so they cannot overlap;
    // kinematic base (platform) top at y=0; dynamic cube at y=1.0 free-falls to rest.
    addStaticGround(w, glm::vec3(20.0f, 0.1f, 20.0f)); // top at y=0
    int kinBase = addKinematicBox(w, glm::vec3(0, 0.8f, 0),
                                  glm::vec3(4.0f, 0.5f, 4.0f), 0.8f); // top at y=1.3
    int cube = addDynamicBox(w, glm::vec3(0, 2.3f, 0), glm::vec3(0.5f), 1.0f, 0.8f);

    // Free-fall 1 s so the cube rests solidly on the kinematic base.
    stepForSeconds(w, 1.0f);

    // Then translate the kinematic body along +X at v0 = 1 m/s every frame.
    const float v0 = 1.0f;
    glm::vec3 cubeStart = w.getPosition(cube);
    runWithKinematicDriver(w, 3.0f, [&](float dt)
                           { advanceKinematicLinear(w, kinBase, glm::vec3(v0, 0.0f, 0.0f), dt); });

    glm::vec3 v = w.getBody(cube).velocity;
    glm::vec3 p = w.getPosition(cube);
    float dx = p.x - cubeStart.x;

    PHYS_CHECK(v.x > 0.6f * v0, "upper cube tangential vel follows Kinematic base");
    PHYS_CHECK(std::fabs(v.y) < 0.2f, "upper cube not bouncing vertically");
    PHYS_CHECK(dx > 1.5f, "upper cube carried at least 1.5m in 3s");
}

// ----------------------------------------------------------------------------
// 3. TwoCubeStackDragsUpperCube — generic two-cube stack drag
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, TwoCubeStackDragsUpperCube)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    // Kinematic lower cube top at y=1.0; dynamic upper cube top at y=2.0.
    int lower = addKinematicBox(w, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.5f), 0.8f);
    int upper = addDynamicBox(w, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.5f), 1.0f, 0.8f);

    stepForSeconds(w, 1.0f); // let the stack settle

    const float v0 = 1.0f;
    glm::vec3 upperStart = w.getPosition(upper);
    runWithKinematicDriver(w, 3.0f, [&](float dt)
                           { advanceKinematicLinear(w, lower, glm::vec3(v0, 0.0f, 0.0f), dt); });

    glm::vec3 pU = w.getPosition(upper);
    glm::vec3 pL = w.getPosition(lower);
    glm::vec3 vU = w.getBody(upper).velocity;

    PHYS_CHECK(vU.x > 0.6f * v0, "upper cube dragged by lower Kinematic");
    PHYS_CHECK(pU.x - upperStart.x > 1.5f, "upper cube moved >= 1.5m in 3s");
    // Upper cube must not slide off: horizontal offset stays below the lower
    // half extent, and it keeps resting on the lower top face.
    PHYS_CHECK(std::fabs(pU.x - pL.x) < 0.6f, "upper stays atop lower (no slide off)");
    PHYS_CHECK(pU.y > pL.y + 0.8f && pU.y < pL.y + 1.2f,
               "upper sits 1.0m above lower (no penetration)");
}

// ----------------------------------------------------------------------------
// 4. LowFrictionSlipsFreely — low-mu cube must lag the base (friction cone not saturated)
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, LowFrictionSlipsFreely)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    // Base large enough that the cube stays over it during the short drive.
    int lower = addKinematicBox(w, glm::vec3(0.0f, 0.5f, 0.0f),
                                glm::vec3(5.0f, 0.5f, 1.0f), 0.05f);
    int upper = addDynamicBox(w, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.5f), 1.0f, 0.05f);

    stepForSeconds(w, 1.0f);

    const float v0 = 1.0f;
    // The friction cone caps tangential impulse at |Jt| <= mu*Jn ~ mu*m*g*dt, so
    // the tangential velocity gain per substep is <= mu*g*dt. Over 0.5 s (30
    // substeps) the saturated ceiling is ~0.05*9.81*0.5 = 0.245 m/s, matching
    // the assertion vU.x < 0.3*v0 = 0.3 m/s with a little engineering margin.
    glm::vec3 upperStart = w.getPosition(upper);
    const float driveDuration = 0.5f;
    runWithKinematicDriver(w, driveDuration, [&](float dt)
                           { advanceKinematicLinear(w, lower, glm::vec3(v0, 0.0f, 0.0f), dt); });

    glm::vec3 pU = w.getPosition(upper);
    glm::vec3 vU = w.getBody(upper).velocity;

    PHYS_CHECK(vU.x < 0.3f * v0, "low friction: upper does not follow fully within 0.5s");
    PHYS_CHECK(pU.x - upperStart.x < 0.2f, "low friction: upper lags far behind base");
    // Still resting on the lower cube (y within 0.1 m of its top).
    float pLy = w.getPosition(lower).y;
    PHYS_CHECK(pU.y > pLy + 0.85f && pU.y < pLy + 1.15f,
               "upper still stably resting on lower");
}
// ----------------------------------------------------------------------------
// 5. StackedCubeResumesRestOnStop — upper cube returns to rest once the lower stops
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, StackedCubeResumesRestOnStop)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    int lower = addKinematicBox(w, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.5f), 0.8f);
    int upper = addDynamicBox(w, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.5f), 1.0f, 0.8f);

    stepForSeconds(w, 1.0f);

    // Phase A: lower drives at constant speed for 3 s.
    const float v0 = 1.0f;
    runWithKinematicDriver(w, 3.0f, [&](float dt)
                           { advanceKinematicLinear(w, lower, glm::vec3(v0, 0.0f, 0.0f), dt); });

    // Phase B: lower stays completely still for 2 s.
    stepForSeconds(w, 2.0f);

    glm::vec3 vU = w.getBody(upper).velocity;
    // Friction must brake the tangential velocity back into the resting range.
    PHYS_CHECK(glm::length(vU) < w.restingTangentLockThreshold + 0.05f,
               "upper returns to resting after Kinematic stops");
}

// ----------------------------------------------------------------------------
// 6. NoGhostForceFromTeleport — gizmo teleports must not inject fake velocity
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, NoGhostForceFromTeleport)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    int lower = addKinematicBox(w, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.5f), 0.8f);
    int upper = addDynamicBox(w, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.5f), 1.0f, 0.8f);

    stepForSeconds(w, 1.0f);

    // One-frame 10 m teleport: refreshKinematicVelocities must treat it as a
    // teleport and zero the implied velocity.
    {
        OBB obb = w.getOBB(lower);
        BodyPose pose;
        pose.position = obb.center + glm::vec3(10.0f, 0.0f, 0.0f);
        pose.orientation = obb.orientation;
        w.setBodyPose(lower, pose);
    }

    // The next stepSimulation consumes the teleport velocity; the teleport
    // protection must keep the upper cube's speed at a sane magnitude.
    w.stepSimulation(w.fixedTimeStep);

    glm::vec3 vU = w.getBody(upper).velocity;
    // After the teleport the upper cube loses support and free-falls, but no
    // huge horizontal velocity may be injected.
    PHYS_CHECK(std::fabs(vU.x) < 5.0f, "no ghost horizontal velocity from teleport");
    PHYS_CHECK(std::fabs(vU.z) < 5.0f, "no ghost z velocity from teleport");
}

// ----------------------------------------------------------------------------
// 7. AngularKinematicRotation — kinematic rotation drives the upper cube's angular velocity
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicFriction, AngularKinematicRotation)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    int lower = addKinematicBox(w, glm::vec3(0.0f, 0.5f, 0.0f),
                                glm::vec3(1.0f, 0.5f, 1.0f), 0.8f);
    int upper = addDynamicBox(w, glm::vec3(0.0f, 1.5f, 0.0f),
                              glm::vec3(0.5f), 1.0f, 0.8f);

    stepForSeconds(w, 1.0f);

    const float omega = 1.0f; // rad/s
    runWithKinematicDriver(w, 3.0f, [&](float dt)
                           { advanceKinematicSpinY(w, lower, omega, dt); });

    glm::vec3 wU = w.getBody(upper).angularVelocity;
    glm::vec3 pU = w.getPosition(upper);

    PHYS_CHECK(wU.y > 0.3f * omega, "upper cube inherits y angular velocity from Kinematic");
    PHYS_CHECK(std::fabs(pU.x) < 1.0f && std::fabs(pU.z) < 1.0f,
               "upper cube does not drift off pivot");
}
