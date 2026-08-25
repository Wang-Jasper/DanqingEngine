// ============================================================================
// test_static_teleport_negative.cpp — a static body teleported horizontally must
// not drag a dynamic cube (negative test). Static velocity is always 0; friction
// requires relative tangential velocity, which is kinematic territory — even when
// setBodyPose moves the static between frames. Guards against any future patch
// deriving a differential velocity for static bodies.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    int addStaticGround(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = glm::vec3(0, -0.5f, 0);
        obb.halfExtents = glm::vec3(20.0f, 0.5f, 20.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addDynamicCube(PhysicsWorld &w, const glm::vec3 &center)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = glm::vec3(0.5f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    void stepSec(PhysicsWorld &w, float sec)
    {
        int steps = static_cast<int>(sec / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
} // namespace

// ----------------------------------------------------------------------------
// Static ground teleported horizontally every frame → cube must not be dragged
// ----------------------------------------------------------------------------
PHYS_TEST(StaticTeleportNegative, StaticGroundHorizontalTeleportDoesNotDragCube)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    int ground = addStaticGround(w);
    int cube = addDynamicCube(w, glm::vec3(0, 1.0f, 0));

    // Let the cube land and settle
    stepSec(w, 2.0f);
    glm::vec3 cubeStart = w.getPosition(cube);

    // Then teleport the static ground +X each frame (a 1 m/s "pseudo-velocity") for 3s
    const float v0 = 1.0f;
    int steps = static_cast<int>(3.0f / w.fixedTimeStep + 0.5f);
    for (int k = 0; k < steps; ++k)
    {
        OBB obb = w.getOBB(ground);
        BodyPose pose;
        pose.position = obb.center + glm::vec3(v0 * w.fixedTimeStep, 0, 0);
        pose.orientation = obb.orientation;
        w.setBodyPose(ground, pose);
        w.stepSimulation(w.fixedTimeStep);
    }

    glm::vec3 cubeEnd = w.getPosition(cube);
    glm::vec3 vCube = w.getBody(cube).velocity;

    float dx = cubeEnd.x - cubeStart.x;

    // Key negative assertion: cube horizontal displacement stays ≈ 0 (not carried)
    PHYS_CHECK(std::fabs(dx) < 0.2f,
               "Static ground horizontal teleport must NOT drag cube sideways");
    PHYS_CHECK(std::fabs(vCube.x) < 0.2f,
               "Static teleport must NOT inject horizontal velocity into cube");
}

// ----------------------------------------------------------------------------
// Static ground pushed up into the cube: this path is geometric penetration +
// normal impulse, independent of relative tangential velocity, so it SHOULD lift
// the cube (control case against the negative test above)
// ----------------------------------------------------------------------------
PHYS_TEST(StaticTeleportNegative, StaticGroundVerticalPushStillTriggersContact)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    int ground = addStaticGround(w);
    int cube = addDynamicCube(w, glm::vec3(0, 1.0f, 0));

    stepSec(w, 2.0f);
    glm::vec3 cubeStart = w.getPosition(cube);

    // Single-frame teleport of the static ground +0.5m up (clear penetration)
    {
        OBB obb = w.getOBB(ground);
        BodyPose pose;
        pose.position = obb.center + glm::vec3(0, 0.5f, 0);
        pose.orientation = obb.orientation;
        w.setBodyPose(ground, pose);
    }
    // Simulate 0.5s more
    stepSec(w, 0.5f);

    glm::vec3 cubeEnd = w.getPosition(cube);
    // Cube must be pushed up: y gain > 0.2m
    PHYS_CHECK(cubeEnd.y - cubeStart.y > 0.2f,
               "Static ground vertical push should lift cube via penetration resolution");
}
