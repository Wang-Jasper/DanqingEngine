// ============================================================================
// test_kinematic_stack_drag.cpp — kinematic base dragging stacked dynamic cubes
// ----------------------------------------------------------------------------
// Automates the manually-verified "cube stack + kinematic plane drag" scenario:
// momentum transfers layer by layer through tangential friction impulses (base
// -> cube1 -> ...) with no kinematic-neighbor special case — only the generic
// friction-cone constraint plus island wake propagation.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <functional>

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
        obb.center = glm::vec3(0, -1.0f, 0);
        obb.halfExtents = glm::vec3(20.0f, 0.1f, 20.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addKinematicBase(PhysicsWorld &w, const glm::vec3 &center,
                         const glm::vec3 &half)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Kinematic;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.05f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
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

    void runDriven(PhysicsWorld &w, float sec,
                   const std::function<void(float dt)> &driver)
    {
        int steps = static_cast<int>(sec / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
        {
            driver(w.fixedTimeStep);
            w.stepSimulation(w.fixedTimeStep);
        }
    }

    void advanceKinematicLinear(PhysicsWorld &w, int bodyIdx,
                                const glm::vec3 &vel, float dt)
    {
        OBB obb = w.getOBB(bodyIdx);
        BodyPose pose;
        pose.position = obb.center + vel * dt;
        pose.orientation = obb.orientation;
        w.setBodyPose(bodyIdx, pose);
    }
} // namespace

// ----------------------------------------------------------------------------
// 1. 3-layer stack: kinematic base drags at constant speed for 3 s; every cube
// must move > 1 m horizontally.
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicStackDrag, ThreeCubeStackDragsByKinematicFloor)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    // Kinematic base top at y=0.5, wide enough that cubes stay on for 3 s.
    int base = addKinematicBase(w, glm::vec3(0, 0.0f, 0), glm::vec3(8.0f, 0.5f, 2.0f));
    // Three dynamic cubes on the base at y = 1.0, 2.0, 3.0.
    int c1 = addDynamicCube(w, glm::vec3(0, 1.0f, 0));
    int c2 = addDynamicCube(w, glm::vec3(0, 2.0f, 0));
    int c3 = addDynamicCube(w, glm::vec3(0, 3.0f, 0));

    stepSec(w, 2.0f); // settle

    glm::vec3 p1Start = w.getPosition(c1);
    glm::vec3 p2Start = w.getPosition(c2);
    glm::vec3 p3Start = w.getPosition(c3);

    const float v0 = 1.0f;
    runDriven(w, 3.0f, [&](float dt)
              { advanceKinematicLinear(w, base, glm::vec3(v0, 0, 0), dt); });

    glm::vec3 p1 = w.getPosition(c1);
    glm::vec3 p2 = w.getPosition(c2);
    glm::vec3 p3 = w.getPosition(c3);
    glm::vec3 v1 = w.getBody(c1).velocity;
    glm::vec3 v2 = w.getBody(c2).velocity;
    glm::vec3 v3 = w.getBody(c3).velocity;

    float dx1 = p1.x - p1Start.x;
    float dx2 = p2.x - p2Start.x;
    float dx3 = p3.x - p3Start.x;

    // Assert 1: every layer is carried; tolerances taper off because upper
    // layers lag slightly.
    PHYS_CHECK(dx1 > 1.5f, "cube1 (bottom of stack) carried > 1.5m");
    PHYS_CHECK(dx2 > 1.0f, "cube2 (middle) carried > 1.0m");
    PHYS_CHECK(dx3 > 0.8f, "cube3 (top) carried > 0.8m");

    // Assert 2: every layer gains forward tangential velocity.
    PHYS_CHECK(v1.x > 0.3f * v0, "cube1 tangential velocity follows base");
    PHYS_CHECK(v2.x > 0.2f * v0, "cube2 tangential velocity follows cube1");
    PHYS_CHECK(v3.x > 0.1f * v0, "cube3 tangential velocity follows cube2");

    // Assert 3: stack intact (vertical order kept, y spacing ≈ 1.0).
    PHYS_CHECK(p2.y - p1.y > 0.85f && p2.y - p1.y < 1.15f,
               "cube2 still sits atop cube1");
    PHYS_CHECK(p3.y - p2.y > 0.85f && p3.y - p2.y < 1.15f,
               "cube3 still sits atop cube2");
}

// ----------------------------------------------------------------------------
// 2. 4-layer stack stress test
// ----------------------------------------------------------------------------
PHYS_TEST(KinematicStackDrag, FourCubeStackDragsByKinematicFloor)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addStaticGround(w);
    int base = addKinematicBase(w, glm::vec3(0, 0.0f, 0), glm::vec3(8.0f, 0.5f, 2.0f));
    int c1 = addDynamicCube(w, glm::vec3(0, 1.0f, 0));
    int c2 = addDynamicCube(w, glm::vec3(0, 2.0f, 0));
    int c3 = addDynamicCube(w, glm::vec3(0, 3.0f, 0));
    int c4 = addDynamicCube(w, glm::vec3(0, 4.0f, 0));

    stepSec(w, 2.5f); // 4 layers need longer to settle

    glm::vec3 p1Start = w.getPosition(c1);
    glm::vec3 p4Start = w.getPosition(c4);

    const float v0 = 1.0f;
    runDriven(w, 3.0f, [&](float dt)
              { advanceKinematicLinear(w, base, glm::vec3(v0, 0, 0), dt); });

    glm::vec3 p1 = w.getPosition(c1);
    glm::vec3 p4 = w.getPosition(c4);

    // Bottom cube follows the base most tightly.
    PHYS_CHECK(p1.x - p1Start.x > 1.5f, "bottom cube carried > 1.5m");
    // Top only needs modest displacement (4-layer momentum transfer; loose 0.3 m bound).
    PHYS_CHECK(p4.x - p4Start.x > 0.3f,
               "top cube carried > 0.3m via 4-layer momentum transfer");

    // Stack structure roughly intact (top cube y still near ~3.5).
    PHYS_CHECK(p4.y > 2.5f && p4.y < 4.5f,
               "top cube still near expected stack height");
}
