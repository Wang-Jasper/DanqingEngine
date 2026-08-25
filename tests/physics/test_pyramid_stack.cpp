// ============================================================================
// test_pyramid_stack.cpp — pyramid stack stability (Box2D / Jolt / PEEL classic)
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace
{
    int addGround(PhysicsWorld &w)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.0f;
        OBB obb;
        obb.center = glm::vec3(0, -0.5f, 0);
        obb.halfExtents = glm::vec3(30.0f, 0.5f, 30.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addPyramidBox(PhysicsWorld &w, const glm::vec3 &center)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.friction = 0.8f;
        mat.restitution = 0.0f;
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

    // Build a pyramid of base width N; ids ordered bottom layer first, left to right
    std::vector<int> buildPyramid(PhysicsWorld &w, int N)
    {
        std::vector<int> ids;
        for (int k = 0; k < N; ++k)
        {
            int countThisLayer = N - k;
            // Center the layer: each box 1.0m wide, layer spans countThisLayer*1.0m
            float xStart = -(countThisLayer - 1) * 0.5f;
            float y = 0.5f + k * 1.0f;
            for (int i = 0; i < countThisLayer; ++i)
            {
                float x = xStart + i * 1.0f;
                ids.push_back(addPyramidBox(w, glm::vec3(x, y, 0)));
            }
        }
        return ids;
    }
} // namespace

PHYS_TEST(PyramidStack, PyramidN5Stable)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    std::vector<int> ids = buildPyramid(w, 5); // 5+4+3+2+1 = 15 boxes

    stepSec(w, 20.0f);

    for (int id : ids)
    {
        glm::vec3 v = w.getBody(id).velocity;
        PHYS_CHECK(glm::length(v) < 0.5f, "pyramid box velocity ~0 after 20s");
    }

    // apex (ids.back()) expected near y ≈ 4.5: 5 layers, top layer k=4 → y=4.5
    glm::vec3 pTop = w.getPosition(ids.back());
    PHYS_CHECK(pTop.y > 3.5f && pTop.y < 5.5f,
               "pyramid apex stays near expected height");

    // bottom layer must not sink into the ground (y should stay ≈ 0.5)
    for (int i = 0; i < 5; ++i)
    {
        glm::vec3 p = w.getPosition(ids[i]);
        PHYS_CHECK(p.y > 0.3f && p.y < 0.8f,
                   "bottom-row box not sunk into ground");
    }
}

PHYS_TEST(PyramidStack, PyramidN8Stable)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    addGround(w);
    std::vector<int> ids = buildPyramid(w, 8); // 8+7+...+1 = 36 boxes

    stepSec(w, 30.0f);

    // Loose: edge boxes may fall, but the core stack must stay stable
    int settledCount = 0;
    for (int id : ids)
    {
        glm::vec3 v = w.getBody(id).velocity;
        if (glm::length(v) < 0.8f)
            ++settledCount;
    }
    PHYS_CHECK(settledCount >= (int)ids.size() * 80 / 100,
               "at least 80% of pyramid boxes stable");

    glm::vec3 pTop = w.getPosition(ids.back());
    PHYS_CHECK(pTop.y > 2.0f && pTop.y < 10.0f,
               "pyramid apex roughly within expected altitude band");
    PHYS_CHECK(std::fabs(pTop.x) < 5.0f && std::fabs(pTop.z) < 5.0f,
               "apex not flown off sideways");
}
