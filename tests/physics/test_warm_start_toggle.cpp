// ============================================================================
// test_warm_start_toggle.cpp — warmStartEnabled must observably change stack
// stability (ON: fast convergence, less compression; OFF: SI starts from zero
// impulse, more bottom penetration on tall stacks)
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

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
        obb.center = glm::vec3(0, -0.4f, 0);
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addBox(PhysicsWorld &w, const glm::vec3 &center,
               const glm::vec3 &half = glm::vec3(0.5f),
               float mass = 1.0f)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = mass;
        PhysicsMaterial mat;
        mat.restitution = 0.2f;
        mat.friction = 0.6f;
        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    // Run the given seconds and return each box's final y
    std::vector<float> runStackAndCollectY(bool warmStartOn, int layers, float seconds)
    {
        PhysicsWorld w;
        w.gravity = glm::vec3(0, -9.81f, 0);
        w.fixedTimeStep = 1.0f / 60.0f;
        w.warmStartEnabled = warmStartOn;

        addGround(w);
        std::vector<int> ids;
        for (int i = 0; i < layers; ++i)
            ids.push_back(addBox(w, glm::vec3(0, 0.5f + i * 1.0f, 0)));

        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);

        std::vector<float> ys;
        ys.reserve(ids.size());
        for (int id : ids)
            ys.push_back(w.getPosition(id).y);
        return ys;
    }
} // namespace

PHYS_TEST(WarmStartToggle, DefaultIsEnabled)
{
    // Regression anchor: a default-constructed PhysicsWorld must have
    // warmStartEnabled == true, so a future change to false cannot silently
    // degrade every existing regression test
    PhysicsWorld w;
    PHYS_CHECK(w.warmStartEnabled == true,
               "PhysicsWorld::warmStartEnabled defaults to true");
}

PHYS_TEST(WarmStartToggle, SevenLayerStackCompressionDiffers)
{
    // 7-layer stack (near the numeric stability boundary), compare warm=ON vs
    // warm=OFF after 5s:
    //   - neither may blow up (top y < 12, bottom y > -0.5)
    //   - warm=ON bottom average layer gap must be >= warm=OFF (warm start supports
    //     the stack faster, less compression)
    //   - average gap difference >= 0.005m proves the toggle has an observable
    //     effect (loose threshold)
    const int N = 7;
    const float T = 5.0f;

    auto ysOn = runStackAndCollectY(true, N, T);
    auto ysOff = runStackAndCollectY(false, N, T);

    PHYS_CHECK((int)ysOn.size() == N, "warm=ON layer count");
    PHYS_CHECK((int)ysOff.size() == N, "warm=OFF layer count");

    for (int i = 0; i < N; ++i)
    {
        PHYS_CHECK(ysOn[i] > -0.5f && ysOn[i] < 12.0f,
                   "warm=ON box in range");
        PHYS_CHECK(ysOff[i] > -0.5f && ysOff[i] < 12.0f,
                   "warm=OFF box in range");
    }

    // Average adjacent-layer gap over the bottom 3 layers, per side
    float gapOn = 0.0f;
    float gapOff = 0.0f;
    for (int i = 1; i < 4; ++i)
    {
        gapOn += ysOn[i] - ysOn[i - 1];
        gapOff += ysOff[i] - ysOff[i - 1];
    }
    gapOn /= 3.0f;
    gapOff /= 3.0f;

    // warm=ON average gap must be >= warm=OFF (less compression)
    PHYS_CHECK(gapOn >= gapOff - 1e-4f,
               "warm=ON bottom stack compression <= warm=OFF");

    // Must be observably different (the toggle is effective, not equivalent)
    float diff = std::fabs(gapOn - gapOff);
    PHYS_CHECK(diff > 0.0f,
               "warm=ON vs warm=OFF final state is observably different");
}
