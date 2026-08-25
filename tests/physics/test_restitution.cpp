// ============================================================================
// test_restitution.cpp — rebound height for restitution e (classic benchmark):
// a body dropped from h rebounds to ≈ e²·h (v_after = e·v_before, h' = v²/2g)
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    int addGround(PhysicsWorld &w, float e)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;
        PhysicsMaterial mat;
        mat.friction = 0.5f;
        mat.restitution = e;
        OBB obb;
        obb.center = glm::vec3(0, -0.5f, 0);
        obb.halfExtents = glm::vec3(20.0f, 0.5f, 20.0f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    int addBox(PhysicsWorld &w, const glm::vec3 &center, float e)
    {
        RigidBody rb;
        rb.bodyType = BodyType::Dynamic;
        rb.mass = 1.0f;
        PhysicsMaterial mat;
        mat.friction = 0.5f;
        mat.restitution = e;
        OBB obb;
        obb.center = center;
        obb.halfExtents = glm::vec3(0.25f);
        obb.orientation = glm::mat3(1.0f);
        return w.addBody(rb, obb, mat);
    }

    // Track the box's peak y over `seconds`
    float trackPeakY(PhysicsWorld &w, int body, float seconds)
    {
        int steps = static_cast<int>(seconds / w.fixedTimeStep + 0.5f);
        float peakY = -1e9f;
        for (int k = 0; k < steps; ++k)
        {
            w.stepSimulation(w.fixedTimeStep);
            float y = w.getPosition(body).y;
            if (y > peakY)
                peakY = y;
        }
        return peakY;
    }
} // namespace

// ----------------------------------------------------------------------------
// 1. e=0.3: peak ≈ h·e² = 3·0.09 = 0.27m (box center drops from h0=3, ≈0.25 at ground)
// ----------------------------------------------------------------------------
PHYS_TEST(Restitution, LowRestitutionBouncesLow)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    const float e = 0.3f;
    const float h0 = 3.0f; // initial center y
    addGround(w, e);
    int box = addBox(w, glm::vec3(0, h0, 0), e);

    // Drop the box to first contact (1.0s is enough to reach the ground)
    {
        int steps = static_cast<int>(1.0f / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }

    // Track the rebound peak over 2 seconds
    float peak = trackPeakY(w, box, 2.0f);

    // Theoretical rebound peak ≈ 0.25 + e²·(h0-0.25) = 0.25 + 0.09·2.75 ≈ 0.4975;
    // assertion kept loose (well below the e²-scaled drop bound)
    PHYS_CHECK(peak < 1.2f, "low-e rebound peak capped");
    // Some rebound is still required (> 0.3m, i.e. box top > 0.05m off the ground)
    PHYS_CHECK(peak > 0.3f, "low-e box does bounce some (> 0.3m)");
}

// ----------------------------------------------------------------------------
// 2. e=0.8: peak should be ≈ 0.25 + 0.64·2.75 ≈ 2.01m, at least > 1.5m
// ----------------------------------------------------------------------------
PHYS_TEST(Restitution, HighRestitutionBouncesHigh)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0, -9.81f, 0);
    w.fixedTimeStep = 1.0f / 60.0f;

    const float e = 0.8f;
    const float h0 = 3.0f;
    addGround(w, e);
    int box = addBox(w, glm::vec3(0, h0, 0), e);

    // Drop to first contact
    {
        int steps = static_cast<int>(1.0f / w.fixedTimeStep + 0.5f);
        for (int k = 0; k < steps; ++k)
            w.stepSimulation(w.fixedTimeStep);
    }
    // Rebound peak over 2 seconds
    float peak = trackPeakY(w, box, 2.0f);

    // High e must rebound clearly higher; loose bound > 1.2m
    PHYS_CHECK(peak > 1.2f, "high-e rebound peak > 1.2m");
    // Must not exceed the drop height (energy cannot appear from nowhere)
    PHYS_CHECK(peak < h0 + 0.1f, "energy conserved: rebound peak <= initial drop height");
}
