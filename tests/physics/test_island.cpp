// ============================================================================
// test_island.cpp — island-construction regression tests
// ----------------------------------------------------------------------------
// Verifies buildIslands() connectivity (auto-invoked at the end of
// detectCollisions): isolated bodies -> separate islands; a contact chain ->
// one island; a static floor links every box touching it. No sleep/resting
// involvement; 1-2 steps per test suffice to produce manifolds.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"

namespace
{
    // Static box floor.
    int addStaticFloor(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half)
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

    // Dynamic box.
    int addDynBox(PhysicsWorld &w, const glm::vec3 &center, const glm::vec3 &half,
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

    // Run N frames so detectCollisions produces manifolds.
    void runFrames(PhysicsWorld &w, int frames)
    {
        for (int i = 0; i < frames; ++i)
            w.stepSimulation(w.fixedTimeStep);
    }

    // True when a and b share a valid (>= 0) islandId.
    bool sameIsland(const std::vector<int> &ids, int a, int b)
    {
        if (a < 0 || b < 0 || a >= (int)ids.size() || b >= (int)ids.size())
            return false;
        return ids[a] >= 0 && ids[a] == ids[b];
    }
} // namespace

// ----------------------------------------------------------------------------
// A: 3 isolated bodies, no contact -> each is its own island
// ----------------------------------------------------------------------------
PHYS_TEST(Island, ThreeIsolatedBodiesAreSeparate)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f); // disable gravity so bodies stay airborne and never collide
    int b0 = addDynBox(w, glm::vec3(-5, 0, 0), glm::vec3(0.5f));
    int b1 = addDynBox(w, glm::vec3(0, 0, 0), glm::vec3(0.5f));
    int b2 = addDynBox(w, glm::vec3(5, 0, 0), glm::vec3(0.5f));

    runFrames(w, 2);

    const auto &ids = w.debugIslandIds();
    PHYS_CHECK_EQ((int)ids.size(), 3);
    PHYS_CHECK(ids[b0] >= 0 && ids[b1] >= 0 && ids[b2] >= 0,
               "all active bodies get a valid islandId");
    PHYS_CHECK(!sameIsland(ids, b0, b1), "b0 / b1 in different islands");
    PHYS_CHECK(!sameIsland(ids, b1, b2), "b1 / b2 in different islands");
    PHYS_CHECK(!sameIsland(ids, b0, b2), "b0 / b2 in different islands");
    PHYS_CHECK_EQ(w.debugIslandCount(), 3);
}

// ----------------------------------------------------------------------------
// B: 3 boxes in a contact chain -> 1 island. No floor; boxes start slightly
// interpenetrating so detectCollisions is guaranteed to hit.
// ----------------------------------------------------------------------------
PHYS_TEST(Island, ThreeStackedBoxesOneIsland)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f);
    // 3 unit cubes stacked along Y; center spacing 0.95 (< 1.0 = half-height
    // sum) forces penetration.
    int b0 = addDynBox(w, glm::vec3(0, 0.0f, 0), glm::vec3(0.5f));
    int b1 = addDynBox(w, glm::vec3(0, 0.95f, 0), glm::vec3(0.5f));
    int b2 = addDynBox(w, glm::vec3(0, 1.90f, 0), glm::vec3(0.5f));

    runFrames(w, 1);

    const auto &ids = w.debugIslandIds();
    PHYS_CHECK(sameIsland(ids, b0, b1), "b0 and b1 in same island (contact)");
    PHYS_CHECK(sameIsland(ids, b1, b2), "b1 and b2 in same island (contact)");
    PHYS_CHECK(sameIsland(ids, b0, b2), "b0 and b2 transitively same island");
    PHYS_CHECK_EQ(w.debugIslandCount(), 1);
}

// ----------------------------------------------------------------------------
// C: A-B contact + C isolated -> 2 islands
// ----------------------------------------------------------------------------
PHYS_TEST(Island, TwoContactingPlusOneIsolated)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f);
    int bA = addDynBox(w, glm::vec3(0, 0.0f, 0), glm::vec3(0.5f));
    int bB = addDynBox(w, glm::vec3(0.95f, 0.0f, 0), glm::vec3(0.5f)); // penetrates A along X
    int bC = addDynBox(w, glm::vec3(10, 0.0f, 0), glm::vec3(0.5f));    // far away

    runFrames(w, 1);

    const auto &ids = w.debugIslandIds();
    PHYS_CHECK(sameIsland(ids, bA, bB), "A and B share island");
    PHYS_CHECK(!sameIsland(ids, bA, bC), "A and C in different islands");
    PHYS_CHECK(!sameIsland(ids, bB, bC), "B and C in different islands");
    PHYS_CHECK_EQ(w.debugIslandCount(), 2);
}

// ----------------------------------------------------------------------------
// D: static floor + 3 dynamic boxes all touching it -> 1 island; the static
// body connects them transitively.
// ----------------------------------------------------------------------------
PHYS_TEST(Island, StaticFloorConnectsDynamicBoxes)
{
    PhysicsWorld w;
    w.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    // Floor top at y = 0.
    int bFloor = addStaticFloor(w, glm::vec3(0, -0.5f, 0), glm::vec3(5.0f, 0.5f, 5.0f));
    // Boxes start with bottoms slightly below 0 (penetrating the floor) so
    // the first frame contacts.
    int b0 = addDynBox(w, glm::vec3(-2.0f, 0.48f, 0), glm::vec3(0.5f));
    int b1 = addDynBox(w, glm::vec3(0.0f, 0.48f, 0), glm::vec3(0.5f));
    int b2 = addDynBox(w, glm::vec3(2.0f, 0.48f, 0), glm::vec3(0.5f));

    runFrames(w, 1);

    const auto &ids = w.debugIslandIds();
    // All 3 boxes share the floor's island -> they share each other's.
    PHYS_CHECK(sameIsland(ids, bFloor, b0), "floor and b0 share island");
    PHYS_CHECK(sameIsland(ids, bFloor, b1), "floor and b1 share island");
    PHYS_CHECK(sameIsland(ids, bFloor, b2), "floor and b2 share island");
    PHYS_CHECK(sameIsland(ids, b0, b2), "b0 and b2 transitively same island via floor");
    PHYS_CHECK_EQ(w.debugIslandCount(), 1);

    // Island holds 4 bodies (3 dynamic + 1 static).
    int idx = ids[bFloor];
    PHYS_CHECK(idx >= 0, "floor has a valid island");
    const Island &isl = w.debugIsland(idx);
    PHYS_CHECK_EQ((int)isl.bodies.size(), 4);
    PHYS_CHECK((int)isl.manifoldIndices.size() >= 3,
               "at least 3 manifolds (floor-b0, floor-b1, floor-b2)");
}
