// ============================================================================
// test_manifold_single.cpp — manifold regression
// ----------------------------------------------------------------------------
// Post-PLAN-3.1: face-face contact gives pointCount 4 (incident corners
// projected to the ref plane), edge-edge/corner falls back to pointCount 1,
// separation 0. aIsRef is the orientation column most aligned with bestAxis;
// two aligned boxes take "a=ref" with 4 points at the bottom corners.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    OBB makeOBB(const glm::vec3 &center, const glm::vec3 &half,
                const glm::mat3 &R = glm::mat3(1.0f))
    {
        OBB o;
        o.center = center;
        o.halfExtents = half;
        o.orientation = R;
        return o;
    }
} // namespace

PHYS_TEST(ManifoldSingle, SeparatedReturnsFalse)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(3, 0, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(a, b, 0, 1, 0, 1, mf);
    PHYS_CHECK(!hit, "separated → no manifold");
    PHYS_CHECK_EQ(mf.pointCount, 0);
}

PHYS_TEST(ManifoldSingle, VerticalPenetrationNormal)
{
    // Face-face scene: floor (A) + box (B); pointCount 4 since PLAN-3.1.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.692f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "vertical overlap must produce manifold");
    PHYS_CHECK_EQ(mf.pointCount, 4);
    // Normal from A (floor) to B (box) = +Y.
    PHYS_CHECK_VEC3_NEAR(mf.normal, glm::vec3(0, 1, 0), 1e-5f);
    // All 4 penetrations equal (symmetric scene) and ≈ box bottom (-1.192)
    // to floor top (-1.1) = 0.092.
    for (int k = 0; k < mf.pointCount; ++k)
    {
        PHYS_CHECK_NEAR(mf.points[k].penetration, 0.092f, 5e-3f);
    }
    // bodyA/B indices pass through correctly.
    PHYS_CHECK_EQ(mf.bodyA, 0);
    PHYS_CHECK_EQ(mf.bodyB, 1);
}

PHYS_TEST(ManifoldSingle, ContactPointNearIncidentBox)
{
    // Phase 5.1 bug-fix core assertion: contact points must sit near the
    // smaller incident box, not drift to a floor corner (meters away ->
    // denomN explodes -> no restitution). With pointCount 4, all points must
    // lie in the box footprint.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.692f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "setup must produce contact");
    for (int k = 0; k < mf.pointCount; ++k)
    {
        const glm::vec3 &cp = mf.points[k].worldPoint;
        float dx = std::fabs(cp.x - box.center.x);
        float dz = std::fabs(cp.z - box.center.z);
        PHYS_CHECK(dx <= 0.51f, "contact point x within box footprint");
        PHYS_CHECK(dz <= 0.51f, "contact point z within box footprint");
        PHYS_CHECK(cp.y >= -1.15f && cp.y <= -1.05f,
                   "contact y near ref plane y=-1.1");
    }
}

PHYS_TEST(ManifoldSingle, FaceFaceFourCornerSymmetry)
{
    // Axis-aligned face-face contact: the 4 points sit near the box bottom
    // corners (±hx, ±hz). This is the core win of multi-point manifolds over
    // the Phase 5.1 single-point version: impulses at all 4 corners stabilize
    // gravity support and suppress tipping.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.692f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit && mf.pointCount == 4, "face-face must give 4 corners");
    // After subtracting box.center, the 4 points' (x, z) signs must cover all
    // 4 quadrants (--, +-, ++, -+), proving no duplicates.
    bool quadrants[4] = {false, false, false, false};
    for (int k = 0; k < 4; ++k)
    {
        float rx = mf.points[k].worldPoint.x - box.center.x;
        float rz = mf.points[k].worldPoint.z - box.center.z;
        int idx = (rx >= 0.0f ? 1 : 0) + (rz >= 0.0f ? 2 : 0);
        quadrants[idx] = true;
        // Each corner's horizontal distance from the box center ≈ half extent 0.5.
        PHYS_CHECK(std::fabs(rx) > 0.4f && std::fabs(rx) < 0.6f,
                   "|rx| near box half extent");
        PHYS_CHECK(std::fabs(rz) > 0.4f && std::fabs(rz) < 0.6f,
                   "|rz| near box half extent");
    }
    for (int q = 0; q < 4; ++q)
        PHYS_CHECK(quadrants[q], "each quadrant has one contact point");
}

PHYS_TEST(ManifoldSingle, LocalCoordsConsistency)
{
    // pose*local + center must equal worldPoint for every point (pointCount
    // can exceed 1 now, so iterate all).
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0, 0, 0), glm::vec3(1.0f, 0.5f, 1.0f));
    // A.top = 0.5; B half 0.5, center y=0.8 -> B.bottom=0.3 < A.top -> 0.2 m penetration
    OBB b = makeOBB(glm::vec3(0, 0.8f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(a, b, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "overlap required");
    for (int k = 0; k < mf.pointCount; ++k)
    {
        const auto &cp = mf.points[k];
        glm::vec3 worldA = a.center + a.orientation * cp.localA;
        glm::vec3 worldB = b.center + b.orientation * cp.localB;
        PHYS_CHECK_VEC3_NEAR(worldA, cp.worldPoint, 1e-4f);
        PHYS_CHECK_VEC3_NEAR(worldB, cp.worldPoint, 1e-4f);
    }
}

PHYS_TEST(ManifoldSingle, InitialImpulsesZero)
{
    // A fresh manifold's accumImpulse must start at 0 (warm-start inheritance
    // happens via merge).
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(0, 0.8f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(a, b, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "overlap required");
    for (int k = 0; k < mf.pointCount; ++k)
    {
        PHYS_CHECK_NEAR(mf.points[k].accumNormalImpulse, 0.0f, 1e-8f);
        PHYS_CHECK_NEAR(mf.points[k].accumTangentImpulse[0], 0.0f, 1e-8f);
        PHYS_CHECK_NEAR(mf.points[k].accumTangentImpulse[1], 0.0f, 1e-8f);
    }
}

PHYS_TEST(ManifoldSingle, NormalDirectionAtoB)
{
    // Normal must point A->B; swapping the input order flips it.
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0, -0.8f, 0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(0, 0, 0), glm::vec3(0.5f));
    ContactManifold ab;
    ContactManifold ba;
    bool hit1 = w.computeBoxBoxManifold(a, b, 0, 1, 0, 1, ab);
    bool hit2 = w.computeBoxBoxManifold(b, a, 1, 0, 1, 0, ba);
    PHYS_CHECK(hit1 && hit2, "both orderings overlap");
    // Directions must be opposite.
    PHYS_CHECK_VEC3_NEAR(ab.normal, -ba.normal, 1e-5f);
    // A->B points +Y (B.y > A.y).
    PHYS_CHECK(ab.normal.y > 0.5f, "normal goes from lower A to upper B");
}
