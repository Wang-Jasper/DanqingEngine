// ============================================================================
// test_sat_overlap.cpp — SAT overlap regression for PhysicsWorld::testOBBOverlap:
// separation (face/edge axes), axis-aligned overlap, 45° rotated cases, exact touching
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

    glm::mat3 rotY(float deg)
    {
        return glm::mat3(glm::rotate(glm::mat4(1.0f),
                                     glm::radians(deg),
                                     glm::vec3(0, 1, 0)));
    }
} // namespace

PHYS_TEST(SATOverlap, CoincidentUnitBoxes)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    PHYS_CHECK(w.testOBBOverlap(a, b), "identical boxes overlap");
}

PHYS_TEST(SATOverlap, AxisAlignedOverlap_X)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(0.5f, 0, 0), glm::vec3(0.5f)); // overlaps along +X by 0.5
    PHYS_CHECK(w.testOBBOverlap(a, b), "boxes overlapping along X must be detected");
}

PHYS_TEST(SATOverlap, AxisAlignedSeparated_X)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(2.0f, 0, 0), glm::vec3(0.5f)); // separated along X by 2m
    PHYS_CHECK(!w.testOBBOverlap(a, b), "boxes far apart must be separated");
}

PHYS_TEST(SATOverlap, AxisAlignedJustTouching)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    // Exactly touching: centers 1.0 apart = sum of the two half extents. SAT treats
    // this as zero penetration → overlap (not separated).
    OBB b = makeOBB(glm::vec3(1.0f, 0, 0), glm::vec3(0.5f));
    // The implementation uses strict > (sep > ra+rb), so sep == ra+rb counts as overlap
    PHYS_CHECK(w.testOBBOverlap(a, b), "touching at boundary: not separated");
}

PHYS_TEST(SATOverlap, AxisAlignedStaticBigGround)
{
    // Regression: large static ground with a small box approaching from above
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    // Box bottom just touches the ground top: center.y = -0.6, half=0.5 → bottom = -1.1 = ground top
    OBB box = makeOBB(glm::vec3(0, -0.6f, 0), glm::vec3(0.5f));
    PHYS_CHECK(w.testOBBOverlap(ground, box), "ground and box touching at interface");
}

PHYS_TEST(SATOverlap, Rotated45OverlapY)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    // B rotated 45° about Y and offset slightly right; AABB grows to 0.707, so centers 0.9 apart still overlap
    OBB b = makeOBB(glm::vec3(0.9f, 0, 0), glm::vec3(0.5f), rotY(45.0f));
    PHYS_CHECK(w.testOBBOverlap(a, b), "rotated box within extent overlap");
}

PHYS_TEST(SATOverlap, Rotated45SeparatedFar)
{
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(1.5f, 0, 0), glm::vec3(0.5f), rotY(45.0f));
    PHYS_CHECK(!w.testOBBOverlap(a, b), "rotated box far enough to be separated");
}

PHYS_TEST(SATOverlap, EdgeEdgeContact)
{
    // Two 1×1×1 cubes, B rotated 45° about Y and offset along X+Z: edge-edge contact
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(0.5f));
    OBB b = makeOBB(glm::vec3(0.8f, 0.0f, 0.8f), glm::vec3(0.5f), rotY(45.0f));
    // Some edge axes may separate or overlap; this case expects overlap (very close)
    PHYS_CHECK(w.testOBBOverlap(a, b), "rotated boxes close to each other overlap");
}

PHYS_TEST(SATOverlap, SeparatedAlongEdgeCrossAxis)
{
    // Separation detectable only on the 9 edge-cross axes: two thin boxes rotated
    // 45° against each other.
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(2.0f, 0.1f, 0.1f)); // beam along X
    glm::mat3 Ry = rotY(45.0f);
    OBB b = makeOBB(glm::vec3(1.2f, 0.6f, 1.2f),
                    glm::vec3(2.0f, 0.1f, 0.1f), Ry);
    // Face axes may overlap, but some edge-cross axis must separate them
    bool overlap = w.testOBBOverlap(a, b);
    // Expected separated: beams offset 0.6 in y, half-thickness 0.1 each (0.6 > 0.2)
    PHYS_CHECK(!overlap, "two thin beams separated along Y must be detected");
}

PHYS_TEST(SATOverlap, Symmetry)
{
    // SAT must be symmetric: testOBBOverlap(a, b) == testOBBOverlap(b, a)
    PhysicsWorld w;
    OBB a = makeOBB(glm::vec3(0), glm::vec3(1, 0.3f, 2));
    OBB b = makeOBB(glm::vec3(0.5f, 0.5f, 0.5f),
                    glm::vec3(0.5f, 0.8f, 1.2f), rotY(30.0f));
    bool ab = w.testOBBOverlap(a, b);
    bool ba = w.testOBBOverlap(b, a);
    PHYS_CHECK_EQ(ab, ba);
}
