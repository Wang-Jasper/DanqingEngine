// ============================================================================
// test_manifold_reduce.cpp — 4-point reduction from >4 candidates
// ----------------------------------------------------------------------------
// Reduction picks the 4 most spread-out points, deepest included. >4
// candidates arise from a rotated incident face half-overlapping the ref face
// (2 inside corners + 2 clipped intersection pairs = 6); clean overhangs yield
// exactly 4 candidates and skip reduction.
// ============================================================================
#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

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

PHYS_TEST(ManifoldReduce, FewerThanFour_NoReduction)
{
    // Narrow floor (half x=0.3) + small box (half 0.5) overhanging near
    // x=0.6: only the box edge sits on the floor, leaving 1-2 candidates —
    // the reduction path is not triggered.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(0.3f, 0.4f, 2.0f));
    // box center x=0.5, left edge x=0, right edge x=1.0; ref face x in
    // [-0.3, 0.3], so only the x in [0, 0.3] strip of the box lies on it
    OBB box = makeOBB(glm::vec3(0.5f, -0.6f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "setup must overlap");
    PHYS_CHECK(mf.pointCount >= 1 && mf.pointCount <= 4,
               "point count in [1, 4]");
    // All contact x <= 0.31 (ref face right edge).
    for (int k = 0; k < mf.pointCount; ++k)
        PHYS_CHECK(mf.points[k].worldPoint.x <= 0.31f,
                   "clipped to ref face right edge");
}

PHYS_TEST(ManifoldReduce, FullFourPointsForSymmetricFace)
{
    // Large floor + centered box, faces fully aligned: 4 candidates (all
    // bottom corners inside), used directly without reduction.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.692f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "contact");
    PHYS_CHECK_EQ(mf.pointCount, 4);
}

PHYS_TEST(ManifoldReduce, DeepestPointIncluded)
{
    // Floor tilted 10 deg about Z + axis-aligned box: the 4 bottom corners
    // penetrate to different depths. Reduction's P0 must be the deepest.
    PhysicsWorld w;
    glm::mat3 Rgz = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                          glm::radians(10.0f),
                                          glm::vec3(0, 0, 1)));
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f), Rgz);
    OBB box = makeOBB(glm::vec3(0, -0.6f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "contact");
    PHYS_CHECK(mf.pointCount >= 1, "at least one contact");
    // P0's penetration must be >= every other point's.
    float maxPen = mf.points[0].penetration;
    for (int k = 1; k < mf.pointCount; ++k)
        maxPen = std::max(maxPen, mf.points[k].penetration);
    // By the reduction definition, P0 (output index 0) is the deepest.
    PHYS_CHECK_NEAR(mf.points[0].penetration, maxPen, 1e-5f);
}

PHYS_TEST(ManifoldReduce, FourPointsAreDistinct)
{
    // Large floor + 45-deg rotated box may clip to >= 4 candidates; reduction
    // must return 4 distinct points.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.692f, 0), glm::vec3(0.5f), rotY(45.0f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "contact");
    PHYS_CHECK_EQ(mf.pointCount, 4);
    // Pairwise distances > 0.1 m (rotated-box diagonal ≈ 1 m).
    for (int i = 0; i < mf.pointCount; ++i)
        for (int j = i + 1; j < mf.pointCount; ++j)
        {
            glm::vec3 d = mf.points[i].worldPoint - mf.points[j].worldPoint;
            PHYS_CHECK(glm::length(d) > 0.1f, "points must be distinct");
        }
}
