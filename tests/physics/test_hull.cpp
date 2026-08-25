// ============================================================================
// test_hull.cpp — ConvexHull support and Hull-Hull / Hull-Box intersection
// ----------------------------------------------------------------------------
// Covers ConvexMeshCache registration and hullCache backfill, shape_support
// for convex hulls, and GJK+EPA manifold validity for overlapping hull-hull /
// hull-box pairs and separated hulls.
// ============================================================================
#include "test_framework.h"
#include "physics/Shape.h"
#include "physics/narrowphase_gjk.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    // Regular tetrahedron, origin-centered with circumradius 1.
    std::vector<glm::vec3> tetrahedronVerts()
    {
        return {
            glm::vec3(1, 1, 1) * (1.0f / std::sqrt(3.0f)),
            glm::vec3(1, -1, -1) * (1.0f / std::sqrt(3.0f)),
            glm::vec3(-1, 1, -1) * (1.0f / std::sqrt(3.0f)),
            glm::vec3(-1, -1, 1) * (1.0f / std::sqrt(3.0f)),
        };
    }

    // 8-vertex cube hull, for comparing the hull path against the box path.
    std::vector<glm::vec3> boxVerts(float h)
    {
        return {
            glm::vec3(-h, -h, -h),
            glm::vec3(h, -h, -h),
            glm::vec3(-h, h, -h),
            glm::vec3(h, h, -h),
            glm::vec3(-h, -h, h),
            glm::vec3(h, -h, h),
            glm::vec3(-h, h, h),
            glm::vec3(h, h, h),
        };
    }

    BodyPose poseAt(const glm::vec3 &p, const glm::mat3 &R = glm::mat3(1.0f))
    {
        BodyPose bp;
        bp.position = p;
        bp.orientation = R;
        return bp;
    }
} // namespace

PHYS_TEST(Hull, ConvexMeshCacheRegister)
{
    // Verify registerConvexHull and the Shape.hullCache backfill.
    PhysicsWorld w;
    int hi = w.registerConvexHull(tetrahedronVerts());
    PHYS_CHECK(hi >= 0, "hull register returns valid index");
    const ConvexMeshCache *c = w.getConvexHull(hi);
    PHYS_CHECK(c != nullptr, "getConvexHull returns cache");
    PHYS_CHECK_EQ((int)c->vertices.size(), 4);
    // localAABB must cover all 4 verts (sqrt(3)/3 ≈ 0.577).
    PHYS_CHECK(c->localAABBMax.x > 0.5f, "AABB max captures max vertex");
    PHYS_CHECK(c->localAABBMin.x < -0.5f, "AABB min captures min vertex");
}

PHYS_TEST(Hull, SupportAxisAligned)
{
    // shape_support(ConvexHull) along ±X/±Y/±Z must return the matching AABB corner.
    PhysicsWorld w;
    int hi = w.registerConvexHull(boxVerts(0.5f)); // 8-vertex Box-equivalent hull (half=0.5)
    Shape s;
    s.type = ShapeType::ConvexHull;
    s.hullIndex = hi;
    s.hullCache = w.getConvexHull(hi);
    glm::vec3 supXp = shape_support(s, glm::vec3(1, 0, 0));
    PHYS_CHECK_NEAR(supXp.x, 0.5f, 1e-5f);
    glm::vec3 supXn = shape_support(s, glm::vec3(-1, 0, 0));
    PHYS_CHECK_NEAR(supXn.x, -0.5f, 1e-5f);
    glm::vec3 supCorner = shape_support(s, glm::vec3(1, 1, 1));
    // Must return the +corner vertex (0.5, 0.5, 0.5).
    PHYS_CHECK_NEAR(supCorner.x, 0.5f, 1e-5f);
    PHYS_CHECK_NEAR(supCorner.y, 0.5f, 1e-5f);
    PHYS_CHECK_NEAR(supCorner.z, 0.5f, 1e-5f);
}

PHYS_TEST(Hull, HullBoxEquivalentToBoxBox)
{
    // GJK/EPA result for the "8-vertex hull" + box must match Box-Box SAT depth and normal.
    PhysicsWorld w;
    int hi = w.registerConvexHull(boxVerts(0.5f));
    Shape hull;
    hull.type = ShapeType::ConvexHull;
    hull.hullIndex = hi;
    hull.hullCache = w.getConvexHull(hi);
    Shape box;
    box.type = ShapeType::Box;
    box.halfExtents = glm::vec3(0.5f);

    BodyPose pHull = poseAt(glm::vec3(0, 0, 0));
    BodyPose pBox = poseAt(glm::vec3(0, 0.8f, 0)); // 0.2 penetration along Y

    Simplex s;
    bool hit = gjk_intersect(hull, pHull, box, pBox, s);
    PHYS_CHECK(hit, "8-vertex hull + box must intersect");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(hull, pHull, box, pBox, s, m), "EPA ok");
    // Normal near +Y (A->B, A below).
    float cosY = glm::dot(glm::normalize(m.normal), glm::vec3(0, 1, 0));
    PHYS_CHECK(cosY > 0.95f, "normal near +Y");
    PHYS_CHECK_NEAR(m.points[0].penetration, 0.2f, 0.05f);
}

PHYS_TEST(Hull, TetrahedronBoxIntersection)
{
    // Regular tetrahedron sinks into the box from above.
    PhysicsWorld w;
    int hi = w.registerConvexHull(tetrahedronVerts());
    Shape tet;
    tet.type = ShapeType::ConvexHull;
    tet.hullIndex = hi;
    tet.hullCache = w.getConvexHull(hi);
    Shape box;
    box.type = ShapeType::Box;
    box.halfExtents = glm::vec3(1.0f, 0.5f, 1.0f);

    BodyPose pTet = poseAt(glm::vec3(0, 0.4f, 0)); // tetra center y=0.4, lowest vertex y≈-0.18
    BodyPose pBox = poseAt(glm::vec3(0, 0, 0));    // box top face y=0.5

    Simplex s;
    bool hit = gjk_intersect(tet, pTet, box, pBox, s);
    PHYS_CHECK(hit, "tetrahedron penetrating box above");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(tet, pTet, box, pBox, s, m), "EPA ok");
    PHYS_CHECK(m.points[0].penetration > 0.0f && m.points[0].penetration < 1.0f,
               "penetration in reasonable range");
}

PHYS_TEST(Hull, HullClearSeparation)
{
    PhysicsWorld w;
    int hi = w.registerConvexHull(tetrahedronVerts());
    Shape a;
    a.type = ShapeType::ConvexHull;
    a.hullIndex = hi;
    a.hullCache = w.getConvexHull(hi);
    Shape b = a;

    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(5, 0, 0)); // far apart
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(!hit, "far-apart hulls must not intersect");
}

PHYS_TEST(Hull, HullHullOverlap)
{
    // Two identical tetrahedra interpenetrating.
    PhysicsWorld w;
    int hi = w.registerConvexHull(tetrahedronVerts());
    Shape a;
    a.type = ShapeType::ConvexHull;
    a.hullIndex = hi;
    a.hullCache = w.getConvexHull(hi);
    Shape b = a;

    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0.5f, 0, 0)); // 0.5 overlap along X
    Simplex s;
    PHYS_CHECK(gjk_intersect(a, pa, b, pb, s), "overlapping hulls intersect");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(a, pa, b, pb, s, m), "EPA ok");
    PHYS_CHECK(m.points[0].penetration > 0.0f, "positive penetration");
}
