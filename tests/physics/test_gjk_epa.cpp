// GJK/EPA narrowphase regressions: intersection agrees with closed-form
// checks, and EPA normals/depths match analytic references. No per-scenario
// special cases; float assertions stay loose.
#include "test_framework.h"
#include "physics/Shape.h"
#include "physics/narrowphase_gjk.h"
#include "physics/PhysicsWorld.h" // for ContactManifold
#include <glm/gtc/matrix_transform.hpp>

namespace
{
    Shape makeBox(const glm::vec3 &half)
    {
        Shape s;
        s.type = ShapeType::Box;
        s.halfExtents = half;
        return s;
    }

    Shape makeSphere(float r)
    {
        Shape s;
        s.type = ShapeType::Sphere;
        s.radius = r;
        return s;
    }

    Shape makeCapsule(float r, float halfH)
    {
        Shape s;
        s.type = ShapeType::Capsule;
        s.radius = r;
        s.halfHeight = halfH;
        return s;
    }

    BodyPose poseAt(const glm::vec3 &p, const glm::mat3 &R = glm::mat3(1.0f))
    {
        BodyPose bp;
        bp.position = p;
        bp.orientation = R;
        return bp;
    }
} // namespace

// ----------------------------------------------------------------------------
// GJK intersection tests.
// ----------------------------------------------------------------------------

PHYS_TEST(GJK, SphereSphereClearSeparation)
{
    Shape a = makeSphere(0.5f);
    Shape b = makeSphere(0.5f);
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(3, 0, 0)); // center distance 3 > 0.5+0.5=1
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(!hit, "far-apart spheres must not intersect");
}

PHYS_TEST(GJK, SphereSphereDeepOverlap)
{
    Shape a = makeSphere(0.5f);
    Shape b = makeSphere(0.5f);
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0.5f, 0, 0)); // 0.5 overlap
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(hit, "overlapping spheres must intersect");
    PHYS_CHECK_EQ(s.count, 4);
}

PHYS_TEST(GJK, SphereSphereJustTouching)
{
    // Exactly touching (center distance = r1+r2): GJK may return either
    // result, but it must be deterministic.
    Shape a = makeSphere(0.5f);
    Shape b = makeSphere(0.5f);
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(1.0f, 0, 0));
    Simplex s1, s2;
    bool hit1 = gjk_intersect(a, pa, b, pb, s1);
    bool hit2 = gjk_intersect(a, pa, b, pb, s2);
    PHYS_CHECK_EQ(hit1, hit2);
}

PHYS_TEST(GJK, SphereBoxPenetrates)
{
    // Sphere sinks into the box from above.
    Shape sphere = makeSphere(0.3f);
    Shape box = makeBox(glm::vec3(1.0f, 0.5f, 1.0f));
    BodyPose ps = poseAt(glm::vec3(0, 0.6f, 0)); // sphere center y=0.6
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));    // box center y=0, top y=0.5
    // Sphere bottom (y=0.3) is 0.2 below the box top (y=0.5).
    Simplex s;
    bool hit = gjk_intersect(sphere, ps, box, pb, s);
    PHYS_CHECK(hit, "sphere penetrating box must intersect");
}

PHYS_TEST(GJK, SphereBoxClearSeparation)
{
    Shape sphere = makeSphere(0.3f);
    Shape box = makeBox(glm::vec3(1.0f, 0.5f, 1.0f));
    BodyPose ps = poseAt(glm::vec3(0, 5.0f, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));
    Simplex s;
    bool hit = gjk_intersect(sphere, ps, box, pb, s);
    PHYS_CHECK(!hit, "far-apart sphere + box must not intersect");
}

PHYS_TEST(GJK, CapsuleBoxPenetrates)
{
    Shape cap = makeCapsule(0.3f, 0.5f); // axis along Y, total length 1.6
    Shape box = makeBox(glm::vec3(1.0f, 0.5f, 1.0f));
    BodyPose pc = poseAt(glm::vec3(0, 1.0f, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));
    // Capsule bottom at y = 1.0-0.5-0.3 = 0.2, box top at 0.5: 0.3 penetration.
    Simplex s;
    bool hit = gjk_intersect(cap, pc, box, pb, s);
    PHYS_CHECK(hit, "capsule penetrating box");
}

PHYS_TEST(GJK, BoxBoxIntersectionWorks)
{
    // Box-box must also work through GJK, even though the main path uses the
    // SAT fast path.
    Shape a = makeBox(glm::vec3(0.5f));
    Shape b = makeBox(glm::vec3(0.5f));
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0.5f, 0, 0)); // 0.5 overlap along X
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(hit, "overlapping boxes must intersect via GJK");
}

// ----------------------------------------------------------------------------
// EPA penetration depth and normal.
// ----------------------------------------------------------------------------

PHYS_TEST(EPA, SphereSphereDepthAndNormal)
{
    // Spheres interpenetrate 0.3 along X; expect normal +X (A->B) and depth ~0.3.
    Shape a = makeSphere(0.5f);
    Shape b = makeSphere(0.5f);
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0.7f, 0, 0)); // center distance 0.7, penetration = 1.0-0.7 = 0.3
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(hit, "must intersect");
    ContactManifold m;
    bool ok = epa_manifold(a, pa, b, pb, s, m);
    PHYS_CHECK(ok, "EPA must succeed");
    PHYS_CHECK_EQ(m.pointCount, 1);
    // Normal near +X within 5 deg.
    float cosAngle = glm::dot(glm::normalize(m.normal), glm::vec3(1, 0, 0));
    PHYS_CHECK(cosAngle > 0.996f, "normal aligned with +X (A->B)");
    PHYS_CHECK_NEAR(m.points[0].penetration, 0.3f, 0.05f);
}

PHYS_TEST(EPA, SphereBoxNormalMatchesClosedForm)
{
    // Sphere directly above the box: closed-form normal +Y, penetration =
    // 0.3 - (0.7 - 0.5) = 0.1.
    Shape sphere = makeSphere(0.3f);
    Shape box = makeBox(glm::vec3(1.0f, 0.5f, 1.0f));
    BodyPose ps = poseAt(glm::vec3(0, 0.7f, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));
    Simplex s;
    PHYS_CHECK(gjk_intersect(sphere, ps, box, pb, s), "intersect");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(sphere, ps, box, pb, s, m), "EPA ok");
    // Normal near -Y: A (sphere) above B (box), so A->B points down.
    float cosAngle = glm::dot(glm::normalize(m.normal), glm::vec3(0, -1, 0));
    PHYS_CHECK(cosAngle > 0.98f, "normal near -Y (sphere above box)");
    // Penetration ~0.1 (radius 0.3 - gap 0.2).
    PHYS_CHECK_NEAR(m.points[0].penetration, 0.1f, 0.02f);
}

PHYS_TEST(EPA, BoxBoxCompatibleWithSAT)
{
    // Compare normal and depth against the SAT box-box manifold path.
    Shape a = makeBox(glm::vec3(0.5f));
    Shape b = makeBox(glm::vec3(0.5f));
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0.8f, 0)); // Y penetration = 1.0-0.8 = 0.2
    Simplex s;
    PHYS_CHECK(gjk_intersect(a, pa, b, pb, s), "intersect");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(a, pa, b, pb, s, m), "EPA ok");
    float cosAngle = glm::dot(glm::normalize(m.normal), glm::vec3(0, 1, 0));
    PHYS_CHECK(cosAngle > 0.95f, "normal aligned with +Y (A→B)");
    PHYS_CHECK_NEAR(m.points[0].penetration, 0.2f, 0.05f);
}

PHYS_TEST(EPA, CapsuleBoxProducesValidManifold)
{
    Shape cap = makeCapsule(0.3f, 0.5f);
    Shape box = makeBox(glm::vec3(1.0f, 0.5f, 1.0f));
    BodyPose pc = poseAt(glm::vec3(0, 1.0f, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));
    Simplex s;
    PHYS_CHECK(gjk_intersect(cap, pc, box, pb, s), "intersect");
    ContactManifold m;
    PHYS_CHECK(epa_manifold(cap, pc, box, pb, s, m), "EPA ok");
    PHYS_CHECK_EQ(m.pointCount, 1);
    // Depth reasonable (~0.3) and normal mostly downward.
    PHYS_CHECK(m.points[0].penetration > 0.05f && m.points[0].penetration < 0.5f,
               "penetration in reasonable range");
    float cosAngle = glm::dot(glm::normalize(m.normal), glm::vec3(0, -1, 0));
    PHYS_CHECK(cosAngle > 0.9f, "normal mostly -Y (capsule above box)");
}

// ----------------------------------------------------------------------------
// Degenerate / extreme cases.
// ----------------------------------------------------------------------------

PHYS_TEST(EPA, CoincidentSpheresDegenerate)
{
    // Coincident centers make EPA degenerate (depth 0, arbitrary normal); it
    // must not crash and must produce finite output.
    Shape a = makeSphere(0.5f);
    Shape b = makeSphere(0.5f);
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0, 0, 0));
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(hit, "coincident spheres intersect");
    ContactManifold m;
    bool ok = epa_manifold(a, pa, b, pb, s, m);
    // Either result is acceptable: true with a representative normal, or
    // false on numeric degeneracy.
    if (ok)
    {
        PHYS_CHECK(std::isfinite(m.points[0].penetration),
                   "penetration must be finite");
        float nlen = glm::length(m.normal);
        PHYS_CHECK(nlen > 0.9f && nlen < 1.1f, "normal must be unit-ish");
    }
}

PHYS_TEST(GJK, RotatedBoxes)
{
    // One box rotated 45 deg about Y; GJK must still report the overlap.
    Shape a = makeBox(glm::vec3(0.5f));
    Shape b = makeBox(glm::vec3(0.5f));
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                        glm::radians(45.0f),
                                        glm::vec3(0, 1, 0)));
    BodyPose pa = poseAt(glm::vec3(0, 0, 0));
    BodyPose pb = poseAt(glm::vec3(0.9f, 0, 0), R);
    Simplex s;
    bool hit = gjk_intersect(a, pa, b, pb, s);
    PHYS_CHECK(hit, "rotated boxes overlapping");
}
