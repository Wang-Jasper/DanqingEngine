// GJK distance query tests: separated pairs report the analytic distance with
// the normal pointing at A; touching and overlapping pairs return false.
#include "test_framework.h"
#include "physics/narrowphase_gjk.h"
#include "physics/Shape.h"

namespace
{
    Shape makeSphere(float r)
    {
        Shape s;
        s.type = ShapeType::Sphere;
        s.radius = r;
        return s;
    }
    Shape makeBox(const glm::vec3 &half)
    {
        Shape s;
        s.type = ShapeType::Box;
        s.halfExtents = half;
        return s;
    }
    BodyPose makePose(const glm::vec3 &pos, const glm::mat3 &R = glm::mat3(1.0f))
    {
        BodyPose p;
        p.position = pos;
        p.orientation = R;
        return p;
    }
} // namespace

PHYS_TEST(GjkDistance, SphereSphereSeparated)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(5, 0, 0));
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    PHYS_CHECK(ok, "separated spheres return true");
    PHYS_CHECK_NEAR(d, 3.0f, 0.01f);
    // Normal points from B to A: (-1, 0, 0) for A(0,0,0) / B(5,0,0).
    PHYS_CHECK_NEAR(n.x, -1.0f, 0.02f);
    PHYS_CHECK_NEAR(n.y, 0.0f, 0.02f);
    PHYS_CHECK_NEAR(n.z, 0.0f, 0.02f);
}

PHYS_TEST(GjkDistance, SphereSphereTouching)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(2.0f, 0, 0)); // just touching
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    // Contact: ~0 distance, returns false.
    PHYS_CHECK(!ok || d < 0.05f, "touching spheres: near-zero distance");
}

PHYS_TEST(GjkDistance, SphereSphereOverlapping)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(1.5f, 0, 0)); // 0.5 penetration
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    PHYS_CHECK(!ok, "overlapping spheres return false");
}

PHYS_TEST(GjkDistance, BoxBoxAxisSeparated)
{
    Shape sA = makeBox(glm::vec3(0.5f));
    Shape sB = makeBox(glm::vec3(0.5f));
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(3.0f, 0, 0));
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    PHYS_CHECK(ok, "separated boxes return true");
    // Two 0.5 boxes, centers 3 apart: distance = 3 - 1 = 2.
    PHYS_CHECK_NEAR(d, 2.0f, 0.01f);
    PHYS_CHECK_NEAR(n.x, -1.0f, 0.02f);
}

PHYS_TEST(GjkDistance, BoxBoxDiagonalSeparated)
{
    Shape sA = makeBox(glm::vec3(0.5f));
    Shape sB = makeBox(glm::vec3(0.5f));
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(3.0f, 3.0f, 0));
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    PHYS_CHECK(ok, "diagonal separated boxes return true");
    // Nearest corners A(0.5,0.5,0) and B(2.5,2.5,0): distance = 2*sqrt(2).
    PHYS_CHECK_NEAR(d, 2.8284f, 0.02f);
}

PHYS_TEST(GjkDistance, BoxBoxOverlap)
{
    Shape sA = makeBox(glm::vec3(0.5f));
    Shape sB = makeBox(glm::vec3(0.5f));
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(0.5f, 0, 0)); // 0.5 overlap
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sA, pA, sB, pB, d, n, ptA, ptB);
    PHYS_CHECK(!ok, "overlapping boxes return false");
}

PHYS_TEST(GjkDistance, SphereBoxMixed)
{
    Shape sphere = makeSphere(0.5f);
    Shape box = makeBox(glm::vec3(0.5f));
    BodyPose pS = makePose(glm::vec3(3, 0, 0));
    BodyPose pB = makePose(glm::vec3(0, 0, 0));
    float d;
    glm::vec3 n, ptA, ptB;
    bool ok = gjk_distance(sphere, pS, box, pB, d, n, ptA, ptB);
    PHYS_CHECK(ok, "separated sphere-box return true");
    // Sphere (r=0.5) at x=3, box (half 0.5) at origin: box face at x=0.5,
    // sphere surface at x=2.5, so distance = 2.0.
    PHYS_CHECK_NEAR(d, 2.0f, 0.02f);
}
