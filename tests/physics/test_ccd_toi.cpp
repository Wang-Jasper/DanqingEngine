// Conservative advancement TOI tests: static or separating pairs give TOI=1,
// head-on high-speed pairs give TOI in (0,1) matching the analytic result,
// initial penetration gives TOI=0, plus a box-box head-on case.
#include "test_framework.h"
#include "physics/ccd_toi.h"
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
    BodyPose makePose(const glm::vec3 &pos)
    {
        BodyPose p;
        p.position = pos;
        p.orientation = glm::mat3(1.0f);
        return p;
    }
} // namespace

PHYS_TEST(CcdToi, BothStaticReturnsOne)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(5, 0, 0));
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(0), sB, pB, glm::vec3(0), 1.0f, 1e-3f, 16, n);
    PHYS_CHECK_NEAR(t, 1.0f, 1e-4f);
}

PHYS_TEST(CcdToi, MovingApartReturnsOne)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(5, 0, 0));
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(-10, 0, 0),
                         sB, pB, glm::vec3(+10, 0, 0),
                         1.0f, 1e-3f, 16, n);
    PHYS_CHECK_NEAR(t, 1.0f, 1e-4f);
}

PHYS_TEST(CcdToi, SphereSphereHeadOnCollision)
{
    // Spheres at x=0 and x=5, r=1: gap 3, closing at 100 m/s, so
    // TOI = 3/100 = 0.03 of dt=1.
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(5, 0, 0));
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(100, 0, 0),
                         sB, pB, glm::vec3(0), 1.0f, 1e-3f, 32, n);
    PHYS_CHECK(t > 0.0f && t < 1.0f, "head-on collision TOI in (0,1)");
    PHYS_CHECK_NEAR(t, 0.03f, 0.002f);
    // Normal points from B to A: (-1, 0, 0).
    PHYS_CHECK_NEAR(n.x, -1.0f, 0.05f);
}

PHYS_TEST(CcdToi, InitialPenetrationReturnsZero)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(0.5f, 0, 0)); // penetrating start
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(100, 0, 0),
                         sB, pB, glm::vec3(0), 1.0f, 1e-3f, 16, n);
    PHYS_CHECK_NEAR(t, 0.0f, 1e-4f);
}

PHYS_TEST(CcdToi, BoxBoxHighSpeedThroughThinWall)
{
    // A (0.5^3 box) flies at 200 m/s toward a thin floor B (half-extents
    // (5, 0.05, 5)) at y=0.5; the gap to contact is 4.5 m, so TOI = 4.5/200 =
    // 0.0225 of dt=1.
    Shape sA = makeBox(glm::vec3(0.5f));
    Shape sB = makeBox(glm::vec3(5.0f, 0.05f, 5.0f));
    BodyPose pA = makePose(glm::vec3(-10.0f, 0.5f, 0));
    BodyPose pB = makePose(glm::vec3(0, 0.5f, 0));
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(200, 0, 0),
                         sB, pB, glm::vec3(0), 1.0f, 1e-3f, 32, n);
    PHYS_CHECK(t > 0.0f && t < 0.05f,
               "high-speed box TOI properly bounded");
}

PHYS_TEST(CcdToi, ParallelMotionNoCollision)
{
    Shape sA = makeSphere(1.0f);
    Shape sB = makeSphere(1.0f);
    BodyPose pA = makePose(glm::vec3(0, 0, 0));
    BodyPose pB = makePose(glm::vec3(5, 0, 0));
    glm::vec3 n;
    float t = computeTOI(sA, pA, glm::vec3(10, 0, 0),
                         sB, pB, glm::vec3(10, 0, 0),
                         1.0f, 1e-3f, 16, n);
    PHYS_CHECK_NEAR(t, 1.0f, 1e-4f);
}
