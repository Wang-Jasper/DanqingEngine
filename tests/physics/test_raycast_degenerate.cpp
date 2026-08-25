// ============================================================================
// test_raycast_degenerate.cpp — regression guard for rayOBBIntersect parallel-axis
// degeneracy. CP-1.4 recalibrated the parallel-axis epsilon from 1e-20f to 1e-8f:
// |localDir[k]| in [1e-20, 1e-8] previously hit the slab formula with invD = 1/tiny,
// overflowing t / polluting results with NaN.
// ============================================================================

#include "test_framework.h"
#include "physics/PhysicsWorld.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace
{
    // Static box at center with halfExtents=half, orientation R; returns the body
    // index for querying via PhysicsWorld::raycast.
    int addStaticBox(PhysicsWorld &w,
                     const glm::vec3 &center,
                     const glm::vec3 &half,
                     const glm::mat3 &R = glm::mat3(1.0f))
    {
        RigidBody rb;
        rb.bodyType = BodyType::Static;
        rb.mass = 0.0f;

        OBB obb;
        obb.center = center;
        obb.halfExtents = half;
        obb.orientation = R;
        return w.addBody(rb, obb);
    }

    glm::mat3 rotY(float deg)
    {
        return glm::mat3(glm::rotate(glm::mat4(1.0f),
                                     glm::radians(deg),
                                     glm::vec3(0, 1, 0)));
    }

    bool isFiniteVec(const glm::vec3 &v)
    {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    }
} // namespace

// ---------------------------------------------------------------------------
// 1a. Ray along world X (= OBB local X, unrotated), origin inside the slab:
//     must hit the +X face, t = (center.x + half.x) - origin.x
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, ParallelToLocalX_OriginInside)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 0.0f); // y=z=0 inside the [-1, 1] slab
    ray.direction = glm::vec3(1.0f, 0.0f, 0.0f);

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(ok, "axis-parallel ray through slab must hit");
    PHYS_CHECK_NEAR(hit.t, 4.0f, 1e-4f); // 5 - 1 = 4
    PHYS_CHECK(isFiniteVec(hit.point), "hit.point must be finite");
    PHYS_CHECK(isFiniteVec(hit.normal), "hit.normal must be finite");
    // ray hits the -X face, so the world normal should be ≈ (-1, 0, 0)
    PHYS_CHECK_NEAR(hit.normal.x, -1.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// 1b. Ray along world X with origin outside the y/z slab → must miss
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, ParallelToLocalX_OriginOutsideY)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(0.0f, 2.0f, 0.0f); // y=2 outside the [-1,1] slab
    ray.direction = glm::vec3(1.0f, 0.0f, 0.0f);

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(!ok, "ray parallel to X but outside Y slab must miss");
}

PHYS_TEST(RaycastDegenerate, ParallelToLocalX_OriginOutsideZ)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 5.0f); // z=5 outside the slab
    ray.direction = glm::vec3(1.0f, 0.0f, 0.0f);

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(!ok, "ray parallel to X but outside Z slab must miss");
}

// ---------------------------------------------------------------------------
// 2. Ray along world Y, origin inside / outside the slab
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, ParallelToLocalY_Hit)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(0.3f, 0.0f, -0.2f); // inside the x/z slab
    ray.direction = glm::vec3(0.0f, 1.0f, 0.0f);

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(ok, "Y-parallel ray through slab must hit");
    PHYS_CHECK_NEAR(hit.t, 4.0f, 1e-4f);
    PHYS_CHECK_NEAR(hit.normal.y, -1.0f, 1e-4f);
}

PHYS_TEST(RaycastDegenerate, ParallelToLocalY_Miss)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(3.0f, 0.0f, 0.0f); // x=3 outside the slab
    ray.direction = glm::vec3(0.0f, 1.0f, 0.0f);

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(!ok, "Y-parallel ray outside X slab must miss");
}

// ---------------------------------------------------------------------------
// 3. OBB rotated 45° about Y — local X axis = world R·(1,0,0). Ray fired along
//    this local X axis; the slab degeneracy test must recognize strict
//    parallelism in local space (localDir.x != 0, localDir.y = localDir.z = 0).
//
//    Box center placed 10 units along worldDir from the origin; the ray must hit
//    the local -X face, near t ≈ 10 - 1 = 9.
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, ParallelToRotatedLocalX)
{
    PhysicsWorld w;
    glm::mat3 R = rotY(45.0f);

    // world direction = OBB local +X axis transformed to world space
    glm::vec3 worldDir = glm::normalize(R * glm::vec3(1, 0, 0));
    // box center 10 units along worldDir (guarantees the ray reaches the box)
    glm::vec3 boxCenter = worldDir * 10.0f;

    addStaticBox(w, boxCenter, glm::vec3(1.0f), R);

    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 0.0f);
    ray.direction = worldDir;

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(ok, "ray parallel to rotated local X must hit");
    PHYS_CHECK_NEAR(hit.t, 9.0f, 2e-3f); // 10 - half=1
    PHYS_CHECK(isFiniteVec(hit.point), "finite point");
    PHYS_CHECK(isFiniteVec(hit.normal), "finite normal");
}

// ---------------------------------------------------------------------------
// 4. Numerical-stability case: ray "almost but not exactly" along local X.
//    |localDir.y|, |localDir.z| = 2e-8f (> the new epsilon 1e-8f, so the slab
//    branch is taken) — invD is ~5e7, so the result must stay finite and the hit
//    geometrically correct.
//
//    Under the old 1e-20f threshold this case also took the slab branch, but t
//    inflated to ~1e8-1e9, risking f32-rounding misjudgments against tmax/maxT.
//    The new threshold keeps invD below 1e8, which is acceptable.
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, NearlyParallelButAboveEpsilon)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 0.0f);
    // dir.y = 2e-8f just above the new epsilon 1e-8f; takes the slab branch, not the degenerate one
    ray.direction = glm::normalize(glm::vec3(1.0f, 2e-8f, 0.0f));

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(ok, "nearly-parallel ray (above epsilon) must still hit");
    PHYS_CHECK(std::isfinite(hit.t), "t must be finite (not Inf/NaN)");
    PHYS_CHECK(isFiniteVec(hit.point), "point must be finite");
    PHYS_CHECK(isFiniteVec(hit.normal), "normal must be finite");
    PHYS_CHECK_NEAR(hit.t, 4.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// 5. Baseline sanity: normal diagonal hit keeps this file aligned with the mainline
// ---------------------------------------------------------------------------
PHYS_TEST(RaycastDegenerate, DiagonalSanity)
{
    PhysicsWorld w;
    addStaticBox(w, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f));

    Ray ray;
    ray.origin = glm::vec3(-3.0f, -3.0f, 0.0f);
    ray.direction = glm::normalize(glm::vec3(1.0f, 1.0f, 0.0f));

    RaycastHit hit;
    bool ok = w.raycast(ray, 100.0f, 0xFFFFFFFFu, hit);
    PHYS_CHECK(ok, "diagonal ray must hit unit box at origin");
    // from (-3,-3,0) along (1,1,0)/sqrt(2), hits the x=-1 face: t = 2*sqrt(2) ≈ 2.828
    PHYS_CHECK_NEAR(hit.t, 2.0f * std::sqrt(2.0f), 1e-3f);
}
