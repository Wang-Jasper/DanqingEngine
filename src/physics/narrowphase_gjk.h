// narrowphase_gjk.h — GJK/EPA interface declarations (implementation lives in
// narrowphase_gjk.cpp). Box-Box pairs keep using the fast SAT path; this API is
// the placeholder for Sphere/Capsule/ConvexHull pairs so adding them needs no
// architecture changes.
#pragma once

#include <glm/glm.hpp>
#include "physics/Shape.h"
#include "physics/PhysicsWorld.h"

// ----------------------------------------------------------------------------
// Simplex — GJK simplex state (up to 4 points)
// ----------------------------------------------------------------------------
// Each vertex records:
//   - point    : Minkowski difference point = supportA - supportB (used by GJK)
//   - supportA : support point on shape A (world space)
//   - supportB : support point on shape B (world space)
// supportA/B let EPA resolve the closest face's barycentric weights back to
// contact points on both shapes.
// ----------------------------------------------------------------------------
struct Simplex
{
    glm::vec3 points[4] = {};   // Minkowski difference
    glm::vec3 supportA[4] = {}; // world support point on A
    glm::vec3 supportB[4] = {}; // world support point on B
    int count = 0;
};

// ----------------------------------------------------------------------------
// gjk_intersect — test whether two shapes intersect
// ----------------------------------------------------------------------------
// Params:
//   shapeA / poseA / shapeB / poseB — shapes + world poses
//   out — simplex returned on intersection (for EPA); on no intersection may
//         only hold the closest simplex
// Returns:
//   true  — intersecting (out is a simplex enclosing the origin)
//   false — separated (out is the closest simplex at GJK termination)
//
// Stub: assert(false) + return false; the real algorithm lands later.
bool gjk_intersect(const Shape &shapeA, const BodyPose &poseA,
                   const Shape &shapeB, const BodyPose &poseB,
                   Simplex &out);

// ----------------------------------------------------------------------------
// epa_manifold — expand the GJK terminal simplex into a contact manifold
// ----------------------------------------------------------------------------
// Params:
//   shapeA / poseA / shapeB / poseB — same as gjk_intersect
//   initial — simplex returned by gjk_intersect
//   out     — manifold (normal + contact points). EPA natively yields a single
//             point; multi-point manifolds need upper-level combination
//             (e.g. EPA + SAT-enhanced clipping for Hull-Hull).
// Returns:
//   true  — contact manifold produced (pointCount >= 1)
//   false — EPA failed (numerical degeneracy, rare)
//
// Stub: assert(false) + return false; the real algorithm lands later.
bool epa_manifold(const Shape &shapeA, const BodyPose &poseA,
                  const Shape &shapeB, const BodyPose &poseB,
                  const Simplex &initial,
                  ContactManifold &out);

// ----------------------------------------------------------------------------
// gjk_distance — closest distance between two shapes (for CCD)
// ----------------------------------------------------------------------------
// Params:
//   shapeA / poseA / shapeB / poseB — shapes + world poses
//   outDist     — closest distance (>= 0; 0 if intersecting)
//   outNormal   — separation direction from B to A (unit; +X fallback when intersecting)
//   outPointA   — closest point on A (world space; undefined when intersecting)
//   outPointB   — closest point on B (world space; undefined when intersecting)
// Returns:
//   true   — shapes separated, outDist > 0
//   false  — shapes intersecting or touching (outDist = 0, outNormal unreliable)
//
// Unlike gjk_intersect, terminates not on enclosing the origin but on no further
// progress of new support points along the current direction (Frank-Wolfe
// convergence), returning the simplex's closest distance to the origin. CCD
// Conservative Advancement needs this distance + normal to estimate advancing
// dist / relSpeed time.
bool gjk_distance(const Shape &shapeA, const BodyPose &poseA,
                  const Shape &shapeB, const BodyPose &poseB,
                  float &outDist, glm::vec3 &outNormal,
                  glm::vec3 &outPointA, glm::vec3 &outPointB);
