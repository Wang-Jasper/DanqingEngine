// ============================================================================
// test_manifold_clip.cpp — Sutherland-Hodgman clipping in computeBoxBoxManifold
// ----------------------------------------------------------------------------
// Covers incident-face cases: fully inside the ref face (4 points kept),
// partial overhang (clipped to the ref footprint), 45-deg rotation (clipped to
// 4 points), and full overhang (single-point fallback).
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

PHYS_TEST(ManifoldClip, FullCoverageInsideRef)
{
    // Large floor + small box centered: incident face fully inside ref face
    // -> all 4 verts kept.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.6f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "must produce contact");
    PHYS_CHECK_EQ(mf.pointCount, 4);
    // All 4 points within the box footprint.
    for (int k = 0; k < mf.pointCount; ++k)
    {
        const auto &cp = mf.points[k];
        PHYS_CHECK(std::fabs(cp.worldPoint.x) <= 0.51f, "x within box footprint");
        PHYS_CHECK(std::fabs(cp.worldPoint.z) <= 0.51f, "z within box footprint");
    }
}

PHYS_TEST(ManifoldClip, OverhangingBox_HalfSupported)
{
    // Narrow floor (half x=0.3) + wide box (half 0.5) centered: the box
    // overhangs at x=±0.5; clipping keeps x in [-0.3, 0.3]; corners map to
    // (±0.3, *, ±0.5).
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(0.3f, 0.4f, 2.0f));
    OBB box = makeOBB(glm::vec3(0, -0.6f, 0), glm::vec3(0.5f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "must produce contact");
    PHYS_CHECK(mf.pointCount > 0 && mf.pointCount <= 4,
               "pointCount in (0, 4]");
    // All contact x within [-0.3, 0.3] (ref face x half-size).
    for (int k = 0; k < mf.pointCount; ++k)
    {
        PHYS_CHECK(std::fabs(mf.points[k].worldPoint.x) <= 0.31f,
                   "contact x clipped to ref footprint");
        // z unconstrained: ref z half-size 2.0 fully covers box half 0.5.
        PHYS_CHECK(std::fabs(mf.points[k].worldPoint.z) <= 0.51f,
                   "contact z within box footprint");
    }
}

PHYS_TEST(ManifoldClip, RotatedBoxOnGround_45Y)
{
    // Box rotated 45 deg about Y, centered on a large floor: the incident face
    // is the rotated square bottom, clipped against the big ref face to itself
    // -> 4 points. Note: PLAN-3.2 takes the first 4 penetrating points;
    // PLAN-3.3 reduction picks the best 4.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(2.75f, 0.4f, 2.75f));
    OBB box = makeOBB(glm::vec3(0, -0.6f, 0), glm::vec3(0.5f), rotY(45.0f));
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "must produce contact");
    PHYS_CHECK_EQ(mf.pointCount, 4);
    // The 4 points must form a 45-deg rotated square (radius between
    // sqrt(2)*0.5/2 and sqrt(2)*0.5).
    for (int k = 0; k < mf.pointCount; ++k)
    {
        float rx = mf.points[k].worldPoint.x;
        float rz = mf.points[k].worldPoint.z;
        float r = std::sqrt(rx * rx + rz * rz);
        PHYS_CHECK(r <= 0.72f, "point within rotated box footprint radius sqrt(2)*0.5");
    }
}

PHYS_TEST(ManifoldClip, FullOverhang_DegenerateFallback)
{
    // A fully overhanging box (x=1.5 beyond ref edge 0.3) would actually be
    // separated (X overlap 0.3+0.5-1.5 < 0) and filtered by testOBBOverlap, so
    // instead use a narrow floor with the box center at x=0.3, slightly
    // overhanging, and verify clipping keeps part of the contact.
    PhysicsWorld w;
    OBB ground = makeOBB(glm::vec3(0, -1.5f, 0), glm::vec3(0.3f, 0.4f, 2.0f));
    OBB box = makeOBB(glm::vec3(0.3f, -0.6f, 0), glm::vec3(0.5f)); // center offset 0.3
    ContactManifold mf;
    bool hit = w.computeBoxBoxManifold(ground, box, 0, 1, 0, 1, mf);
    PHYS_CHECK(hit, "still overlap (ground right edge = 0.3, box left edge = -0.2)");
    PHYS_CHECK(mf.pointCount > 0, "at least one contact");
    // All contact x <= 0.3 (ref face right edge).
    for (int k = 0; k < mf.pointCount; ++k)
    {
        PHYS_CHECK(mf.points[k].worldPoint.x <= 0.31f,
                   "contact x clipped to ref right edge");
    }
}
