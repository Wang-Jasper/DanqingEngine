// Algebraic tests for the three geometry2d.h helpers across typical and
// degenerate cases; no physics involved.

#include "test_framework.h"
#include "physics/geometry2d.h"
#include <vector>

// ----------------------------------------------------------------------------
// Convex hull: 4-point square.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, ConvexHullSquare)
{
    std::vector<glm::vec2> pts = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    auto hull = geometry2d::convexHull2D(pts);
    PHYS_CHECK_EQ((int)hull.size(), 4);
}

// ----------------------------------------------------------------------------
// Convex hull: interior point filtered out, 4-point outline returned.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, ConvexHullFiltersInteriorPoint)
{
    std::vector<glm::vec2> pts = {
        {0.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 2.0f}, {0.0f, 2.0f}, {1.0f, 1.0f} // interior point, dropped
    };
    auto hull = geometry2d::convexHull2D(pts);
    PHYS_CHECK_EQ((int)hull.size(), 4);
}

// ----------------------------------------------------------------------------
// Degenerate hull: all-collinear points collapse to the two endpoints.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, ConvexHullCollinearReturnsEndpoints)
{
    std::vector<glm::vec2> pts = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    auto hull = geometry2d::convexHull2D(pts);
    PHYS_CHECK_EQ((int)hull.size(), 2);
}

// ----------------------------------------------------------------------------
// Point-to-segment distance: point beyond the endpoint.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointToSegmentDistance_Endpoint)
{
    float d = geometry2d::pointToSegmentDistance2D(
        {-1.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f});
    PHYS_CHECK_NEAR(d, 1.0f, 1e-5f);
}

// ----------------------------------------------------------------------------
// Point-to-segment distance: perpendicular projection inside the segment.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointToSegmentDistance_Perpendicular)
{
    float d = geometry2d::pointToSegmentDistance2D(
        {0.5f, 2.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f});
    PHYS_CHECK_NEAR(d, 2.0f, 1e-5f);
}

// ----------------------------------------------------------------------------
// Point in convex polygon: origin at the square center.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointInConvexPolygon_Inside)
{
    std::vector<glm::vec2> sq = {
        {-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    auto hull = geometry2d::convexHull2D(sq);
    bool inside = geometry2d::pointInConvexPolygon2D(hull, {0.0f, 0.0f}, 0.0f);
    PHYS_CHECK(inside, "origin inside unit square");
}

// ----------------------------------------------------------------------------
// Point in convex polygon: point outside.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointInConvexPolygon_Outside)
{
    std::vector<glm::vec2> sq = {
        {-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    auto hull = geometry2d::convexHull2D(sq);
    bool inside = geometry2d::pointInConvexPolygon2D(hull, {2.0f, 0.0f}, 0.01f);
    PHYS_CHECK(!inside, "point (2,0) outside unit square");
}

// ----------------------------------------------------------------------------
// Point in convex polygon: boundary tolerance.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointInConvexPolygon_BoundaryEps)
{
    std::vector<glm::vec2> sq = {
        {-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    auto hull = geometry2d::convexHull2D(sq);
    // 0.05 outside with eps 0.1: inside.
    bool inside = geometry2d::pointInConvexPolygon2D(hull, {1.05f, 0.0f}, 0.1f);
    PHYS_CHECK(inside, "point just outside but within eps should be inside");
    // Same point with eps 0.01: outside.
    bool outside = geometry2d::pointInConvexPolygon2D(hull, {1.05f, 0.0f}, 0.01f);
    PHYS_CHECK(!outside, "point outside eps should be outside");
}

// ----------------------------------------------------------------------------
// Point in convex polygon: degenerate (segment) hull, i.e. the edge-contact case.
// ----------------------------------------------------------------------------
PHYS_TEST(Geometry2D, PointInConvexPolygon_DegenerateSegment)
{
    // Collinear hull degenerates to a segment.
    std::vector<glm::vec2> hull = {
        {-1.0f, 0.0f}, {1.0f, 0.0f}};
    bool onLine = geometry2d::pointInConvexPolygon2D(hull, {0.0f, 0.0f}, 0.01f);
    PHYS_CHECK(onLine, "origin on segment is inside");
    bool offLine = geometry2d::pointInConvexPolygon2D(hull, {0.0f, 0.5f}, 0.01f);
    PHYS_CHECK(!offLine, "point off segment is outside");
}
