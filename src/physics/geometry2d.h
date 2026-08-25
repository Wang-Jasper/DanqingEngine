// ============================================================================
// Pure 2D geometry primitives (convex hull, point-in-polygon, distances) for
// the sleep criteria in PhysicsWorld: projecting contact points onto the
// gravity plane and testing the COM projection against the support hull.
// No exceptions, no global state; degenerate inputs have defined returns.
// ============================================================================
#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace geometry2d
{
    // ------------------------------------------------------------------------
    // convexHull2D — Andrew's monotone chain.
    // Output: hull vertices in CCW order; collinear input yields the two
    // endpoints; <=1 point or all-coincident input is returned unchanged.
    // O(n log n) sort + O(n) scan.
    // ------------------------------------------------------------------------
    inline std::vector<glm::vec2> convexHull2D(std::vector<glm::vec2> pts)
    {
        const int n = static_cast<int>(pts.size());
        if (n <= 1)
            return pts;

        // Sort lexicographically by (x, y).
        std::sort(pts.begin(), pts.end(), [](const glm::vec2 &a, const glm::vec2 &b)
                  { return a.x < b.x || (a.x == b.x && a.y < b.y); });

        // Orientation test: positive means CCW turn.
        auto cross = [](const glm::vec2 &o, const glm::vec2 &a, const glm::vec2 &b) -> float
        {
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        };

        std::vector<glm::vec2> hull;
        hull.reserve(static_cast<size_t>(n) * 2u);

        // Lower hull.
        for (int i = 0; i < n; ++i)
        {
            while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0.0f)
                hull.pop_back();
            hull.push_back(pts[i]);
        }
        // Upper hull.
        const size_t lower_size = hull.size() + 1;
        for (int i = n - 2; i >= 0; --i)
        {
            while (hull.size() >= lower_size && cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0.0f)
                hull.pop_back();
            hull.push_back(pts[i]);
        }
        // Drop the duplicated start point.
        if (!hull.empty())
            hull.pop_back();

        return hull;
    }

    // ------------------------------------------------------------------------
    // pointToSegmentDistance2D — distance from p to segment ab.
    // ------------------------------------------------------------------------
    inline float pointToSegmentDistance2D(const glm::vec2 &p,
                                          const glm::vec2 &a,
                                          const glm::vec2 &b)
    {
        glm::vec2 ab = b - a;
        float len2 = glm::dot(ab, ab);
        if (len2 < 1e-12f)
            return glm::length(p - a); // Degenerate: a == b, distance to the point.

        float t = glm::dot(p - a, ab) / len2;
        t = std::clamp(t, 0.0f, 1.0f);
        glm::vec2 proj = a + t * ab;
        return glm::length(p - proj);
    }

    // ------------------------------------------------------------------------
    // pointInConvexPolygon2D — point in convex polygon, inclusive boundary, eps.
    // Degenerate hulls: 0 points → false; 1 point → point distance <= eps;
    // 2 points (collinear) → point-to-segment distance <= eps.
    // ------------------------------------------------------------------------
    inline bool pointInConvexPolygon2D(const std::vector<glm::vec2> &hull,
                                       const glm::vec2 &p,
                                       float eps)
    {
        const size_t n = hull.size();
        if (n == 0)
            return false;
        if (n == 1)
            return glm::length(p - hull[0]) <= eps;
        if (n == 2)
            return pointToSegmentDistance2D(p, hull[0], hull[1]) <= eps;

        // For a CCW hull, p is inside iff cross(b - a, p - a) >= -eps on every edge.
        for (size_t i = 0; i < n; ++i)
        {
            const glm::vec2 &a = hull[i];
            const glm::vec2 &b = hull[(i + 1) % n];
            glm::vec2 ab = b - a;
            glm::vec2 ap = p - a;
            float cross = ab.x * ap.y - ab.y * ap.x; // Negative outside a CCW hull.
            // Normalize: turn cross into signed distance to the edge.
            float abLen = std::sqrt(ab.x * ab.x + ab.y * ab.y);
            if (abLen < 1e-12f)
                continue; // Skip degenerate edges.
            float signedDist = cross / abLen;
            if (signedDist < -eps)
                return false;
        }
        return true;
    }

} // namespace geometry2d
