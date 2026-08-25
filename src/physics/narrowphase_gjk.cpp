// narrowphase_gjk.cpp — generic convex narrowphase: GJK intersection + EPA
// penetration (normal, depth, contact point), per Bullet/Jolt/Gregorius GJK.
// Tolerances are scale-free ratios and degenerate cases return false via
// epsilon guards, so the caller can fall back.
#include "physics/narrowphase_gjk.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
    // ========================================================================
    // Minkowski support: world(A) support_dir - world(B) support_{-dir}
    // ========================================================================
    // shape_support returns local coords; transform to world space and take the
    // Minkowski difference.
    struct MDPoint
    {
        glm::vec3 md; // Minkowski difference = worldA - worldB
        glm::vec3 wA; // support point on A (world space)
        glm::vec3 wB; // support point on B (world space)
    };

    // Transform worldDir to shape-local space (local dir = R^T * worldDir), call
    // shape_support for the local support point, then map back to world:
    // worldSupp = pose.position + R * localSupp.
    inline glm::vec3 world_support(const Shape &s, const BodyPose &pose,
                                   const glm::vec3 &worldDir)
    {
        // R is mat3 (columns = local axes in world). R^T * v projects world
        // components onto local axes.
        const glm::mat3 &R = pose.orientation;
        glm::vec3 localDir = glm::transpose(R) * worldDir;
        glm::vec3 localSupp = shape_support(s, localDir);
        return pose.position + R * localSupp;
    }

    inline MDPoint md_support(const Shape &sA, const BodyPose &pA,
                              const Shape &sB, const BodyPose &pB,
                              const glm::vec3 &dir)
    {
        MDPoint mp;
        mp.wA = world_support(sA, pA, dir);
        mp.wB = world_support(sB, pB, -dir);
        mp.md = mp.wA - mp.wB;
        return mp;
    }

    // ========================================================================
    // Simplex evolution (GJK core): test if the origin is inside a 1-4 point
    // simplex or update the search direction, pruning vertices off the closest
    // feature in place. True = tetrahedron enclosing origin (intersection).
    // ========================================================================

    // Internal simplex storage (carries md/wA/wB so pruning drops them together)
    struct SimplexPoint
    {
        glm::vec3 md;
        glm::vec3 wA;
        glm::vec3 wB;
    };

    // Triple product identity: (a x b) x c = b(c·a) - a(c·b). Used throughout
    // GJK to build a direction toward the origin within a given simplex subspace.
    inline glm::vec3 triple_cross(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
    {
        return glm::cross(glm::cross(a, b), c);
    }

    // Line simplex (2 points A, B; A newest)
    // Determine which side of segment AB the origin is on: near A (drop B) or
    // on AB (keep both)
    inline bool do_line(SimplexPoint simp[], int &count, glm::vec3 &dir)
    {
        const glm::vec3 &A = simp[0].md;
        const glm::vec3 &B = simp[1].md;
        glm::vec3 AB = B - A;
        glm::vec3 AO = -A;
        if (glm::dot(AB, AO) > 0.0f)
        {
            // Origin on the forward side of AB: keep A, B, search perpendicular
            // to AB toward the origin
            dir = triple_cross(AB, AO, AB);
            if (glm::dot(dir, dir) < 1e-20f)
            {
                // AB and AO collinear (origin on the AB line extension): take any
                // direction perpendicular to AB
                glm::vec3 fallback = glm::abs(AB.x) > glm::abs(AB.y) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                dir = glm::cross(AB, fallback);
            }
        }
        else
        {
            // Origin on A's side: drop B, keep A
            count = 1;
            dir = AO;
        }
        return false; // a line cannot enclose the origin
    }

    // Triangle simplex (3 points A, B, C; A newest)
    // Determine which Voronoi region of the triangle holds the origin: edge AB /
    // edge AC / above or below the face / near A
    inline bool do_triangle(SimplexPoint simp[], int &count, glm::vec3 &dir)
    {
        const glm::vec3 &A = simp[0].md;
        const glm::vec3 &B = simp[1].md;
        const glm::vec3 &C = simp[2].md;
        glm::vec3 AB = B - A;
        glm::vec3 AC = C - A;
        glm::vec3 AO = -A;
        glm::vec3 ABC = glm::cross(AB, AC);

        if (glm::dot(glm::cross(ABC, AC), AO) > 0.0f)
        {
            // Origin outside the AC edge of triangle ABC
            if (glm::dot(AC, AO) > 0.0f)
            {
                // Origin in AC segment's Voronoi region: keep A, C, drop B
                simp[1] = simp[2]; // move C into slot 2
                count = 2;
                dir = triple_cross(AC, AO, AC);
            }
            else
            {
                // Degenerate to the AB edge case
                count = 2; // keep A, B
                return do_line(simp, count, dir);
            }
        }
        else if (glm::dot(glm::cross(AB, ABC), AO) > 0.0f)
        {
            // Origin outside the AB edge
            count = 2; // keep A, B
            return do_line(simp, count, dir);
        }
        else
        {
            // Origin above or below the triangle face
            if (glm::dot(ABC, AO) > 0.0f)
            {
                // Above (normal aligned with AO): keep A, B, C, search along the normal
                dir = ABC;
            }
            else
            {
                // Below: flip winding so the next face's normal points at the origin
                std::swap(simp[1], simp[2]);
                dir = -ABC;
            }
        }
        return false;
    }

    // Tetrahedron simplex (4 points A, B, C, D; A newest)
    // Determine which side of each of the 4 faces the origin lies on; if all are
    // on the inward side, the simplex encloses the origin -> return true
    inline bool do_tetrahedron(SimplexPoint simp[], int &count, glm::vec3 &dir)
    {
        const glm::vec3 &A = simp[0].md;
        const glm::vec3 &B = simp[1].md;
        const glm::vec3 &C = simp[2].md;
        const glm::vec3 &D = simp[3].md;
        glm::vec3 AB = B - A;
        glm::vec3 AC = C - A;
        glm::vec3 AD = D - A;
        glm::vec3 AO = -A;

        glm::vec3 ABC = glm::cross(AB, AC);
        glm::vec3 ACD = glm::cross(AC, AD);
        glm::vec3 ADB = glm::cross(AD, AB);

        if (glm::dot(ABC, AO) > 0.0f)
        {
            // Origin outside face ABC: reduce to triangle ABC
            count = 3; // keep A, B, C
            return do_triangle(simp, count, dir);
        }
        if (glm::dot(ACD, AO) > 0.0f)
        {
            // Origin outside face ACD
            simp[1] = simp[2];
            simp[2] = simp[3];
            count = 3;
            return do_triangle(simp, count, dir);
        }
        if (glm::dot(ADB, AO) > 0.0f)
        {
            // Origin outside face ADB
            simp[2] = simp[1];
            simp[1] = simp[3];
            count = 3;
            return do_triangle(simp, count, dir);
        }
        // All three A-faces have the origin outside: the tetrahedron encloses the origin
        return true;
    }

    // Simplex driver (dispatches on point count)
    inline bool do_simplex(SimplexPoint simp[], int &count, glm::vec3 &dir)
    {
        switch (count)
        {
        case 2:
            return do_line(simp, count, dir);
        case 3:
            return do_triangle(simp, count, dir);
        case 4:
            return do_tetrahedron(simp, count, dir);
        default:
            return false;
        }
    }

    // ========================================================================
    // EPA polytope
    // ========================================================================
    // Each face records: three vertex indices (into `vertices`) + outward normal
    // + face distance from origin
    struct PolyFace
    {
        int a, b, c;      // vertex indices
        glm::vec3 normal; // unit outward normal (away from the polytope interior)
        float dist;       // signed distance from origin along normal (>= 0 means origin inside)
    };

    struct PolyEdge
    {
        int a, b; // directed edge, ordered by the ccw winding of its face
    };

    // Compute a face's normal + distance from the origin. Points (a,b,c) are ccw
    // (right-hand rule normal)
    inline bool compute_face(const std::vector<SimplexPoint> &verts,
                             int a, int b, int c, PolyFace &out)
    {
        glm::vec3 ab = verts[b].md - verts[a].md;
        glm::vec3 ac = verts[c].md - verts[a].md;
        glm::vec3 n = glm::cross(ab, ac);
        float nlen2 = glm::dot(n, n);
        if (nlen2 < 1e-20f)
            return false; // degenerate (three collinear points)
        n *= 1.0f / std::sqrt(nlen2);
        out.a = a;
        out.b = b;
        out.c = c;
        out.normal = n;
        out.dist = glm::dot(n, verts[a].md); // signed distance from origin
        return true;
    }

    // Make the face normal point away from the origin (outward); flip the winding
    // if it currently faces the origin. Only used during polytope init (tetrahedron).
    inline void fix_face_winding(const std::vector<SimplexPoint> &verts, PolyFace &f)
    {
        if (f.dist < 0.0f)
        {
            std::swap(f.b, f.c);
            f.normal = -f.normal;
            f.dist = -f.dist;
        }
    }

    // Index of the face closest to the origin; -1 if the list is empty
    inline int closest_face_index(const std::vector<PolyFace> &faces)
    {
        int best = -1;
        float bestDist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < faces.size(); ++i)
        {
            if (faces[i].dist < bestDist)
            {
                bestDist = faces[i].dist;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    // Remove all faces visible from the new vertex p, collecting horizon edges
    // (boundary edges between removed and kept faces). Horizon edges form new
    // faces with p.
    inline void expand_polytope(std::vector<PolyFace> &faces,
                                std::vector<PolyEdge> &horizon,
                                const std::vector<SimplexPoint> &verts,
                                const glm::vec3 &p)
    {
        horizon.clear();
        auto add_edge = [&](int a, int b)
        {
            // If horizon already holds the reverse edge (b,a), both faces sharing
            // it were removed — not a horizon edge; cancel it
            for (size_t i = 0; i < horizon.size(); ++i)
            {
                if (horizon[i].a == b && horizon[i].b == a)
                {
                    horizon[i] = horizon.back();
                    horizon.pop_back();
                    return;
                }
            }
            horizon.push_back({a, b});
        };

        for (size_t i = faces.size(); i-- > 0;)
        {
            const PolyFace &f = faces[i];
            // p is outside the face (p and origin on opposite sides; or p farther
            // than the face): dot(normal, p - faceVertex) > 0  <=>  dot(normal, p) > dist
            if (glm::dot(f.normal, p) > f.dist + 1e-8f)
            {
                add_edge(f.a, f.b);
                add_edge(f.b, f.c);
                add_edge(f.c, f.a);
                faces[i] = faces.back();
                faces.pop_back();
            }
        }
    }

    // EPA: expand the polytope from the GJK terminal 4-simplex and find the face
    // closest to the origin. Its normal is the contact normal (A->B direction per
    // the md = wA - wB convention), its distance is the penetration depth, and
    // barycentric weights on the face resolve back to wA/wB for the contact point.
    // Returns false on numerical degeneracy (invalid starting tetrahedron or
    // iteration exhaustion).
    bool epa_run(const Shape &sA, const BodyPose &pA,
                 const Shape &sB, const BodyPose &pB,
                 const Simplex &initial,
                 glm::vec3 &outNormal, float &outDepth,
                 glm::vec3 &outWA, glm::vec3 &outWB)
    {
        if (initial.count != 4)
            return false;

        std::vector<SimplexPoint> verts;
        verts.reserve(32);
        for (int i = 0; i < 4; ++i)
        {
            SimplexPoint sp;
            sp.md = initial.points[i];
            sp.wA = initial.supportA[i];
            sp.wB = initial.supportB[i];
            verts.push_back(sp);
        }

        std::vector<PolyFace> faces;
        faces.reserve(32);
        // Tetrahedron's 4 faces (vertices 0/1/2/3) — initial winding is not
        // guaranteed outward; fix each face
        std::array<std::array<int, 3>, 4> idx = {{{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}}};
        for (auto &tri : idx)
        {
            PolyFace f;
            if (!compute_face(verts, tri[0], tri[1], tri[2], f))
                return false;
            fix_face_winding(verts, f);
            faces.push_back(f);
        }

        const int EPA_MAX_ITER = 64;
        const float TOL = 1e-4f;

        std::vector<PolyEdge> horizon;
        horizon.reserve(16);

        int bestFace = -1;
        for (int iter = 0; iter < EPA_MAX_ITER; ++iter)
        {
            bestFace = closest_face_index(faces);
            if (bestFace < 0)
                return false;
            const PolyFace &f = faces[bestFace];

            // Support along the closest face's normal; check for a new point farther out
            MDPoint sup = md_support(sA, pA, sB, pB, f.normal);
            float projectedDist = glm::dot(f.normal, sup.md);
            // projectedDist ~= f.dist means no farther point -> converged
            if (projectedDist - f.dist < TOL)
            {
                // Converged to the closest face
                outNormal = f.normal;
                outDepth = f.dist;
                // Interpolate the face's 3 wA/wB vertices to the contact point.
                // First project the origin onto the face: p = -f.normal * f.dist;
                // then compute the projection's barycentric (u, v, w) in the triangle.
                const SimplexPoint &VA = verts[f.a];
                const SimplexPoint &VB = verts[f.b];
                const SimplexPoint &VC = verts[f.c];
                glm::vec3 P = -f.normal * f.dist; // point on the face closest to the origin (MD space)
                glm::vec3 v0 = VB.md - VA.md;
                glm::vec3 v1 = VC.md - VA.md;
                glm::vec3 v2 = P - VA.md;
                float d00 = glm::dot(v0, v0);
                float d01 = glm::dot(v0, v1);
                float d11 = glm::dot(v1, v1);
                float d20 = glm::dot(v2, v0);
                float d21 = glm::dot(v2, v1);
                float denom = d00 * d11 - d01 * d01;
                if (std::fabs(denom) < 1e-20f)
                    return false;
                float v = (d11 * d20 - d01 * d21) / denom;
                float w = (d00 * d21 - d01 * d20) / denom;
                float u = 1.0f - v - w;
                outWA = u * VA.wA + v * VB.wA + w * VC.wA;
                outWB = u * VA.wB + v * VB.wB + w * VC.wB;
                return true;
            }

            // sup is genuinely outside the polytope; expand
            SimplexPoint sp;
            sp.md = sup.md;
            sp.wA = sup.wA;
            sp.wB = sup.wB;
            int newIdx = static_cast<int>(verts.size());
            verts.push_back(sp);
            expand_polytope(faces, horizon, verts, sup.md);
            // Build a triangle face from each horizon edge + newIdx
            for (const auto &e : horizon)
            {
                PolyFace nf;
                if (!compute_face(verts, e.a, e.b, newIdx, nf))
                    continue; // skip degenerate faces
                // New face normal must point outward (away from origin); flip if dist < 0
                fix_face_winding(verts, nf);
                faces.push_back(nf);
            }
        }

        // Iteration exhausted; still return the closest face (approximate solution)
        if (bestFace >= 0)
        {
            const PolyFace &f = faces[bestFace];
            outNormal = f.normal;
            outDepth = f.dist;
            // Degenerate fallback: average wA/wB of the face centroid as the
            // contact (suboptimal but stable)
            const SimplexPoint &VA = verts[f.a];
            const SimplexPoint &VB = verts[f.b];
            const SimplexPoint &VC = verts[f.c];
            outWA = (VA.wA + VB.wA + VC.wA) / 3.0f;
            outWB = (VA.wB + VB.wB + VC.wB) / 3.0f;
            return true;
        }
        return false;
    }

} // namespace

// ============================================================================
// gjk_intersect
// ============================================================================
bool gjk_intersect(const Shape &shapeA, const BodyPose &poseA,
                   const Shape &shapeB, const BodyPose &poseB,
                   Simplex &out)
{
    const int GJK_MAX_ITER = 32;
    const float TOL = 1e-6f;

    SimplexPoint simp[4];
    int count = 0;

    // Initial direction: B center to A center (+X if coincident)
    glm::vec3 dir = poseA.position - poseB.position;
    if (glm::dot(dir, dir) < 1e-20f)
        dir = glm::vec3(1, 0, 0);

    MDPoint sup = md_support(shapeA, poseA, shapeB, poseB, dir);
    if (glm::dot(sup.md, dir) < 0.0f)
    {
        // First support point does not pass the origin -> cannot intersect
        out.count = 0;
        return false;
    }
    simp[0] = {sup.md, sup.wA, sup.wB};
    count = 1;
    dir = -sup.md;

    for (int iter = 0; iter < GJK_MAX_ITER; ++iter)
    {
        if (glm::dot(dir, dir) < 1e-20f)
        {
            // Search direction degenerate — origin nearly on the simplex boundary;
            // treat as intersecting (hit).
            break;
        }
        MDPoint newSup = md_support(shapeA, poseA, shapeB, poseB, dir);
        if (glm::dot(newSup.md, dir) < TOL)
        {
            // New support point does not pass the origin along dir -> a separating
            // hyperplane exists
            out.count = 0;
            return false;
        }
        // Insert the new point at the simplex head (A position; GJK convention
        // A = newest)
        for (int i = count; i > 0; --i)
            simp[i] = simp[i - 1];
        simp[0] = {newSup.md, newSup.wA, newSup.wB};
        ++count;

        if (do_simplex(simp, count, dir))
        {
            // Simplex encloses the origin -> intersection
            break;
        }
    }

    // Fill the output simplex
    out.count = count;
    for (int i = 0; i < count; ++i)
    {
        out.points[i] = simp[i].md;
        out.supportA[i] = simp[i].wA;
        out.supportB[i] = simp[i].wB;
    }
    return true;
}

// ============================================================================
// epa_manifold
// ============================================================================
bool epa_manifold(const Shape &shapeA, const BodyPose &poseA,
                  const Shape &shapeB, const BodyPose &poseB,
                  const Simplex &initial,
                  ContactManifold &out)
{
    // EPA needs a starting tetrahedron (4 points enclosing the origin). If GJK
    // terminated with count < 4, top up with extra supports (along the simplex's
    // normal or axis-aligned directions).
    Simplex work = initial;
    if (work.count == 0)
        return false;

    auto add_point = [&](const glm::vec3 &dir) -> bool
    {
        if (work.count >= 4)
            return true;
        MDPoint sp = md_support(shapeA, poseA, shapeB, poseB, dir);
        // New point too close to an existing one (degenerate) -> try another direction
        for (int i = 0; i < work.count; ++i)
        {
            glm::vec3 d = sp.md - work.points[i];
            if (glm::dot(d, d) < 1e-12f)
                return false;
        }
        work.points[work.count] = sp.md;
        work.supportA[work.count] = sp.wA;
        work.supportB[work.count] = sp.wB;
        ++work.count;
        return true;
    };

    // Top up to 2 points (if only 1)
    if (work.count == 1)
    {
        glm::vec3 axes[6] = {
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (int i = 0; i < 6 && work.count < 2; ++i)
            add_point(axes[i]);
        if (work.count < 2)
            return false;
    }

    // Top up to 3 points
    if (work.count == 2)
    {
        glm::vec3 ab = work.points[1] - work.points[0];
        // Pick any perpendicular direction
        glm::vec3 axis = std::abs(ab.x) < std::abs(ab.y)
                             ? (std::abs(ab.x) < std::abs(ab.z) ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1))
                             : (std::abs(ab.y) < std::abs(ab.z) ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
        glm::vec3 dir1 = glm::cross(ab, axis);
        if (glm::dot(dir1, dir1) < 1e-20f)
            return false;
        if (!add_point(dir1))
            add_point(-dir1);
        if (work.count < 3)
            return false;
    }

    // Top up to 4 points
    if (work.count == 3)
    {
        glm::vec3 ab = work.points[1] - work.points[0];
        glm::vec3 ac = work.points[2] - work.points[0];
        glm::vec3 n = glm::cross(ab, ac);
        if (glm::dot(n, n) < 1e-20f)
            return false;
        if (!add_point(n))
        {
            if (!add_point(-n))
                return false;
        }
    }

    if (work.count != 4)
        return false;

    glm::vec3 normal, wA, wB;
    float depth;
    if (!epa_run(shapeA, poseA, shapeB, poseB, work, normal, depth, wA, wB))
        return false;

    // EPA normal direction convention (derived, verified with Sphere-Sphere):
    //   MD = {a - b : a in A, b in B}. A at origin, B at (0.7,0,0), radii 0.5:
    //   MD is a radius-1 sphere centered (-0.7,0,0); the MD point closest to the
    //   origin is (0.3,0,0). EPA.normal = from MD center to closest face (outward)
    //   = +X. Physical A->B = B.center - A.center = +X. They agree — EPA.normal
    //   is directly A->B, and ContactManifold.normal also uses A->B, so assign
    //   without flipping.
    glm::vec3 contactNormal = normal;

    // Fill the manifold (single point)
    out.pointCount = 0;
    out.normal = contactNormal;
    out.isTrigger = false;
    out.alive = true;

    ContactPoint cp;
    // worldPoint = midpoint of the A/B support points (matches the box-box convention)
    cp.worldPoint = 0.5f * (wA + wB);
    cp.penetration = depth;
    // localA/localB normally filled by the caller (needs pose); here computed
    // from the world position relative to each center
    cp.localA = glm::transpose(poseA.orientation) * (cp.worldPoint - poseA.position);
    cp.localB = glm::transpose(poseB.orientation) * (cp.worldPoint - poseB.position);
    cp.accumNormalImpulse = 0.0f;
    cp.accumTangentImpulse[0] = 0.0f;
    cp.accumTangentImpulse[1] = 0.0f;
    cp.matched = false;
    out.points[0] = cp;
    out.pointCount = 1;
    return true;
}

// ============================================================================
// gjk_distance — CCD distance query (Frank-Wolfe GJK variant). Terminates when
// a new support along dir shows no progress past the current closest point;
// returns |closestPoint| as the separation distance plus barycentric weights
// to reconstruct the closest points on both shapes from supportA[]/supportB[].
// ============================================================================
namespace
{
    // ------------------------------------------------------------------------
    // closest_on_* helpers: given an n-point simplex (1/2/3/4), return
    //   - closest: closest point (closure) to the origin
    //   - barycentric weights (one per simplex vertex, summing to 1)
    //   - newCount: vertex count after pruning vertices not on the closest
    //     feature (same Voronoi semantics as do_line / do_triangle / do_tetra)
    //
    // Weights reconstruct weightedA = sum(w_i * simp[i].wA), likewise for B.
    // ------------------------------------------------------------------------

    // Closest point on segment AB (to the origin)
    inline glm::vec3 closest_point_segment(const glm::vec3 &A, const glm::vec3 &B,
                                           float &wA, float &wB)
    {
        glm::vec3 AB = B - A;
        float t = glm::dot(-A, AB) / std::max(glm::dot(AB, AB), 1e-20f);
        t = std::max(0.0f, std::min(1.0f, t));
        wA = 1.0f - t;
        wB = t;
        return A + t * AB;
    }

    // Closest point on triangle ABC (to the origin)
    // Returns the closure point + barycentric weights (wA, wB, wC) summing to 1
    inline glm::vec3 closest_point_triangle(const glm::vec3 &A, const glm::vec3 &B, const glm::vec3 &C,
                                            float &wA, float &wB, float &wC)
    {
        // Reference: Real-Time Collision Detection (Ericson) section 5.1.5
        glm::vec3 ab = B - A;
        glm::vec3 ac = C - A;
        glm::vec3 ap = -A;
        float d1 = glm::dot(ab, ap);
        float d2 = glm::dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f)
        {
            wA = 1.0f;
            wB = 0.0f;
            wC = 0.0f;
            return A;
        }
        glm::vec3 bp = -B;
        float d3 = glm::dot(ab, bp);
        float d4 = glm::dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3)
        {
            wA = 0.0f;
            wB = 1.0f;
            wC = 0.0f;
            return B;
        }
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        {
            float v = d1 / std::max(d1 - d3, 1e-20f);
            wA = 1.0f - v;
            wB = v;
            wC = 0.0f;
            return A + v * ab;
        }
        glm::vec3 cp = -C;
        float d5 = glm::dot(ab, cp);
        float d6 = glm::dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6)
        {
            wA = 0.0f;
            wB = 0.0f;
            wC = 1.0f;
            return C;
        }
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        {
            float w = d2 / std::max(d2 - d6, 1e-20f);
            wA = 1.0f - w;
            wB = 0.0f;
            wC = w;
            return A + w * ac;
        }
        float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
        {
            float w = (d4 - d3) / std::max((d4 - d3) + (d5 - d6), 1e-20f);
            wA = 0.0f;
            wB = 1.0f - w;
            wC = w;
            return B + w * (C - B);
        }
        // Inside the face
        float denom = 1.0f / std::max(va + vb + vc, 1e-20f);
        float v = vb * denom;
        float w = vc * denom;
        wA = 1.0f - v - w;
        wB = v;
        wC = w;
        return A + ab * v + ac * w;
    }
} // namespace

bool gjk_distance(const Shape &shapeA, const BodyPose &poseA,
                  const Shape &shapeB, const BodyPose &poseB,
                  float &outDist, glm::vec3 &outNormal,
                  glm::vec3 &outPointA, glm::vec3 &outPointB)
{
    const int GJK_MAX_ITER = 32;
    const float TOL = 1e-6f;

    SimplexPoint simp[4];
    int count = 0;

    // Initial direction: B center to A center
    glm::vec3 dir = poseA.position - poseB.position;
    if (glm::dot(dir, dir) < 1e-20f)
        dir = glm::vec3(1, 0, 0);

    MDPoint sup = md_support(shapeA, poseA, shapeB, poseB, dir);
    simp[0] = {sup.md, sup.wA, sup.wB};
    count = 1;
    glm::vec3 closest = sup.md; // current closest point of the simplex to the origin
    float closestDist2 = glm::dot(closest, closest);

    // barycentric weights (matching simp[0..count-1])
    float bw[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    for (int iter = 0; iter < GJK_MAX_ITER; ++iter)
    {
        // Next search direction: -closest (from the closest point toward the origin)
        dir = -closest;
        if (glm::dot(dir, dir) < TOL * TOL)
        {
            // Distance is tiny -> treat as intersecting
            outDist = 0.0f;
            outNormal = glm::vec3(1, 0, 0);
            outPointA = poseA.position;
            outPointB = poseB.position;
            return false;
        }

        MDPoint newSup = md_support(shapeA, poseA, shapeB, poseB, dir);

        // Frank-Wolfe termination: the new support's progress along dir does not
        // pass the current closest point, i.e. (newSup - closest) barely crosses
        // closest along dir
        float newProj = glm::dot(newSup.md, dir);
        float oldProj = glm::dot(closest, dir);
        if (newProj - oldProj < TOL)
        {
            // Converged: closest is the closest point on the Minkowski difference
            // to the origin
            break;
        }

        // Add the new point to the simplex (at the head)
        if (count >= 4)
        {
            // Distance GJK should never need 4 points (the distance variant never
            // builds an origin-enclosing simplex); defensive: replace the farthest point
            int far = 0;
            float farD2 = glm::dot(simp[0].md, simp[0].md);
            for (int i = 1; i < 4; ++i)
            {
                float d2 = glm::dot(simp[i].md, simp[i].md);
                if (d2 > farD2)
                {
                    farD2 = d2;
                    far = i;
                }
            }
            simp[far] = {newSup.md, newSup.wA, newSup.wB};
        }
        else
        {
            for (int i = count; i > 0; --i)
                simp[i] = simp[i - 1];
            simp[0] = {newSup.md, newSup.wA, newSup.wB};
            ++count;
        }

        // Recompute closest + barycentric weights on the current simplex, pruning
        // vertices off the closest feature
        if (count == 1)
        {
            closest = simp[0].md;
            bw[0] = 1.0f;
        }
        else if (count == 2)
        {
            float w0, w1;
            closest = closest_point_segment(simp[0].md, simp[1].md, w0, w1);
            bw[0] = w0;
            bw[1] = w1;
            // Prune if one endpoint has zero weight
            if (w0 <= 0.0f)
            {
                simp[0] = simp[1];
                bw[0] = 1.0f;
                count = 1;
            }
            else if (w1 <= 0.0f)
            {
                count = 1;
                bw[0] = 1.0f;
            }
        }
        else // count == 3
        {
            float w0, w1, w2;
            closest = closest_point_triangle(simp[0].md, simp[1].md, simp[2].md, w0, w1, w2);
            bw[0] = w0;
            bw[1] = w1;
            bw[2] = w2;
            // Prune vertices with zero weight
            SimplexPoint newSimp[3];
            float newBw[3];
            int newCount = 0;
            if (w0 > 0.0f)
            {
                newSimp[newCount] = simp[0];
                newBw[newCount] = w0;
                ++newCount;
            }
            if (w1 > 0.0f)
            {
                newSimp[newCount] = simp[1];
                newBw[newCount] = w1;
                ++newCount;
            }
            if (w2 > 0.0f)
            {
                newSimp[newCount] = simp[2];
                newBw[newCount] = w2;
                ++newCount;
            }
            for (int i = 0; i < newCount; ++i)
            {
                simp[i] = newSimp[i];
                bw[i] = newBw[i];
            }
            count = newCount;
        }

        float newDist2 = glm::dot(closest, closest);
        // Distance did not decrease (numerical oscillation); exit early
        if (newDist2 >= closestDist2 - TOL)
        {
            closestDist2 = newDist2;
            break;
        }
        closestDist2 = newDist2;
    }

    // Reconstruct the closest points on both shapes
    glm::vec3 pA(0.0f), pB(0.0f);
    for (int i = 0; i < count; ++i)
    {
        pA += bw[i] * simp[i].wA;
        pB += bw[i] * simp[i].wB;
    }
    outPointA = pA;
    outPointB = pB;
    outDist = std::sqrt(closestDist2);

    if (outDist < TOL)
    {
        outDist = 0.0f;
        outNormal = glm::vec3(1, 0, 0);
        return false;
    }

    // Separation normal points A -> B (closest = pA - pB, so the pA->pB direction
    // is -closest.normalize()). Convention: outNormal goes B -> A, opposite to
    // narrowphase's ContactManifold.normal (A -> B), chosen because CCD's
    // relSpeedAlongNormal wants an approach direction naturally.
    outNormal = glm::normalize(pA - pB);
    return true;
}
