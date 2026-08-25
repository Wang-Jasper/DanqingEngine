#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

// ============================================================================
// ShapeType / Shape — generic shape abstraction, starting with Box.
// Adding Sphere/Capsule/Hull later only requires new shape_* branches, not
// changes to PhysicsWorld's data flow. Non-Box branches assert(false) until
// implemented.
// ============================================================================

enum class ShapeType : uint8_t
{
    Box = 0,
    Sphere = 1,
    Capsule = 2,
    ConvexHull = 3,
    Compound = 4,   // ECS-level sugar; physics realizes it as multiple shapes per body,
                    // so this type is only a placeholder
};

// ============================================================================
// ConvexMeshCache — shared vertex storage for ConvexHull shapes; multiple Shapes
// may reference one cache. Only the vertex list is stored: support does an O(n)
// scan, fine for n < 100. Lifetime is managed by PhysicsWorld (or the caller);
// this struct owns nothing.
// ============================================================================
struct ConvexMeshCache
{
    std::vector<glm::vec3> vertices; // Local-space vertices, relative to the shape center
    // Precomputed local-space AABB, used to accelerate shape_world_aabb.
    glm::vec3 localAABBMin = glm::vec3(0.0f);
    glm::vec3 localAABBMax = glm::vec3(0.0f);
};

struct Shape
{
    ShapeType type = ShapeType::Box;

    // --- Primitive parameters, used per type ---
    glm::vec3 halfExtents = glm::vec3(0.5f);    // Box
    float radius = 0.5f;                        // Sphere / Capsule
    float halfHeight = 0.5f;                    // Capsule
    int hullIndex = -1;                         // ConvexHull → ConvexMeshCache index (ECS-side reference)
    const ConvexMeshCache *hullCache = nullptr; // ConvexHull fast pointer, backfilled when PhysicsWorld registers it
    // Compound is realized as multiple shapes per body (CollidersComponent), so
    // Shape needs no extra field; the enum value is only ECS-level sugar.
};

// Body pose in world space, decoupled from the shape so the two evolve
// independently (one pose per body, even with several shapes).
struct BodyPose
{
    glm::vec3 position = glm::vec3(0.0f);
    glm::mat3 orientation = glm::mat3(1.0f);
};

// World-space axis-aligned box used for broadphase candidates.
struct AABB
{
    glm::vec3 min = glm::vec3(0.0f);
    glm::vec3 max = glm::vec3(0.0f);
};

// ============================================================================
// Generic shape free functions, decoupling shape math from PhysicsWorld's loop:
// local diagonal inertia, world-space AABB (broadphase), local support point
// (GJK/EPA). Only the Box branch is implemented; other types assert(false).
// ============================================================================

// Box inertia: I = m/12 * (other two dimensions squared), with w=2hx, h=2hy,
// d=2hz; clamped to minI to guard degenerate shapes.
inline glm::vec3 shape_inertia_local(const Shape &s, float mass)
{
    constexpr float minI = 1e-6f;
    switch (s.type)
    {
    case ShapeType::Box:
    {
        if (mass <= 0.0f)
            return glm::vec3(0.0f);
        float w = 2.0f * s.halfExtents.x;
        float h = 2.0f * s.halfExtents.y;
        float d = 2.0f * s.halfExtents.z;
        float m12 = mass / 12.0f;
        glm::vec3 I(m12 * (h * h + d * d),
                    m12 * (w * w + d * d),
                    m12 * (w * w + h * h));
        I.x = std::max(I.x, minI);
        I.y = std::max(I.y, minI);
        I.z = std::max(I.z, minI);
        return I;
    }
    case ShapeType::Sphere:
    {
        // Solid sphere: I = 2/5 * m * r², equal on all axes.
        if (mass <= 0.0f)
            return glm::vec3(0.0f);
        float I = 0.4f * mass * s.radius * s.radius;
        I = std::max(I, minI);
        return glm::vec3(I);
    }
    case ShapeType::Capsule:
    {
        // Capsule = cylinder (height 2h, radius r) plus two hemispheres (r).
        // Long axis is local Y. Mass is split between cylinder and spheres by
        // volume; closed-form diagonal inertia about the COM:
        //   mCyl = m * Vcyl / (Vcyl + Vsph)
        //   Iy = mCyl * r²/2 + mSph * 2r²/5
        //   Ix = Iz = mCyl * (r²/4 + h²/3) + mSph * (2r²/5 + h² + 3hr/4)
        if (mass <= 0.0f)
            return glm::vec3(0.0f);
        float r = s.radius;
        float h = s.halfHeight;
        float Vcyl = 3.14159265f * r * r * (2.0f * h);
        float Vsph = (4.0f / 3.0f) * 3.14159265f * r * r * r;
        float Vtot = std::max(Vcyl + Vsph, 1e-9f);
        float mCyl = mass * Vcyl / Vtot;
        float mSph = mass - mCyl;
        float Iy = mCyl * 0.5f * r * r + mSph * 0.4f * r * r;
        float Ixz_cyl = mCyl * (0.25f * r * r + (h * h) / 3.0f);
        float Ixz_sph = mSph * (0.4f * r * r + h * h + 0.75f * h * r);
        float Ix = Ixz_cyl + Ixz_sph;
        Iy = std::max(Iy, minI);
        Ix = std::max(Ix, minI);
        return glm::vec3(Ix, Iy, Ix);
    }
    case ShapeType::ConvexHull:
    {
        // Conservative approximation: treat the hull's local AABB as an equivalent
        // box. A true polyhedral tensor is deferred; the box approximation avoids
        // rotation instability from tiny inertia values.
        if (mass <= 0.0f || s.hullCache == nullptr)
            return glm::vec3(minI);
        glm::vec3 ext = s.hullCache->localAABBMax - s.hullCache->localAABBMin;
        float w = std::max(ext.x, 1e-3f);
        float h = std::max(ext.y, 1e-3f);
        float d = std::max(ext.z, 1e-3f);
        float m12 = mass / 12.0f;
        glm::vec3 I(m12 * (h * h + d * d),
                    m12 * (w * w + d * d),
                    m12 * (w * w + h * h));
        I.x = std::max(I.x, minI);
        I.y = std::max(I.y, minI);
        I.z = std::max(I.z, minI);
        return I;
    }
    case ShapeType::Compound:
    default:
        // Compound is dissolved into individual shapes before reaching here; each
        // child calls this function separately.
        assert(false && "shape_inertia_local: Compound unused at physics layer");
        return glm::vec3(minI);
    }
}

// World AABB: closed-form projection of the rotated half-extents onto each world
// axis (faster than enumerating 8 corners).
inline AABB shape_world_aabb(const Shape &s, const BodyPose &pose)
{
    switch (s.type)
    {
    case ShapeType::Box:
    {
        // Extent along world axis k: |R[0][k]|*hx + |R[1][k]|*hy + |R[2][k]|*hz.
        const glm::mat3 &R = pose.orientation;
        glm::vec3 e(
            std::abs(R[0][0]) * s.halfExtents.x + std::abs(R[1][0]) * s.halfExtents.y + std::abs(R[2][0]) * s.halfExtents.z,
            std::abs(R[0][1]) * s.halfExtents.x + std::abs(R[1][1]) * s.halfExtents.y + std::abs(R[2][1]) * s.halfExtents.z,
            std::abs(R[0][2]) * s.halfExtents.x + std::abs(R[1][2]) * s.halfExtents.y + std::abs(R[2][2]) * s.halfExtents.z);
        AABB box;
        box.min = pose.position - e;
        box.max = pose.position + e;
        return box;
    }
    case ShapeType::Sphere:
    {
        // Sphere AABB is orientation-independent.
        glm::vec3 e(s.radius);
        AABB box;
        box.min = pose.position - e;
        box.max = pose.position + e;
        return box;
    }
    case ShapeType::Capsule:
    {
        // Local long axis is Y (|y| <= halfHeight), inflated by radius.
        // AABB = min/max of the two world-space cap centers ± radius.
        const glm::mat3 &R = pose.orientation;
        glm::vec3 axisY(R[1][0], R[1][1], R[1][2]); // Column 2 of R is local Y in world space.
        glm::vec3 p1 = pose.position + axisY * s.halfHeight;
        glm::vec3 p2 = pose.position - axisY * s.halfHeight;
        glm::vec3 e(s.radius);
        AABB box;
        box.min = glm::min(p1, p2) - e;
        box.max = glm::max(p1, p2) + e;
        return box;
    }
    case ShapeType::ConvexHull:
    {
        // Transform all vertices to world space and take min/max; n is usually
        // < 100, so the per-frame broadphase cost is acceptable.
        if (s.hullCache == nullptr || s.hullCache->vertices.empty())
        {
            AABB degen;
            degen.min = pose.position;
            degen.max = pose.position;
            return degen;
        }
        const glm::mat3 &R = pose.orientation;
        glm::vec3 world0 = pose.position + R * s.hullCache->vertices[0];
        glm::vec3 mn = world0;
        glm::vec3 mx = world0;
        for (size_t i = 1; i < s.hullCache->vertices.size(); ++i)
        {
            glm::vec3 w = pose.position + R * s.hullCache->vertices[i];
            mn = glm::min(mn, w);
            mx = glm::max(mx, w);
        }
        AABB box;
        box.min = mn;
        box.max = mx;
        return box;
    }
    case ShapeType::Compound:
    default:
        assert(false && "shape_world_aabb: Compound unused at physics layer");
        AABB degenerate;
        degenerate.min = pose.position;
        degenerate.max = pose.position;
        return degenerate;
    }
}

// Farthest vertex of the shape along dirLocal in local space; the core GJK/EPA
// primitive.
inline glm::vec3 shape_support(const Shape &s, const glm::vec3 &dirLocal)
{
    switch (s.type)
    {
    case ShapeType::Box:
    {
        return glm::vec3(
            (dirLocal.x >= 0.0f ? s.halfExtents.x : -s.halfExtents.x),
            (dirLocal.y >= 0.0f ? s.halfExtents.y : -s.halfExtents.y),
            (dirLocal.z >= 0.0f ? s.halfExtents.z : -s.halfExtents.z));
    }
    case ShapeType::Sphere:
    {
        // Support = r * normalize(dir); a zero dir falls back to +X.
        float len2 = glm::dot(dirLocal, dirLocal);
        if (len2 < 1e-20f)
            return glm::vec3(s.radius, 0.0f, 0.0f);
        return dirLocal * (s.radius / std::sqrt(len2));
    }
    case ShapeType::Capsule:
    {
        // Long axis is local Y: pick the cap center by dir.y's sign, then add the
        // sphere support r * normalize(dir).
        float cy = (dirLocal.y >= 0.0f) ? s.halfHeight : -s.halfHeight;
        glm::vec3 center(0.0f, cy, 0.0f);
        float len2 = glm::dot(dirLocal, dirLocal);
        if (len2 < 1e-20f)
            return center + glm::vec3(s.radius, 0.0f, 0.0f);
        return center + dirLocal * (s.radius / std::sqrt(len2));
    }
    case ShapeType::ConvexHull:
    {
        // Vertex scan for max(dot(v, dirLocal)). Could switch to hill-climbing over
        // an adjacency list (expected O(log n)) if GJK ever becomes a hot spot.
        if (s.hullCache == nullptr || s.hullCache->vertices.empty())
            return glm::vec3(0.0f);
        const auto &verts = s.hullCache->vertices;
        int bestIdx = 0;
        float bestDot = glm::dot(verts[0], dirLocal);
        for (size_t i = 1; i < verts.size(); ++i)
        {
            float d = glm::dot(verts[i], dirLocal);
            if (d > bestDot)
            {
                bestDot = d;
                bestIdx = static_cast<int>(i);
            }
        }
        return verts[bestIdx];
    }
    case ShapeType::Compound:
    default:
        // Compound is dissolved before narrowphase; never supported directly.
        assert(false && "shape_support: Compound unused at physics layer");
        return glm::vec3(0.0f);
    }
}

// ============================================================================
// Local-space volume, used to split mass across colliders by volume:
//   Box: 8·hx·hy·hz; Sphere: 4/3·π·r³; Capsule: π·r²·2h + 4/3·π·r³;
//   ConvexHull: conservative local-AABB volume (true integral deferred).
// ============================================================================
inline float shape_volume(const Shape &s)
{
    switch (s.type)
    {
    case ShapeType::Box:
    {
        return std::max(8.0f * s.halfExtents.x * s.halfExtents.y * s.halfExtents.z, 1e-6f);
    }
    case ShapeType::Sphere:
    {
        return std::max((4.0f / 3.0f) * 3.14159265f * s.radius * s.radius * s.radius, 1e-6f);
    }
    case ShapeType::Capsule:
    {
        float cyl = 3.14159265f * s.radius * s.radius * (2.0f * s.halfHeight);
        float sph = (4.0f / 3.0f) * 3.14159265f * s.radius * s.radius * s.radius;
        return std::max(cyl + sph, 1e-6f);
    }
    case ShapeType::ConvexHull:
    {
        if (s.hullCache == nullptr)
            return 1.0f;
        glm::vec3 ext = s.hullCache->localAABBMax - s.hullCache->localAABBMin;
        return std::max(std::max(ext.x, 1e-3f) * std::max(ext.y, 1e-3f) * std::max(ext.z, 1e-3f),
                        1e-6f);
    }
    case ShapeType::Compound:
    default:
        return 1.0f;
    }
}
