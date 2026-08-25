// ============================================================================
// Frustum.h — header-only frustum culling: extract 6 planes from a
// view-projection matrix (Gribb-Hartmann), conservative AABB/sphere tests.
// Planes are ax+by+cz+d=0 with n pointing *into* the frustum; true = possibly visible.
// ============================================================================
#pragma once

#include <glm/glm.hpp>
#include <array>
#include <limits>

namespace FrustumCulling
{
    struct Plane
    {
        glm::vec3 n{0.0f}; // Unit normal pointing into the frustum
        float     d = 0.0f;
    };

    struct Frustum
    {
        // Order: Left, Right, Bottom, Top, Near, Far
        std::array<Plane, 6> planes;
    };

    // Gribb-Hartmann assumes a row-major viewProjection; GLM is column-major,
    // so m[col][row] reads the rows correctly
    inline Frustum extractFromViewProjection(const glm::mat4 &vp)
    {
        Frustum f;
        // row(i) = (vp[0][i], vp[1][i], vp[2][i], vp[3][i])
        auto row = [&](int i) {
            return glm::vec4(vp[0][i], vp[1][i], vp[2][i], vp[3][i]);
        };
        glm::vec4 r0 = row(0); // x
        glm::vec4 r1 = row(1); // y
        glm::vec4 r2 = row(2); // z
        glm::vec4 r3 = row(3); // w

        // Left:   r3 + r0,   Right: r3 - r0
        // Bottom: r3 + r1,   Top:   r3 - r1
        //
        // Near/Far depend on the clip-space depth range:
        //   - GL ([-1, 1]): Near = r3 + r2, Far = r3 - r2
        //   - Vulkan/D3D ([0, 1]; GLM_FORCE_DEPTH_ZERO_TO_ONE is defined in
        //     CMakeLists.txt): Near = r2, Far = r3 - r2
        glm::vec4 raw[6] = {
            r3 + r0, r3 - r0,
            r3 + r1, r3 - r1,
            r2,      r3 - r2,
        };
        for (int i = 0; i < 6; ++i)
        {
            glm::vec3 n(raw[i].x, raw[i].y, raw[i].z);
            float len = glm::length(n);
            if (len < 1e-8f) len = 1.0f;
            f.planes[i].n = n / len;
            f.planes[i].d = raw[i].w / len;
        }
        return f;
    }

    // Local AABB -> loose world AABB via 8 corners; conservative under rotation / non-uniform scale
    inline void transformLocalAABBToWorld(const glm::mat4 &world,
                                          const glm::vec3 &localCenter,
                                          const glm::vec3 &localExtents,
                                          glm::vec3 &outMin,
                                          glm::vec3 &outMax)
    {
        const glm::vec3 mn = localCenter - localExtents;
        const glm::vec3 mx = localCenter + localExtents;
        const glm::vec3 corners[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
            {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
            {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z},
        };
        outMin = glm::vec3( std::numeric_limits<float>::max());
        outMax = glm::vec3(-std::numeric_limits<float>::max());
        for (int i = 0; i < 8; ++i)
        {
            glm::vec4 w = world * glm::vec4(corners[i], 1.0f);
            glm::vec3 v(w.x, w.y, w.z);
            outMin = glm::min(outMin, v);
            outMax = glm::max(outMax, v);
        }
    }

    // true = "possibly visible" (conservative, may over-report). p-vertex test:
    // for each plane take the corner most toward its positive side; if even that
    // corner is outside (n·p + d < 0), the whole AABB is culled.
    inline bool testAABB(const Frustum &f, const glm::vec3 &mn, const glm::vec3 &mx)
    {
        for (int i = 0; i < 6; ++i)
        {
            const glm::vec3 &n = f.planes[i].n;
            // p-vertex: sign of each n component picks mx or mn
            glm::vec3 p(
                n.x >= 0.0f ? mx.x : mn.x,
                n.y >= 0.0f ? mx.y : mn.y,
                n.z >= 0.0f ? mx.z : mn.z);
            if (glm::dot(n, p) + f.planes[i].d < 0.0f)
                return false; // Entirely outside this plane
        }
        return true;
    }

    // true = "possibly visible". Culled only when fully outside a plane
    // (signed center distance < -radius). Normals point inward, so the signed
    // distance is n·c + d.
    inline bool testSphere(const Frustum &f, const glm::vec3 &center, float radius)
    {
        for (int i = 0; i < 6; ++i)
        {
            float signedDist = glm::dot(f.planes[i].n, center) + f.planes[i].d;
            if (signedDist < -radius)
                return false; // Entirely outside this plane
        }
        return true;
    }
} // namespace FrustumCulling
