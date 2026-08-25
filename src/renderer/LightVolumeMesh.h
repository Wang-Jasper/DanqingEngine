#pragma once

// LightVolumeMesh — low-poly icosphere proxy for point/spot lights; shader maps
// worldPos = lightPos + vertex * lightRadius. Vertices expanded x1.02 so the
// mesh never clips the true sphere edge; positions only (fragment shader reads G-Buffer).

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace LightVolumeMesh
{
    struct MeshData
    {
        std::vector<glm::vec3> positions;
        std::vector<uint32_t> indices;
    };

    // Generate an icosphere.
    // subdivisions: 0 = 20 triangles, 1 = 80, 2 = 320
    // scale: vertex expansion (>1 avoids edge clipping; default 1.02)
    inline MeshData generateIcosphere(int subdivisions = 1, float scale = 1.02f)
    {
        MeshData m;

        // Step 1: initial 12-vertex icosahedron (golden ratio phi)
        const float t = (1.0f + std::sqrt(5.0f)) * 0.5f; // phi
        const float invLen = 1.0f / std::sqrt(1.0f + t * t);

        // 12 vertices, normalized to the unit sphere
        std::vector<glm::vec3> verts = {
            glm::vec3(-1, t, 0) * invLen,
            glm::vec3(1, t, 0) * invLen,
            glm::vec3(-1, -t, 0) * invLen,
            glm::vec3(1, -t, 0) * invLen,
            glm::vec3(0, -1, t) * invLen,
            glm::vec3(0, 1, t) * invLen,
            glm::vec3(0, -1, -t) * invLen,
            glm::vec3(0, 1, -t) * invLen,
            glm::vec3(t, 0, -1) * invLen,
            glm::vec3(t, 0, 1) * invLen,
            glm::vec3(-t, 0, -1) * invLen,
            glm::vec3(-t, 0, 1) * invLen,
        };

        // 20 initial triangle indices
        std::vector<uint32_t> tris = {
            0,
            11,
            5,
            0,
            5,
            1,
            0,
            1,
            7,
            0,
            7,
            10,
            0,
            10,
            11,
            1,
            5,
            9,
            5,
            11,
            4,
            11,
            10,
            2,
            10,
            7,
            6,
            7,
            1,
            8,
            3,
            9,
            4,
            3,
            4,
            2,
            3,
            2,
            6,
            3,
            6,
            8,
            3,
            8,
            9,
            4,
            9,
            5,
            2,
            4,
            11,
            6,
            2,
            10,
            8,
            6,
            7,
            9,
            8,
            1,
        };

        // Step 2: subdivide (each pass splits every triangle into 4)
        auto key = [](uint32_t a, uint32_t b) -> uint64_t
        {
            if (a > b)
                std::swap(a, b);
            return (static_cast<uint64_t>(a) << 32) | b;
        };

        for (int s = 0; s < subdivisions; ++s)
        {
            std::vector<uint32_t> newTris;
            newTris.reserve(tris.size() * 4);
            std::unordered_map<uint64_t, uint32_t> midCache;

            auto getMid = [&](uint32_t a, uint32_t b) -> uint32_t
            {
                uint64_t k = key(a, b);
                auto it = midCache.find(k);
                if (it != midCache.end())
                    return it->second;
                glm::vec3 mid = glm::normalize((verts[a] + verts[b]) * 0.5f);
                uint32_t idx = static_cast<uint32_t>(verts.size());
                verts.push_back(mid);
                midCache[k] = idx;
                return idx;
            };

            for (size_t i = 0; i < tris.size(); i += 3)
            {
                uint32_t a = tris[i + 0];
                uint32_t b = tris[i + 1];
                uint32_t c = tris[i + 2];
                uint32_t ab = getMid(a, b);
                uint32_t bc = getMid(b, c);
                uint32_t ca = getMid(c, a);
                newTris.insert(newTris.end(), {a, ab, ca,
                                               b, bc, ab,
                                               c, ca, bc,
                                               ab, bc, ca});
            }
            tris.swap(newTris);
        }

        // Step 3: expand vertices
        m.positions.reserve(verts.size());
        for (auto &v : verts)
            m.positions.push_back(v * scale);
        m.indices = std::move(tris);
        return m;
    }
}
