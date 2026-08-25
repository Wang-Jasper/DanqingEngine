// ============================================================================
// test_shape_aabb.cpp — numeric regression for shape_world_aabb(Box): translation,
// 45° Y rotation, per-axis rotations, asymmetric extents
// ============================================================================
#include "test_framework.h"
#include "physics/Shape.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace
{
    Shape makeBox(const glm::vec3 &half)
    {
        Shape s;
        s.type = ShapeType::Box;
        s.halfExtents = half;
        return s;
    }

    BodyPose poseAt(const glm::vec3 &pos, const glm::mat3 &R = glm::mat3(1.0f))
    {
        BodyPose p;
        p.position = pos;
        p.orientation = R;
        return p;
    }
} // namespace

PHYS_TEST(ShapeAABB, UnitBoxAtOrigin)
{
    auto s = makeBox(glm::vec3(0.5f));
    auto pose = poseAt(glm::vec3(0.0f));
    AABB aabb = shape_world_aabb(s, pose);
    PHYS_CHECK_VEC3_NEAR(aabb.min, glm::vec3(-0.5f), 1e-6f);
    PHYS_CHECK_VEC3_NEAR(aabb.max, glm::vec3(0.5f), 1e-6f);
}

PHYS_TEST(ShapeAABB, TranslatedBox)
{
    auto s = makeBox(glm::vec3(0.5f, 1.0f, 2.0f));
    auto pose = poseAt(glm::vec3(10.0f, -5.0f, 3.0f));
    AABB aabb = shape_world_aabb(s, pose);
    PHYS_CHECK_VEC3_NEAR(aabb.min, glm::vec3(9.5f, -6.0f, 1.0f), 1e-6f);
    PHYS_CHECK_VEC3_NEAR(aabb.max, glm::vec3(10.5f, -4.0f, 5.0f), 1e-6f);
}

PHYS_TEST(ShapeAABB, RotateY45)
{
    // 1×1×1 box rotated 45° about Y → X/Z extent grows to sqrt(2)/2 ≈ 0.7071
    auto s = makeBox(glm::vec3(0.5f));
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                        glm::radians(45.0f),
                                        glm::vec3(0, 1, 0)));
    auto pose = poseAt(glm::vec3(0.0f), R);
    AABB aabb = shape_world_aabb(s, pose);
    const float k = 0.70710678f;
    PHYS_CHECK_NEAR(aabb.min.x, -k, 1e-5f);
    PHYS_CHECK_NEAR(aabb.max.x, k, 1e-5f);
    PHYS_CHECK_NEAR(aabb.min.z, -k, 1e-5f);
    PHYS_CHECK_NEAR(aabb.max.z, k, 1e-5f);
    // Y axis unaffected by the rotation
    PHYS_CHECK_NEAR(aabb.min.y, -0.5f, 1e-6f);
    PHYS_CHECK_NEAR(aabb.max.y, 0.5f, 1e-6f);
}

PHYS_TEST(ShapeAABB, RotateX90NonCube)
{
    // half=(1,2,3) rotated 90° about X → y<->z swap
    auto s = makeBox(glm::vec3(1.0f, 2.0f, 3.0f));
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                        glm::radians(90.0f),
                                        glm::vec3(1, 0, 0)));
    auto pose = poseAt(glm::vec3(0.0f), R);
    AABB aabb = shape_world_aabb(s, pose);
    // X unchanged; Y extends to 3; Z extends to 2
    PHYS_CHECK_NEAR(aabb.max.x, 1.0f, 1e-5f);
    PHYS_CHECK_NEAR(aabb.max.y, 3.0f, 1e-5f);
    PHYS_CHECK_NEAR(aabb.max.z, 2.0f, 1e-5f);
}

PHYS_TEST(ShapeAABB, ArbitraryRotation)
{
    // 60° about (1,1,1)/sqrt(3): closed-form symmetry, extents should match across axes
    auto s = makeBox(glm::vec3(0.5f));
    glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(1.0f),
                                        glm::radians(60.0f),
                                        glm::normalize(glm::vec3(1, 1, 1))));
    auto pose = poseAt(glm::vec3(0.0f), R);
    AABB aabb = shape_world_aabb(s, pose);
    // Symmetry: min and max magnitudes must be equal
    PHYS_CHECK_NEAR(aabb.min.x, -aabb.max.x, 1e-5f);
    PHYS_CHECK_NEAR(aabb.min.y, -aabb.max.y, 1e-5f);
    PHYS_CHECK_NEAR(aabb.min.z, -aabb.max.z, 1e-5f);
    // Equal extent on all three axes (cube + symmetric rotation axis)
    PHYS_CHECK_NEAR(aabb.max.x, aabb.max.y, 1e-5f);
    PHYS_CHECK_NEAR(aabb.max.y, aabb.max.z, 1e-5f);
    // Extent must exceed 0.5 (rotation necessarily grows the AABB) and stay <= sqrt(3)/2 ≈ 0.866
    PHYS_CHECK(aabb.max.x > 0.5f, "rotated AABB must grow");
    PHYS_CHECK(aabb.max.x <= 0.866f + 1e-5f, "rotated AABB bounded by diagonal");
}
