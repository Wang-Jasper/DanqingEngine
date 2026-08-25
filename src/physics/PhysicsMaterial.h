#pragma once

#include <algorithm>
#include <cstdint>

// ============================================================================
// PhysicsMaterial — physics material resource. One shape binds one material;
// when two shapes touch, combineMaterial blends the pair per the Combine policy.
// Combine semantics (industry-common):
//   Average  — neutral friction
//   Minimum  — either side smooth makes the pair smooth (e.g. ice)
//   Maximum  — either side bouncy makes the pair bouncy (e.g. a ball)
//   Multiply — mutual damping, common for cloth/rubber
// Struct defaults: friction=Average, restitution=Minimum (matches the old
// `std::min(restitution)` behavior).
// ============================================================================

struct PhysicsMaterial
{
    enum class Combine : uint8_t
    {
        Average = 0,
        Minimum = 1,
        Maximum = 2,
        Multiply = 3,
    };

    float friction = 0.5f;
    float restitution = 0.3f;
    // Below this relative normal speed, restitution is treated as 0 (kills
    // low-speed bounce noise). Default 0.3 m/s matches the old hardcoded
    // bounceVelThreshold so there is no regression; raise it if heavier contacts
    // still bounce audibly.
    float restitutionThreshold = 0.3f;
    float rollingFriction = 0.0f;

    Combine frictionCombine = Combine::Average;
    // Default Minimum keeps the old `std::min(restitution)` behavior; Unity/Unreal
    // default to Maximum, so switch explicitly if "bouncy side dominates" is wanted.
    Combine restitutionCombine = Combine::Minimum;
};

// ============================================================================
// Combine two materials into a per-contact CombinedMaterial.
// A's combine mode wins when A and B disagree (Unity PhysX reads A's mode,
// Unreal uses max priority). A-priority is deterministic and only mildly breaks
// commutativity on rare mixed-material pairs.
// ============================================================================

inline float applyCombine(PhysicsMaterial::Combine mode, float a, float b)
{
    switch (mode)
    {
    case PhysicsMaterial::Combine::Average:
        return (a + b) * 0.5f;
    case PhysicsMaterial::Combine::Minimum:
        return std::min(a, b);
    case PhysicsMaterial::Combine::Maximum:
        return std::max(a, b);
    case PhysicsMaterial::Combine::Multiply:
        return a * b;
    }
    return (a + b) * 0.5f;
}

struct CombinedMaterial
{
    float friction = 0.5f;
    float restitution = 0.3f;
    float restitutionThreshold = 1.0f;
    float rollingFriction = 0.0f;
};

inline CombinedMaterial combineMaterial(const PhysicsMaterial &a, const PhysicsMaterial &b)
{
    CombinedMaterial c;
    c.friction = applyCombine(a.frictionCombine, a.friction, b.friction);
    c.restitution = applyCombine(a.restitutionCombine, a.restitution, b.restitution);
    // Threshold and rolling friction take max: if either side demands a higher
    // threshold, it is honored.
    c.restitutionThreshold = std::max(a.restitutionThreshold, b.restitutionThreshold);
    c.rollingFriction = std::max(a.rollingFriction, b.rollingFriction);
    return c;
}
