#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include "physics/PhysicsWorld.h"
#include "physics/Shape.h"
#include "physics/PhysicsMaterial.h"

// ============================================================================
// RigidBodyComponent — ECS rigid-body component.
// friction/restitution live on collider materials only: the solver reads
// shape.material via combineMaterial, so there is no body-level fallback (the
// old body-level fields were removed as a broken edit path — updateBody never
// synced them to shape materials).
// layer/mask: body-level collision filter; shapes inherit them unless the shape
// sets its own.
// ============================================================================
struct RigidBodyComponent
{
    BodyType bodyType = BodyType::Dynamic;
    float mass = 1.0f;

    glm::vec3 velocity = glm::vec3(0.0f);

    // Collision filter: default layer = bit 0, mask = all bits.
    uint32_t layer = 0u;
    uint32_t mask = 0xFFFFFFFFu;

    int physicsIndex = -1; // Index into PhysicsWorld's body array (runtime-bound)
};

// ============================================================================
// BoxColliderComponent — single-collider convenience entry (kept so existing
// scene assets / SceneSerializer / EditorUI keep working). One Box shape plus a
// local-space center and material. If the entity also has CollidersComponent,
// that one wins; otherwise PhysicsSystem registers this as a one-element list.
// `material` is where friction/restitution live on the single-collider path,
// mirroring ColliderDesc::material for multi-collider.
// ============================================================================
struct BoxColliderComponent
{
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec3 halfExtents = glm::vec3(0.5f);
    PhysicsMaterial material; // Defaults (friction=0.5, restitution=0.3) match PhysicsMaterial.
};

// ============================================================================
// ColliderDesc / CollidersComponent — multi-collider data.
// PhysicsWorld expands the list into shapes_ and maps body → shape range via
// bodyFirstShape_ / bodyShapeCount_.
// World shape pose = body.pose ⊕ (localPosition, localRotation); body.pose comes
// from the ECS Transform on first register, then the solver owns it.
// ============================================================================
struct ColliderDesc
{
    Shape shape;
    glm::vec3 localPosition = glm::vec3(0.0f);
    glm::quat localRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    PhysicsMaterial material; // Default material.
    uint32_t layer = 0u;      // Shape layer; 0 means inherit from the body.
    uint32_t mask = 0u;       // Shape mask; 0 means inherit from the body.
    bool isTrigger = false;   // Trigger: detected but not solved.

    // ConvexHull vertices supplied by the ECS layer (serialization/Inspector/
    // scripts); shape.hullIndex/hullCache are invalid until registered. In
    // prepareShapeForWorld, non-empty hullLocalVertices triggers a
    // PhysicsWorld::registerConvexHull call that backfills shape.hullIndex —
    // PhysicsWorld stays the single owner of hull storage.
    std::vector<glm::vec3> hullLocalVertices;

    int physicsShapeIndex = -1; // Runtime index into PhysicsWorld.shapes_.
};

struct CollidersComponent
{
    std::vector<ColliderDesc> list;
};
