#pragma once

#include <entt/entt.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include "ecs/Components.h"
#include "physics/PhysicsComponents.h"
#include "physics/PhysicsWorld.h"

// ECS physics system (translation + rotation).
//
// Per-frame flow: sync ECS -> PhysicsWorld (Transform + BoxCollider to world OBB),
// stepSimulation(dt), then sync PhysicsWorld -> ECS (OBB center/orientation back).
namespace PhysicsSystem
{

    inline bool simulationRunning = false;

    // Pause lives in the physics layer as a dimension independent of "running":
    // running=true, paused=true -> update() skips stepping, but UI / Gizmo /
    // rendering keep refreshing so the current frame can be inspected and edited.
    inline bool simulationPaused = false;

    inline void setPaused(bool v) { simulationPaused = v; }

    // Builds a world-space OBB from Transform + BoxCollider.
    //
    //   worldScale   = extractWorldScale(tc.worldMatrix)   // includes parent scale
    //   obb.center   = worldMatrix * bc.center             // world-space center
    //   obb.halfExt  = bc.halfExtents * |worldScale|       // halfExtents stay non-negative
    //   obb.orient   = normalizeRotationMatrix(worldMatrix[0..2])
    //
    // Uses worldScale instead of tc.scale so parent scale propagates correctly.
    inline OBB buildWorldOBB(const TransformComponent &tc, const BoxColliderComponent &bc)
    {
        OBB obb;
        obb.center = glm::vec3(tc.worldMatrix * glm::vec4(bc.center, 1.0f));

        glm::vec3 worldScale = TransformMath::extractWorldScale(tc.worldMatrix);
        // halfExtents are non-negative sizes; mirrors (negative scale) live in the orientation's det flip
        obb.halfExtents = bc.halfExtents * glm::abs(worldScale);

        obb.orientation = TransformMath::normalizeRotationMatrix(glm::mat3(tc.worldMatrix));
        return obb;
    }

    // Extracts XYZ Euler angles, matching getLocalMatrix's Rx*Ry*Rz order.
    inline glm::vec3 extractEulerXYZ(const glm::mat3 &R)
    {
        float pitch, yaw, roll; // x, y, z
        glm::mat4 m4(R);
        glm::extractEulerAngleXYZ(m4, pitch, yaw, roll);
        return glm::degrees(glm::vec3(pitch, yaw, roll));
    }

    // Scales a ColliderDesc shape by the Transform world scale and registers
    // ConvexHull shapes with PhysicsWorld.hulls_; returns a Shape ready for attachShape.
    //
    // Scaling convention (generic per shape type, no scene special cases):
    //   - Box       : halfExtents *= |worldScale| (per-axis)
    //   - Sphere    : radius      *= max(|worldScale|) (scalar radius; conservative max)
    //   - Capsule   : radius, halfHeight scaled the same way
    //   - ConvexHull: localVertices *= |worldScale|, then re-registered; the shared
    //                 hullCache is never mutated in place (other entities may reference it)
    //   - Compound  : not consumed here (ECS expresses it as one body, many ColliderDescs)
    //
    // Does not modify cd; returns a value Shape for this attachShape call.
    inline Shape prepareShapeForWorld(const ColliderDesc &cd,
                                      const glm::vec3 &worldScale,
                                      PhysicsWorld &world)
    {
        Shape sh = cd.shape;
        const glm::vec3 absScale = glm::abs(worldScale);
        const float maxS = std::max(absScale.x, std::max(absScale.y, absScale.z));
        switch (sh.type)
        {
        case ShapeType::Box:
            sh.halfExtents = cd.shape.halfExtents * absScale;
            break;
        case ShapeType::Sphere:
            sh.radius = cd.shape.radius * maxS;
            break;
        case ShapeType::Capsule:
            sh.radius = cd.shape.radius * maxS;
            sh.halfHeight = cd.shape.halfHeight * maxS;
            break;
        case ShapeType::ConvexHull:
        {
            // Prefer ECS-provided hullLocalVertices (the SceneSerializer / Inspector path).
            if (!cd.hullLocalVertices.empty())
            {
                std::vector<glm::vec3> scaled;
                scaled.reserve(cd.hullLocalVertices.size());
                const bool needScale = (std::abs(maxS - 1.0f) > 1e-5f) ||
                                       (absScale.x != absScale.y || absScale.y != absScale.z);
                if (needScale)
                {
                    for (const auto &v : cd.hullLocalVertices)
                        scaled.push_back(v * absScale);
                }
                else
                {
                    scaled = cd.hullLocalVertices;
                }
                sh.hullIndex = world.registerConvexHull(scaled);
                sh.hullCache = nullptr; // attachShape re-fills hullCache from hullIndex
                break;
            }

            // Without hullLocalVertices: if a hullCache pointer was provided (script
            // reuse) and scale is 1, keep hullIndex so attachShape re-fills it directly;
            // otherwise copy, scale, and re-register.
            const ConvexMeshCache *srcCache = cd.shape.hullCache;
            const bool needScale = (std::abs(maxS - 1.0f) > 1e-5f) ||
                                   (absScale.x != absScale.y || absScale.y != absScale.z);
            if (srcCache && !srcCache->vertices.empty() && needScale)
            {
                std::vector<glm::vec3> scaled;
                scaled.reserve(srcCache->vertices.size());
                for (const auto &v : srcCache->vertices)
                    scaled.push_back(v * absScale);
                sh.hullIndex = world.registerConvexHull(scaled);
                sh.hullCache = nullptr; // attachShape re-fills hullCache from hullIndex
            }
            // else: keep cd.shape.hullIndex; attachShape re-fills the cache from it
            break;
        }
        case ShapeType::Compound:
        default:
            // Compound not consumed directly; ECS expresses it via multiple ColliderDescs.
            break;
        }
        return sh;
    }

    inline void init(entt::registry &reg, PhysicsWorld &world)
    {
        world.clear();

        // Register every entity with RigidBodyComponent + Transform, multi-collider
        // first: a non-empty CollidersComponent uses addEmptyBody + attachShape per
        // collider; otherwise the BoxColliderComponent path uses addBody.
        auto rbView = reg.view<RigidBodyComponent, TransformComponent>();
        for (auto e : rbView)
        {
            auto &rb = rbView.get<RigidBodyComponent>(e);
            auto &tc = rbView.get<TransformComponent>(e);

            RigidBody body;
            body.bodyType = rb.bodyType;
            body.mass = rb.mass;
            body.velocity = rb.velocity;

            bool hasMulti = reg.all_of<CollidersComponent>(e) &&
                            !reg.get<CollidersComponent>(e).list.empty();

            if (hasMulti)
            {
                // Multi-collider path: body origin is the Transform origin (no BoxColliderComponent.center offset).
                BodyPose pose;
                pose.position = glm::vec3(tc.worldMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
                pose.orientation = TransformMath::normalizeRotationMatrix(glm::mat3(tc.worldMatrix));
                int bodyIdx = world.addEmptyBody(body, pose);
                rb.physicsIndex = bodyIdx;
                world.setBodyLayerMask(bodyIdx, rb.layer, rb.mask);

                glm::vec3 worldScale = TransformMath::extractWorldScale(tc.worldMatrix);
                auto &cc = reg.get<CollidersComponent>(e);
                for (auto &cd : cc.list)
                {
                    Shape sh = prepareShapeForWorld(cd, worldScale, world);
                    BodyPose local;
                    local.position = cd.localPosition;
                    local.orientation = glm::mat3_cast(cd.localRotation);
                    uint32_t layer = (cd.layer == 0u ? rb.layer : cd.layer);
                    uint32_t mask = (cd.mask == 0u ? rb.mask : cd.mask);
                    cd.physicsShapeIndex = world.attachShape(bodyIdx, sh, local, cd.material,
                                                             layer, mask, cd.isTrigger);
                }
            }
            else if (reg.all_of<BoxColliderComponent>(e))
            {
                auto &bc = reg.get<BoxColliderComponent>(e);
                OBB obb = buildWorldOBB(tc, bc);
                // CP-3.2c: pass bc.material into addBody so it lands in the auto-created Box shape's shapeMaterial_.
                int bodyIdx = world.addBody(body, obb, bc.material);
                rb.physicsIndex = bodyIdx;
                world.setBodyLayerMask(bodyIdx, rb.layer, rb.mask);
            }
        }
    }

    inline void update(entt::registry &reg, PhysicsWorld &world, float deltaTime)
    {
        // Early-out before the ECS->PhysicsWorld sync so the physics world stays
        // fully frozen while paused; Transform / Gizmo / rendering keep refreshing
        // normally from the outer layer.
        if (!simulationRunning || simulationPaused)
            return;

        // 1. Sync ECS -> PhysicsWorld.
        // Iterate all entities with RigidBodyComponent (not just BoxCollider) so
        // entities carrying only a CollidersComponent are supported.
        auto view = reg.view<RigidBodyComponent, TransformComponent>();
        for (auto e : view)
        {
            auto &rb = view.get<RigidBodyComponent>(e);
            auto &tc = view.get<TransformComponent>(e);

            if (rb.physicsIndex < 0)
                continue;

            bool hasMulti = reg.all_of<CollidersComponent>(e) &&
                            !reg.get<CollidersComponent>(e).list.empty();

            if (rb.bodyType == BodyType::Static || rb.bodyType == BodyType::Kinematic)
            {
                // Static / Kinematic: body pose follows Transform exactly.
                //   Static    - pose is basically fixed (edited by user drag); setBodyPose doesn't touch prevPoses_
                //   Kinematic - pose changes per frame; stepSimulation derives linear/angular
                //               velocity from the prevPoses_ vs poses_ diff for the solver
                if (hasMulti)
                {
                    // Multi-collider: sync only the body's world pose; shape sizes were
                    // written at registration and runtime edits go through the Inspector.
                    BodyPose pose;
                    pose.position = glm::vec3(tc.worldMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
                    pose.orientation = TransformMath::normalizeRotationMatrix(glm::mat3(tc.worldMatrix));
                    world.setBodyPose(rb.physicsIndex, pose);
                }
                else if (reg.all_of<BoxColliderComponent>(e))
                {
                    auto &bc = reg.get<BoxColliderComponent>(e);
                    OBB obb = buildWorldOBB(tc, bc);
                    world.updateOBB(rb.physicsIndex, obb);
                }
            }
            else
            {
                // Dynamic: center/orientation are solver-managed and never overwritten.
                // Shape sizes can still be edited at runtime in the Inspector, so the
                // local halfExtents plus parent scale are synced to physics.
                if (!hasMulti && reg.all_of<BoxColliderComponent>(e))
                {
                    auto &bc = reg.get<BoxColliderComponent>(e);
                    glm::vec3 worldScale = TransformMath::extractWorldScale(tc.worldMatrix);
                    world.updateShapeExtents(rb.physicsIndex, bc.halfExtents * glm::abs(worldScale));
                }
                // Runtime size edits for multi-collider bodies are unsupported (needs Inspector work).
            }

            // Sync rigid body attributes (velocity and angularVelocity are left alone).
            // CP-3.2: friction/restitution moved into shape.material and are pushed via
            // the collider path; no longer synced through RigidBody.
            RigidBody body = world.getBody(rb.physicsIndex);
            body.bodyType = rb.bodyType;
            body.mass = rb.mass;
            world.updateBody(rb.physicsIndex, body);

            // Layer/mask may be edited at runtime in the editor.
            world.setBodyLayerMask(rb.physicsIndex, rb.layer, rb.mask);
        }

        // 2. Step physics.
        world.stepSimulation(deltaTime);

        // 3. Sync PhysicsWorld -> ECS (position + rotation).
        for (auto e : view)
        {
            auto &rb = view.get<RigidBodyComponent>(e);
            auto &tc = view.get<TransformComponent>(e);

            if (rb.physicsIndex < 0)
                continue;
            // Static / Kinematic pose authority is ECS; never write back. Their pose
            // comes from the ECS->PhysicsWorld sync above; physics only consumes it.
            if (rb.bodyType == BodyType::Static || rb.bodyType == BodyType::Kinematic)
                continue;

            const auto &body = world.getBody(rb.physicsIndex);

            // --- Write back position ---
            glm::vec3 newPos;
            glm::mat3 newOrient;

            bool hasMulti = reg.all_of<CollidersComponent>(e) &&
                            !reg.get<CollidersComponent>(e).list.empty();

            if (hasMulti)
            {
                // Multi-collider: body.position is the Transform's world origin directly.
                newPos = world.getPosition(rb.physicsIndex);
                newOrient = world.getOBB(rb.physicsIndex).orientation;
            }
            else
            {
                auto &bc = reg.get<BoxColliderComponent>(e);
                const auto &obb = world.getOBB(rb.physicsIndex);
                // Forward: obb.center = worldMatrix * bc.center = origin + orient * (worldScale * bc.center)
                // Reverse: origin = obb.center - orient * (worldScale * bc.center)
                glm::vec3 worldScale = TransformMath::extractWorldScale(tc.worldMatrix);
                glm::vec3 worldOffset = obb.orientation * (worldScale * bc.center);
                newPos = obb.center - worldOffset;
                newOrient = obb.orientation;
            }

            // Account for parent hierarchy.
            if (reg.all_of<HierarchyComponent>(e))
            {
                auto &hc = reg.get<HierarchyComponent>(e);
                if (hc.parent != entt::null && reg.valid(hc.parent) && reg.all_of<TransformComponent>(hc.parent))
                {
                    auto &parentTC = reg.get<TransformComponent>(hc.parent);
                    glm::mat4 invParent = glm::inverse(parentTC.worldMatrix);

                    glm::vec4 localPos = invParent * glm::vec4(newPos, 1.0f);
                    tc.position = glm::vec3(localPos);

                    glm::mat3 parentRot = glm::mat3(parentTC.worldMatrix);
                    parentRot = TransformMath::normalizeRotationMatrix(parentRot);
                    glm::mat3 localOri = glm::transpose(parentRot) * newOrient;
                    tc.setRotationMatrix(localOri);
                }
                else
                {
                    tc.position = newPos;
                    tc.setRotationMatrix(newOrient);
                }
            }
            else
            {
                tc.position = newPos;
                tc.setRotationMatrix(newOrient);
            }

            // Sync velocity back to ECS.
            rb.velocity = body.velocity;
        }
    }

    inline void registerEntity(entt::registry &reg, entt::entity e, PhysicsWorld &world)
    {
        if (!reg.all_of<RigidBodyComponent, TransformComponent>(e))
            return;
        auto &rb = reg.get<RigidBodyComponent>(e);
        auto &tc = reg.get<TransformComponent>(e);

        RigidBody body;
        body.bodyType = rb.bodyType;
        body.mass = rb.mass;
        body.velocity = rb.velocity;

        bool hasMulti = reg.all_of<CollidersComponent>(e) &&
                        !reg.get<CollidersComponent>(e).list.empty();

        if (hasMulti)
        {
            BodyPose pose;
            pose.position = glm::vec3(tc.worldMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            pose.orientation = TransformMath::normalizeRotationMatrix(glm::mat3(tc.worldMatrix));
            int bodyIdx = world.addEmptyBody(body, pose);
            rb.physicsIndex = bodyIdx;
            world.setBodyLayerMask(bodyIdx, rb.layer, rb.mask);

            glm::vec3 worldScale = TransformMath::extractWorldScale(tc.worldMatrix);
            auto &cc = reg.get<CollidersComponent>(e);
            for (auto &cd : cc.list)
            {
                Shape sh = prepareShapeForWorld(cd, worldScale, world);
                BodyPose local;
                local.position = cd.localPosition;
                local.orientation = glm::mat3_cast(cd.localRotation);
                uint32_t layer = (cd.layer == 0u ? rb.layer : cd.layer);
                uint32_t mask = (cd.mask == 0u ? rb.mask : cd.mask);
                cd.physicsShapeIndex = world.attachShape(bodyIdx, sh, local, cd.material,
                                                         layer, mask, cd.isTrigger);
            }
        }
        else if (reg.all_of<BoxColliderComponent>(e))
        {
            auto &bc = reg.get<BoxColliderComponent>(e);
            OBB obb = buildWorldOBB(tc, bc);
            // CP-3.2c: mirror the init path; registerEntity also passes bc.material.
            int bodyIdx = world.addBody(body, obb, bc.material);
            rb.physicsIndex = bodyIdx;
            world.setBodyLayerMask(bodyIdx, rb.layer, rb.mask);
        }
    }

    inline void unregisterEntity(entt::registry &reg, entt::entity e, PhysicsWorld &world)
    {
        if (!reg.all_of<RigidBodyComponent>(e))
            return;
        auto &rb = reg.get<RigidBodyComponent>(e);
        if (rb.physicsIndex >= 0)
        {
            world.removeBody(rb.physicsIndex);
            rb.physicsIndex = -1;
        }
    }

    // syncTransformToPhysics: writes an edited Transform back to the physics body
    // after a Play-mode Gizmo drag. Without this, the next substep would overwrite
    // the new Transform with the stale pose.
    //
    // Behavior (mirrors Unity runtime drag semantics):
    //   - body unregistered (physicsIndex < 0) or no RigidBodyComponent: return (no side effects)
    //   - Dynamic: write pose + zero linear/angular velocity + wake the body
    //   - Static / Kinematic: write pose only; no velocity (logically zero), no wake needed
    //
    // Branches only on "body registered or not" - no per-entity/per-scene special
    // cases. Colliders are not rebuilt: Gizmo drags only change TRS, and the
    // update() ECS->PhysicsWorld sync refreshes shapes next frame; this only
    // resolves the pose-overwrite race.
    inline void syncTransformToPhysics(entt::registry &reg, entt::entity e, PhysicsWorld &world)
    {
        if (!reg.valid(e))
            return;
        if (!reg.all_of<RigidBodyComponent, TransformComponent>(e))
            return;

        auto &rb = reg.get<RigidBodyComponent>(e);
        if (rb.physicsIndex < 0)
            return;

        auto &tc = reg.get<TransformComponent>(e);

        // body.position is the Transform's world origin in the multi-collider path;
        // the single-collider (BoxCollider) write-back also derives origin as
        // obb.center - offset, so writing the world origin is correct for both.
        BodyPose pose;
        pose.position = glm::vec3(tc.worldMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        pose.orientation = TransformMath::normalizeRotationMatrix(glm::mat3(tc.worldMatrix));
        world.setBodyPose(rb.physicsIndex, pose);

        if (rb.bodyType == BodyType::Dynamic)
        {
            world.setLinearVelocity(rb.physicsIndex, glm::vec3(0.0f));
            world.setAngularVelocity(rb.physicsIndex, glm::vec3(0.0f));
            // Mirror on the ECS side.
            rb.velocity = glm::vec3(0.0f);
        }

        // setBodyPose / setLinearVelocity already wake internally; the explicit wake
        // states the intent: user interaction -> body stays active.
        world.wakeBody(rb.physicsIndex);
    }

    // Play/Stop/load-scene lifecycle helpers: clear the world, take/restore
    // snapshots, and sync simulationRunning, so the logic doesn't scatter across UI
    // code. Snapshot is a template parameter so physics doesn't depend on
    // renderer/ecs structs; it must provide entity, position, orientation, scale.

    // Snapshots all simulatable entities' Transforms. Snapshot contents are written
    // only here, never inheriting leftovers from the previous run.
    template <typename SnapshotVec>
    inline void captureTransformSnapshots(entt::registry &reg, SnapshotVec &snapshots)
    {
        snapshots.clear();
        auto view = reg.view<RigidBodyComponent, TransformComponent>();
        snapshots.reserve(view.size_hint());
        for (auto e : view)
        {
            auto &tc = view.get<TransformComponent>(e);
            using Snap = typename SnapshotVec::value_type;
            snapshots.push_back(Snap{e, tc.position, tc.rotation, tc.orientation, tc.scale});
        }
    }

    // Editor -> Play: clear physicsIndex, snapshot Transforms, init PhysicsWorld
    // (which clears and rebuilds), then set running. Every path into Play (button,
    // hotkey, automated test) must go through this.
    template <typename SnapshotVec>
    inline void enterPlay(entt::registry &reg, PhysicsWorld &world, SnapshotVec &snapshots)
    {
        // Clear all physicsIndex values first so stale indices aren't used before init.
        auto rbView = reg.view<RigidBodyComponent>();
        for (auto e : rbView)
            rbView.get<RigidBodyComponent>(e).physicsIndex = -1;

        captureTransformSnapshots(reg, snapshots);
        init(reg, world); // init() clears the world and adds bodies; no extra clear needed
        simulationRunning = true;
        simulationPaused = false; // A new Play run always starts unpaused.
    }

    // Play -> Editor: clear running, optionally restore snapshots, clear world and snapshots.
    // restoreSnapshots = true: Stop button (restore pre-Play positions)
    // restoreSnapshots = false: Load Scene (no old snapshots needed)
    //
    // Writes physicsIndex = -1 for ALL RigidBodyComponents, not just snapshotted
    // ones, so bodies added during Play don't keep dangling indices. Whether
    // Play-added bodies should roll back is a separate decision.
    template <typename SnapshotVec>
    inline void exitPlay(entt::registry &reg, PhysicsWorld &world, SnapshotVec &snapshots,
                         bool restoreSnapshots)
    {
        simulationRunning = false;
        simulationPaused = false; // Leaving Play must clear the paused state.

        if (restoreSnapshots)
        {
            for (auto &snap : snapshots)
            {
                if (reg.valid(snap.entity) && reg.all_of<TransformComponent>(snap.entity))
                {
                    auto &tc = reg.get<TransformComponent>(snap.entity);
                    tc.position = snap.position;
                    tc.setOrientation(snap.orientation);
                    tc.scale = snap.scale;
                }
                if (reg.valid(snap.entity) && reg.all_of<RigidBodyComponent>(snap.entity))
                {
                    auto &rb = reg.get<RigidBodyComponent>(snap.entity);
                    rb.velocity = glm::vec3(0.0f);
                }
            }
        }

        // Reset physicsIndex for all RigidBodyComponents, covering bodies added during Play.
        auto rbView = reg.view<RigidBodyComponent>();
        for (auto e : rbView)
            rbView.get<RigidBodyComponent>(e).physicsIndex = -1;

        snapshots.clear();
        world.clear();
    }

} // namespace PhysicsSystem
