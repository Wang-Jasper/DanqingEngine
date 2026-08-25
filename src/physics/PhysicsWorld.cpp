#include "physics/PhysicsWorld.h"
#include "physics/narrowphase_gjk.h"
#include "physics/geometry2d.h" // sleep static-stability criterion
#include "physics/ccd_toi.h"    // Conservative Advancement TOI
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <fstream>
#include <iomanip>
#include <limits>

// Global log file; opened and written for the first 500 frames only when
// debugLogEnabled=true.
static std::ofstream g_physLog;
static int g_physFrame = 0;
static const int MAX_LOG_FRAMES = 500;

static void openPhysLogIfEnabled(bool enabled)
{
    if (!enabled)
        return;
    if (!g_physLog.is_open())
    {
        g_physLog.open("physics_debug.log", std::ios::trunc);
        g_physLog << std::fixed << std::setprecision(4);
    }
}

int PhysicsWorld::addEmptyBody(const RigidBody &body, const BodyPose &pose)
{
    int bodyIdx = -1;
    for (int i = 0; i < static_cast<int>(active_.size()); ++i)
    {
        if (!active_[i])
        {
            bodyIdx = i;
            bodies_[i] = body;
            poses_[i] = pose;
            prevPoses_[i] = pose; // prevPose = initial pose, so first-frame delta velocity is 0
            active_[i] = true;
            bodyLayer_[i] = 0u;
            bodyMask_[i] = 0xFFFFFFFFu;
            bodyShapes_[i].clear();
            broadphaseProxy_[i] = DynamicAABBTree::NullNode; // proxy inserted by refreshBroadphase on first use
            break;
        }
    }
    if (bodyIdx < 0)
    {
        bodies_.push_back(body);
        poses_.push_back(pose);
        prevPoses_.push_back(pose); // keep in sync with poses_; first-frame delta velocity = 0
        active_.push_back(true);
        bodyLayer_.push_back(0u);
        bodyMask_.push_back(0xFFFFFFFFu);
        bodyShapes_.emplace_back();
        broadphaseProxy_.push_back(DynamicAABBTree::NullNode);
        bodyIdx = static_cast<int>(bodies_.size()) - 1;
    }

    // No-shape inertia refresh (empty shapeList branch in updateInertia sets
    // the minI fallback).
    updateInertia(bodyIdx);
    return bodyIdx;
}

// addBody - register a body with one Box shape (single-collider convenience).
// Goes through addEmptyBody + attachShape; for multiple colliders call
// addEmptyBody + attachShape repeatedly. material is an explicit parameter
// (CP-3.2): the old implicit copy from body.friction/restitution fallback is
// gone since RigidBody no longer holds those fields (default = friction 0.5 /
// restitution 0.3, matching historical behavior).
int PhysicsWorld::addBody(const RigidBody &body, const OBB &obb,
                          const PhysicsMaterial &material)
{
    BodyPose pose;
    pose.position = obb.center;
    pose.orientation = obb.orientation;

    int bodyIdx = addEmptyBody(body, pose);

    // Attach one Box shape: local=identity, material from the caller.
    Shape shape;
    shape.type = ShapeType::Box;
    shape.halfExtents = obb.halfExtents;

    BodyPose localPose; // identity

    attachShape(bodyIdx, shape, localPose, material, 0u, 0xFFFFFFFFu, false);
    return bodyIdx;
}

void PhysicsWorld::removeBody(int bodyIndex)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()))
        return;
    // Mark the body inactive and take its shapes offline (slots are reused by
    // later attachShape calls).
    active_[bodyIndex] = false;
    for (int s : bodyShapes_[bodyIndex])
    {
        if (s >= 0 && s < static_cast<int>(shapeActive_.size()))
            shapeActive_[s] = false;
    }
    bodyShapes_[bodyIndex].clear();

    // Remove from broadphase in sync.
    if (bodyIndex < static_cast<int>(broadphaseProxy_.size()) &&
        broadphaseProxy_[bodyIndex] != DynamicAABBTree::NullNode)
    {
        broadphase_.remove(broadphaseProxy_[bodyIndex]);
        broadphaseProxy_[bodyIndex] = DynamicAABBTree::NullNode;
    }

    // Drop all persistent manifolds touching this body so a reused slot cannot
    // inherit the previous body's accumulated impulses.
    manifolds_.erase(std::remove_if(manifolds_.begin(), manifolds_.end(),
                                    [bodyIndex](const ContactManifold &m)
                                    { return m.bodyA == bodyIndex || m.bodyB == bodyIndex; }),
                     manifolds_.end());
}

void PhysicsWorld::clear()
{
    bodies_.clear();
    poses_.clear();
    prevPoses_.clear(); // keep in sync with poses_
    active_.clear();
    bodyLayer_.clear();
    bodyMask_.clear();
    bodyShapes_.clear();
    broadphaseProxy_.clear();
    broadphase_.clear();

    shapes_.clear();
    shapeLocal_.clear();
    shapeMaterial_.clear();
    shapeLayer_.clear();
    shapeMask_.clear();
    shapeIsTrigger_.clear();
    shapeBody_.clear();
    shapeActive_.clear();

    contacts_.clear();

    // Clear cross-frame contact-pair cache / events.
    prevContactPairs_.clear();
    currContactPairs_.clear();
    events_.clear();

    // Clear persistent manifolds.
    manifolds_.clear();

    stats_ = PhysicsStats{};

    accumulator_ = 0.0f;
    g_physFrame = 0;
    if (g_physLog.is_open())
        g_physLog.close();
}

// attachShape / detachShape - multi-collider API.
int PhysicsWorld::attachShape(int bodyIndex,
                              const Shape &shape,
                              const BodyPose &localPose,
                              const PhysicsMaterial &material,
                              uint32_t layer,
                              uint32_t mask,
                              bool isTrigger)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()))
        return -1;

    // Reuse a deactivated shape slot, else append.
    int shapeIdx = -1;
    for (int i = 0; i < static_cast<int>(shapeActive_.size()); ++i)
    {
        if (!shapeActive_[i])
        {
            shapeIdx = i;
            shapes_[i] = shape;
            shapeLocal_[i] = localPose;
            shapeMaterial_[i] = material;
            shapeLayer_[i] = layer;
            shapeMask_[i] = mask;
            shapeIsTrigger_[i] = isTrigger;
            shapeBody_[i] = bodyIndex;
            shapeActive_[i] = true;
            break;
        }
    }
    if (shapeIdx < 0)
    {
        shapes_.push_back(shape);
        shapeLocal_.push_back(localPose);
        shapeMaterial_.push_back(material);
        shapeLayer_.push_back(layer);
        shapeMask_.push_back(mask);
        shapeIsTrigger_.push_back(isTrigger);
        shapeBody_.push_back(bodyIndex);
        shapeActive_.push_back(true);
        shapeIdx = static_cast<int>(shapes_.size()) - 1;
    }

    bodyShapes_[bodyIndex].push_back(shapeIdx);

    // ConvexHull with a hullIndex: backfill the hullCache pointer so the
    // physics layer dereferences the index given at shape construction.
    if (shapes_[shapeIdx].type == ShapeType::ConvexHull)
    {
        int hi = shapes_[shapeIdx].hullIndex;
        if (hi >= 0 && hi < static_cast<int>(hulls_.size()))
            shapes_[shapeIdx].hullCache = &hulls_[hi];
        // Invalid hullIndex leaves hullCache null; shape_support returns a
        // degenerate value.
    }

    updateInertia(bodyIndex);

    // No broadphase insert here; refreshBroadphase() inserts/updates once at
    // the start of each singleStep, which avoids repeated insert+remove during
    // multi-collider registration and keeps broadphase consistent for queries
    // outside play (call refreshBroadphase() once first - see raycast/overlapAABB).
    return shapeIdx;
}

void PhysicsWorld::detachShape(int bodyIndex, int shapeIndex)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodyShapes_.size()))
        return;
    if (shapeIndex < 0 || shapeIndex >= static_cast<int>(shapeActive_.size()))
        return;
    if (shapeBody_[shapeIndex] != bodyIndex)
        return;

    shapeActive_[shapeIndex] = false;
    auto &list = bodyShapes_[bodyIndex];
    list.erase(std::remove(list.begin(), list.end(), shapeIndex), list.end());
    updateInertia(bodyIndex);
}

// registerConvexHull / getConvexHull - shared ConvexHull data registration.
// ----------------------------------------------------------------------------
// Pointer stability: shape.hullCache points into hulls_ (std::vector), so any
// reallocation invalidates all backfilled pointers. Strategy: reserve(32) on
// first registration; if more than 32 distinct hulls appear, rebuild all
// pointers after realloc. Current scenes stay under 32, so rebuild never fires.
// ============================================================================
int PhysicsWorld::registerConvexHull(const std::vector<glm::vec3> &localVertices)
{
    if (localVertices.empty())
        return -1;

    // Keep initial capacity (first grow to 32).
    if (hulls_.capacity() == 0)
        hulls_.reserve(32);

    const bool willRealloc = (hulls_.size() + 1 > hulls_.capacity());

    ConvexMeshCache cache;
    cache.vertices = localVertices;
    // Precompute local AABB (reused by shape_world_aabb / inertia).
    cache.localAABBMin = localVertices[0];
    cache.localAABBMax = localVertices[0];
    for (size_t i = 1; i < localVertices.size(); ++i)
    {
        cache.localAABBMin = glm::min(cache.localAABBMin, localVertices[i]);
        cache.localAABBMax = glm::max(cache.localAABBMax, localVertices[i]);
    }
    hulls_.push_back(std::move(cache));

    // A realloc invalidated every ConvexHull hullCache pointer - rebuild them all.
    if (willRealloc)
    {
        for (size_t si = 0; si < shapes_.size(); ++si)
        {
            if (!shapeActive_[si])
                continue;
            if (shapes_[si].type != ShapeType::ConvexHull)
                continue;
            int hi = shapes_[si].hullIndex;
            if (hi >= 0 && hi < static_cast<int>(hulls_.size()))
                shapes_[si].hullCache = &hulls_[hi];
            else
                shapes_[si].hullCache = nullptr;
        }
    }

    return static_cast<int>(hulls_.size()) - 1;
}

const ConvexMeshCache *PhysicsWorld::getConvexHull(int hullIndex) const
{
    if (hullIndex < 0 || hullIndex >= static_cast<int>(hulls_.size()))
        return nullptr;
    return &hulls_[hullIndex];
}

void PhysicsWorld::setBodyLayerMask(int bodyIndex, uint32_t layer, uint32_t mask)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()))
        return;
    bodyLayer_[bodyIndex] = layer;
    bodyMask_[bodyIndex] = mask;
}

// Force / wake API. Conventions:
//   - invalid bodyIndex or Static body returns early (Static takes no force/impulse)
//   - every entry point calls wakeBody: the whole island wakes the same frame
//     (island-wide wake replaces per-frame contagion - pushing the top of a
//     10-high stack wakes the whole chain immediately)
//   - worldPoint creates angular impulse/torque: impulse applied at
//     worldPoint - bodyPos forms a moment arm
void PhysicsWorld::wakeBody(int bodyIndex)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;

    // The sleeping flag only matters for Dynamic: Static never participates in
    // sleep checks (every sleep path does `if (isStatic) continue`) and
    // Kinematic poses are externally written, so only Dynamic gets
    // sleeping=false.
    //
    // Do NOT early-return on Static/Kinematic, or the island neighbor-wake
    // loop below is skipped - it is the required wake channel for:
    //   A) dragging a Static floor with the Gizmo in play mode
    //      (syncTransformToPhysics calls wakeBody(groundIndex)): a sleeping
    //      Dynamic cube resting on it must wake to receive the next frame's
    //      normal/friction impulses;
    //   B) dragging a Kinematic base: it never sleeps itself, but its Dynamic
    //      neighbors can only be woken through this path.
    if (bodies_[bodyIndex].bodyType == BodyType::Dynamic)
    {
        bodies_[bodyIndex].sleeping = false;
        bodies_[bodyIndex].sleepFrames = 0;
    }

    // Wake all Dynamic bodies of the island when island data exists.
    // islandId_ is refreshed by buildIslands at the end of detectCollisions and
    // is empty before the first frame - waking nothing for Static/Kinematic is
    // a correct no-op then.
    if (bodyIndex < (int)islandId_.size())
    {
        int isl = islandId_[bodyIndex];
        if (isl >= 0 && isl < (int)islands_.size())
        {
            for (int bi : islands_[isl].bodies)
            {
                if (bodies_[bi].bodyType != BodyType::Dynamic)
                    continue;
                bodies_[bi].sleeping = false;
                bodies_[bi].sleepFrames = 0;
            }
            islands_[isl].sleepFrames = 0;
        }
    }
}

void PhysicsWorld::applyImpulse(int bodyIndex, const glm::vec3 &impulse,
                                const glm::vec3 &worldPoint)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;
    auto &body = bodies_[bodyIndex];
    // Static/Kinematic take no impulse (kinematic velocity is pose-delta driven).
    if (body.isStatic() || body.isKinematic() || body.mass <= 0.0f)
        return;

    wakeBody(bodyIndex);

    const float invMass = body.inverseMass();
    body.velocity += invMass * impulse;

    const glm::vec3 r = worldPoint - poses_[bodyIndex].position;
    body.angularVelocity += body.inverseInertiaWorld * glm::cross(r, impulse);
}

void PhysicsWorld::applyForce(int bodyIndex, const glm::vec3 &force,
                              const glm::vec3 &worldPoint)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;
    auto &body = bodies_[bodyIndex];
    // Static/Kinematic take no force.
    if (body.isStatic() || body.isKinematic() || body.mass <= 0.0f)
        return;

    wakeBody(bodyIndex);

    body.forceAccum += force;
    const glm::vec3 r = worldPoint - poses_[bodyIndex].position;
    body.torqueAccum += glm::cross(r, force);
}

void PhysicsWorld::applyTorque(int bodyIndex, const glm::vec3 &torque)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;
    auto &body = bodies_[bodyIndex];
    // Static/Kinematic take no torque.
    if (body.isStatic() || body.isKinematic() || body.mass <= 0.0f)
        return;

    wakeBody(bodyIndex);
    body.torqueAccum += torque;
}

void PhysicsWorld::setLinearVelocity(int bodyIndex, const glm::vec3 &v)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;
    auto &body = bodies_[bodyIndex];
    // Kinematic velocity is pose-delta driven; a manual set would be
    // overwritten by next frame's refreshKinematicVelocities, so early-return
    // like Static.
    if (body.isStatic() || body.isKinematic())
        return;
    wakeBody(bodyIndex);
    body.velocity = v;
}

void PhysicsWorld::setAngularVelocity(int bodyIndex, const glm::vec3 &w)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;
    if (!active_[bodyIndex])
        return;
    auto &body = bodies_[bodyIndex];
    // Kinematic angular velocity is likewise pose-delta managed.
    if (body.isStatic() || body.isKinematic())
        return;
    wakeBody(bodyIndex);
    body.angularVelocity = w;
}

// updateOBB / updateShapeExtents - operate on the body's FIRST shape
// (compatibility semantics).
void PhysicsWorld::updateOBB(int bodyIndex, const OBB &obb)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(poses_.size()))
        return;
    poses_[bodyIndex].position = obb.center;
    poses_[bodyIndex].orientation = obb.orientation;

    // Sync first shape's halfExtents (keeps the "OBB as single shape" semantics).
    auto &list = bodyShapes_[bodyIndex];
    if (!list.empty())
    {
        int s = list.front();
        shapes_[s].halfExtents = obb.halfExtents;
    }
    updateInertia(bodyIndex);
}

void PhysicsWorld::updateShapeExtents(int bodyIndex, const glm::vec3 &halfExtents)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodyShapes_.size()))
        return;
    auto &list = bodyShapes_[bodyIndex];
    if (list.empty())
        return;
    int s = list.front();
    glm::vec3 &he = shapes_[s].halfExtents;
    const float eps = 1e-6f;
    if (std::abs(he.x - halfExtents.x) < eps &&
        std::abs(he.y - halfExtents.y) < eps &&
        std::abs(he.z - halfExtents.z) < eps)
        return; // unchanged; skip pointless inertia recompute
    he = halfExtents;
    updateInertia(bodyIndex);
}

void PhysicsWorld::setBodyPose(int bodyIndex, const BodyPose &pose)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(poses_.size()))
        return;
    // Refresh the world inverse inertia tensor only when orientation actually
    // changed: inverseInertiaWorld = R * I_local^-1 * R^T is unchanged by
    // position edits, avoiding a 3x3 inverse and two matrix multiplies.
    // Per-column epsilon compare (exact float equality is unstable); 1e-7f is
    // far below any meaningful rotation component change.
    const glm::mat3 &oldR = poses_[bodyIndex].orientation;
    const glm::mat3 &newR = pose.orientation;
    bool orientChanged = false;
    const float kEps = 1e-7f;
    for (int c = 0; c < 3 && !orientChanged; ++c)
    {
        for (int r = 0; r < 3; ++r)
        {
            if (std::fabs(oldR[c][r] - newR[c][r]) > kEps)
            {
                orientChanged = true;
                break;
            }
        }
    }

    // Wake same-island Dynamic neighbors on pose change.
    // Problem: moving a body externally (Gizmo-dragged Static floor, animated
    // Kinematic platform) causes geometric penetration next step, but a
    // sleeping neighbor never wakes: the new penetration cannot trip
    // `maxInitialImpact > wakeImpactThreshold` (both velocities are 0) nor
    // `wakeByNeighborMotion` (opposite velocity still 0), so the penetration is
    // never resolved.
    // Generality:
    //   * wakeBody skips the sleeping write for Static/Kinematic (they never
    //     sleep) but still runs the island neighbor-wake loop - the side effect
    //     we need.
    //   * A Dynamic body calling setBodyPose is equally reasonable: a pose jump
    //     should notify neighbors.
    //   * Unchanged poses (same orient/pos) early-skip to avoid spurious wakes.
    bool posChanged = false;
    {
        glm::vec3 dp = pose.position - poses_[bodyIndex].position;
        if (glm::dot(dp, dp) > kEps * kEps)
            posChanged = true;
    }

    poses_[bodyIndex] = pose;
    if (orientChanged)
        updateInertia(bodyIndex);

    if (posChanged || orientChanged)
        wakeBody(bodyIndex);
}

void PhysicsWorld::updateBody(int bodyIndex, const RigidBody &body)
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodies_.size()))
        return;

    RigidBody &current = bodies_[bodyIndex];

    // BodyType switches must reset runtime transient state, or the body carries
    // the old type's velocity/force/sleep into the new one - e.g. Static ->
    // Dynamic keeping sleeping=true skips the first frame's integrate. Generic
    // behavior for any body at any time, not a scene special case.
    const bool bodyTypeChanged = (current.bodyType != body.bodyType);
    const bool massChanged = (current.mass != body.mass);

    current = body;

    if (bodyTypeChanged)
    {
        current.velocity = glm::vec3(0.0f);
        current.angularVelocity = glm::vec3(0.0f);
        current.forceAccum = glm::vec3(0.0f);
        current.torqueAccum = glm::vec3(0.0f);
        current.sleeping = false;
        current.sleepFrames = 0;
    }

    // Either mass or bodyType change invalidates the inertia tensor; refresh it.
    if (bodyTypeChanged || massChanged)
        updateInertia(bodyIndex);
}

glm::vec3 PhysicsWorld::getPosition(int bodyIndex) const
{
    return poses_[bodyIndex].position;
}

// composeShapePose / composeShapeOBB - compose the shape's body-local pose
// with the body's world pose (pose returns BodyPose; OBB adds halfExtents).
// World position = body.pos + body.orient * shape.local.pos
// World orientation = body.orient * shape.local.orient (both rotation matrices)
// halfExtents is the shape's own local size, unaffected by the body.
BodyPose PhysicsWorld::composeShapePose(int shapeIndex) const
{
    BodyPose wp;
    int bi = shapeBody_[shapeIndex];
    const BodyPose &bp = poses_[bi];
    const BodyPose &sl = shapeLocal_[shapeIndex];
    wp.position = bp.position + bp.orientation * sl.position;
    wp.orientation = bp.orientation * sl.orientation;
    return wp;
}

OBB PhysicsWorld::composeShapeOBB(int shapeIndex) const
{
    OBB obb;
    BodyPose wp = composeShapePose(shapeIndex);
    obb.center = wp.position;
    obb.orientation = wp.orientation;
    obb.halfExtents = shapes_[shapeIndex].halfExtents;
    return obb;
}

// getOBB - world OBB of the body's FIRST shape (compatibility semantics).
OBB PhysicsWorld::getOBB(int bodyIndex) const
{
    if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodyShapes_.size()) ||
        bodyShapes_[bodyIndex].empty())
    {
        OBB obb;
        if (bodyIndex >= 0 && bodyIndex < static_cast<int>(poses_.size()))
        {
            obb.center = poses_[bodyIndex].position;
            obb.orientation = poses_[bodyIndex].orientation;
        }
        return obb;
    }
    return composeShapeOBB(bodyShapes_[bodyIndex].front());
}

// passLayerMask - bidirectional layer/mask test. layer is a bit index (0 = bit
// 0, up to 32 layers); mask is the set of layers allowed to collide. Two bodies
// collide iff A's mask contains B's layer bit AND B's mask contains A's layer
// bit - the standard bidirectional filter (Unity/Unreal/Bullet), not a special
// case.
bool PhysicsWorld::passLayerMask(int bodyIndexA, int bodyIndexB) const
{
    uint32_t layerA = bodyLayer_[bodyIndexA];
    uint32_t layerB = bodyLayer_[bodyIndexB];
    uint32_t maskA = bodyMask_[bodyIndexA];
    uint32_t maskB = bodyMask_[bodyIndexB];
    uint32_t bitA = 1u << (layerA & 31u);
    uint32_t bitB = 1u << (layerB & 31u);
    return (maskA & bitB) && (maskB & bitA);
}

// updateInertia - multi-collider inertia composition (parallel-axis theorem).
// I_body_local = sum_i [ R_i * diag(I_i_local) * R_i^T  +  m_i * ((d_i*d_i) * E - d_i x d_i) ]
// where R_i is shape_i's local rotation (body local space), d_i its local
// offset, and m_i = body.mass * volume_i / total_volume (mass split by volume).
// Single shape + identity local + zero offset degenerates to
// I_body_local = diag(I_shape_local).
void PhysicsWorld::updateInertia(int bodyIndex)
{
    auto &b = bodies_[bodyIndex];

    // Static/Kinematic take the zero-inertia branch (inverseInertiaWorld = 0):
    // no angular response to impulses, consistent with inverseMass()=0;
    // kinematic angular velocity comes entirely from orientation deltas in
    // refreshKinematicVelocities.
    if (b.isStatic() || b.isKinematic() || b.mass <= 0.0f)
    {
        b.inertiaLocalMat = glm::mat3(0.0f);
        b.inverseInertiaWorld = glm::mat3(0.0f);
        return;
    }

    const auto &shapeList = bodyShapes_[bodyIndex];
    if (shapeList.empty())
    {
        // Shapeless body: tiny inertia to avoid division by zero.
        const float minI = 1e-6f;
        b.inertiaLocalMat = glm::mat3(minI);
        glm::mat3 R = poses_[bodyIndex].orientation;
        glm::mat3 invI(0.0f);
        invI[0][0] = invI[1][1] = invI[2][2] = 1.0f / minI;
        b.inverseInertiaWorld = R * invI * glm::transpose(R);
        return;
    }

    // 1. Total volume (per-shape closed form / approximation).
    float totalVolume = 0.0f;
    for (int s : shapeList)
    {
        totalVolume += shape_volume(shapes_[s]);
    }
    if (totalVolume <= 0.0f)
        totalVolume = 1.0f;

    // 2. Parallel-axis composition.
    glm::mat3 Ibody(0.0f);
    for (int s : shapeList)
    {
        const Shape &sh = shapes_[s];
        float volume = shape_volume(sh);
        float mi = b.mass * volume / totalVolume;

        // I_shape_local is the shape's own diagonal inertia (Box closed form).
        glm::vec3 Iloc = shape_inertia_local(sh, mi);
        glm::mat3 Idiag(0.0f);
        Idiag[0][0] = Iloc.x;
        Idiag[1][1] = Iloc.y;
        Idiag[2][2] = Iloc.z;

        // R_i: shape's body-local rotation (shapeLocal_[s].orientation is a mat3).
        glm::mat3 Ri = shapeLocal_[s].orientation;
        // Rotated inertia: R * diag * R^T.
        glm::mat3 Irot = Ri * Idiag * glm::transpose(Ri);

        // Parallel-axis term: m_i * ((d.d) * E - d x d).
        glm::vec3 d = shapeLocal_[s].position;
        float ddot = glm::dot(d, d);
        // glm::outerProduct(a, b) yields out[j][i] = a[i]*b[j] (column-major);
        // for a == b it is the symmetric outer product d x d, equivalent to the
        // former hand-written expansion.
        glm::mat3 outer = glm::outerProduct(d, d);
        glm::mat3 parallel = glm::mat3(1.0f) * ddot - outer;

        Ibody += Irot + mi * parallel;
    }

    // Degenerate protection: keep diagonal elements from shrinking and blowing
    // up the inverse.
    const float minI = 1e-6f;
    Ibody[0][0] = std::max(Ibody[0][0], minI);
    Ibody[1][1] = std::max(Ibody[1][1], minI);
    Ibody[2][2] = std::max(Ibody[2][2], minI);

    b.inertiaLocalMat = Ibody;

    // World-space inverse inertia: R * I_local^-1 * R^T.
    glm::mat3 R = poses_[bodyIndex].orientation;
    glm::mat3 invILocal = glm::inverse(Ibody);
    b.inverseInertiaWorld = R * invILocal * glm::transpose(R);
}

// computeBodyAABB - body-level AABB (union of all active shapes' AABBs).
AABB PhysicsWorld::computeBodyAABB(int bodyIndex) const
{
    AABB box;
    box.min = poses_[bodyIndex].position;
    box.max = poses_[bodyIndex].position;

    bool first = true;
    for (int s : bodyShapes_[bodyIndex])
    {
        if (s < 0 || s >= static_cast<int>(shapeActive_.size()) || !shapeActive_[s])
            continue;

        // Shape world pose = body.pose + shape.local (unified via member function).
        BodyPose worldShapePose = composeShapePose(s);
        AABB sb = shape_world_aabb(shapes_[s], worldShapePose);

        if (first)
        {
            box = sb;
            first = false;
        }
        else
        {
            box.min = glm::min(box.min, sb.min);
            box.max = glm::max(box.max, sb.max);
        }
    }
    return box;
}

// refreshBroadphase - refresh every active body's proxy at the start of each
// singleStep. For each active body compute the latest AABB: insert when no
// proxy exists, else tree.update(proxy, aabb). Inactive/removed bodies already
// had broadphaseProxy_ cleared in removeBody.
//
// Updates every proxy each step, even sleeping ones. Sleeping AABBs rarely
// change, so remove+reinsert is slightly wasteful, but simple and correct; a
// fat-AABB + motion-prediction variant is deferred.
void PhysicsWorld::refreshBroadphase() const
{
    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        if (!active_[i])
            continue;
        AABB aabb = computeBodyAABB(i);
        int &proxy = broadphaseProxy_[i];
        if (proxy == DynamicAABBTree::NullNode)
            proxy = broadphase_.insert(aabb, i);
        else
            broadphase_.update(proxy, aabb);
    }
    // broadphaseNodeCount is aggregated at the end of stepSimulation.
}

// raycast - nearest ray hit. Broadphase queryRay filters candidates, then a
// "ray vs OBB" precise test per active shape, returning the smallest-t hit with
// bodyIndex/shapeIndex/world point/world normal. Ray vs OBB transforms the ray
// into OBB local space (removing rotation), i.e. a ray-vs-AABB slab test.
static bool rayOBBIntersect(const OBB &obb, const glm::vec3 &from, const glm::vec3 &dir,
                            float maxT, float &tOut, glm::vec3 &normalOut)
{
    // World ray -> OBB local space (orientation^T; OBB is orthonormal).
    glm::mat3 R = obb.orientation;
    glm::mat3 Rt = glm::transpose(R);
    glm::vec3 localFrom = Rt * (from - obb.center);
    glm::vec3 localDir = Rt * dir;

    float tmin = 0.0f;
    float tmax = maxT;
    int hitAxis = -1;
    float hitSign = 0.0f;

    for (int k = 0; k < 3; ++k)
    {
        // CP-1.4: parallel-axis degeneracy epsilon recalibrated 1e-20f -> 1e-8f.
        // Float's smallest normalized positive value is ~1.2e-38, so 1e-20f is
        // "almost exactly zero" for f32: reasonable degenerate inputs with
        // |localDir[k]| in [1e-20, 1e-8] got pushed through the slab formula,
        // and invD = 1/tiny amplified t1/t2 by 10^8..10^20, wrecking stability.
        // Industry practice (Bullet btRayAabb / Box2D b2RayCastOutput / Embree)
        // sits at 1e-6..1e-8; 1e-8f is the generic lower bound of f32's stable
        // division range.
        if (std::abs(localDir[k]) < 1e-8f)
        {
            // Ray parallel to this axis; origin must already be inside the
            // slab, else no hit.
            if (localFrom[k] < -obb.halfExtents[k] || localFrom[k] > obb.halfExtents[k])
                return false;
            continue;
        }
        float invD = 1.0f / localDir[k];
        float t1 = (-obb.halfExtents[k] - localFrom[k]) * invD;
        float t2 = (obb.halfExtents[k] - localFrom[k]) * invD;
        float sign = -1.0f; // hit the -half-extent face
        if (t1 > t2)
        {
            std::swap(t1, t2);
            sign = 1.0f;
        }
        if (t1 > tmin)
        {
            tmin = t1;
            hitAxis = k;
            hitSign = sign;
        }
        tmax = std::min(tmax, t2);
        if (tmin > tmax)
            return false;
    }
    tOut = tmin;
    if (hitAxis >= 0)
    {
        glm::vec3 localN(0.0f);
        localN[hitAxis] = hitSign;
        normalOut = R * localN; // world normal
    }
    else
    {
        normalOut = glm::vec3(0.0f);
    }
    return true;
}

bool PhysicsWorld::raycast(const Ray &ray, float maxT, uint32_t mask, RaycastHit &out) const
{
    // Refresh before querying so broadphase matches current poses even outside play.
    refreshBroadphase();

    out.bodyIndex = -1;
    float bestT = maxT;

    broadphase_.queryRay(ray.origin, ray.direction, maxT,
                         [&](int bodyIndex) -> bool
                         {
                             if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()) || !active_[bodyIndex])
                                 return true;
                             // layer/mask: the mask is "allowed to intersect";
                             // the body's layer bit must be set in it
                             uint32_t bit = 1u << (bodyLayer_[bodyIndex] & 31u);
                             if ((mask & bit) == 0)
                                 return true;

                             for (int s : bodyShapes_[bodyIndex])
                             {
                                 if (s < 0 || !shapeActive_[s])
                                     continue;
                                 if (shapeIsTrigger_[s])
                                     continue; // rays skip triggers (common convention; optional flag later)
                                 OBB obb = composeShapeOBB(s);
                                 float t;
                                 glm::vec3 n;
                                 if (rayOBBIntersect(obb, ray.origin, ray.direction, bestT, t, n) && t < bestT)
                                 {
                                     bestT = t;
                                     out.bodyIndex = bodyIndex;
                                     out.shapeIndex = s;
                                     out.t = t;
                                     out.point = ray.origin + t * ray.direction;
                                     out.normal = glm::length(n) > 0.0f ? glm::normalize(n) : glm::vec3(0.0f);
                                 }
                             }
                             return true; // keep scanning for the nearest
                         });

    return out.bodyIndex >= 0;
}

// overlapAABB - AABB-overlap body list (coarse only, no narrowphase).
int PhysicsWorld::overlapAABB(const AABB &aabb, uint32_t mask,
                              int *outBodies, int maxOut) const
{
    if (!outBodies || maxOut <= 0)
        return 0;
    refreshBroadphase();
    int count = 0;
    broadphase_.queryAABB(aabb,
                          [&](int bodyIndex)
                          {
                              if (count >= maxOut)
                                  return;
                              if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()) || !active_[bodyIndex])
                                  return;
                              uint32_t bit = 1u << (bodyLayer_[bodyIndex] & 31u);
                              if ((mask & bit) == 0)
                                  return;
                              // Dedup: queryAABB visits each leaf once, so body
                              // indices are naturally unique.
                              outBodies[count++] = bodyIndex;
                          });
    return count;
}

// Closest point from center to OBB (classic world->local clamp distance).
static float distancePointOBB(const glm::vec3 &p, const OBB &obb)
{
    glm::mat3 Rt = glm::transpose(obb.orientation);
    glm::vec3 local = Rt * (p - obb.center);
    glm::vec3 clamped = glm::clamp(local, -obb.halfExtents, obb.halfExtents);
    return glm::length(local - clamped);
}

int PhysicsWorld::overlapSphere(const glm::vec3 &center, float radius, uint32_t mask,
                                int *outBodies, int maxOut) const
{
    if (!outBodies || maxOut <= 0 || radius < 0.0f)
        return 0;
    refreshBroadphase();

    AABB aabb;
    aabb.min = center - glm::vec3(radius);
    aabb.max = center + glm::vec3(radius);

    int count = 0;
    broadphase_.queryAABB(aabb,
                          [&](int bodyIndex)
                          {
                              if (count >= maxOut)
                                  return;
                              if (bodyIndex < 0 || bodyIndex >= static_cast<int>(active_.size()) || !active_[bodyIndex])
                                  return;
                              uint32_t bit = 1u << (bodyLayer_[bodyIndex] & 31u);
                              if ((mask & bit) == 0)
                                  return;

                              // Precise test: center-to-any-shape distance <= radius.
                              bool hit = false;
                              for (int s : bodyShapes_[bodyIndex])
                              {
                                  if (s < 0 || !shapeActive_[s])
                                      continue;
                                  OBB obb = composeShapeOBB(s);
                                  if (distancePointOBB(center, obb) <= radius)
                                  {
                                      hit = true;
                                      break;
                                  }
                              }
                              if (hit)
                                  outBodies[count++] = bodyIndex;
                          });
    return count;
}

// stepSimulation - fixed-timestep accumulator.
void PhysicsWorld::stepSimulation(float deltaTime)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // Kinematic velocities derived once per frame (not per substep), so all
    // substeps share one velocity and solveVelocity's relVel keeps reflecting
    // base motion; prevPoses_ = poses_ snapshots once at the end for the next
    // frame's delta. Skip when deltaTime <= 0 (defensive; physics rarely sees
    // this).
    if (deltaTime > 0.0f)
        refreshKinematicVelocities(deltaTime);

    accumulator_ += deltaTime;
    float maxAccum = fixedTimeStep * maxSubSteps;
    if (accumulator_ > maxAccum)
        accumulator_ = maxAccum;

    int substeps = 0;
    while (accumulator_ >= fixedTimeStep)
    {
        singleStep(fixedTimeStep);
        accumulator_ -= fixedTimeStep;
        ++substeps;
    }

    // Snapshot poses once after all substeps (not per substep) - the next
    // frame's refreshKinematicVelocities diffs against this "previous pose".
    // Dynamic poses are snapshotted too but filtered by bodyType on the read
    // side, so they are never misused.
    if (prevPoses_.size() != poses_.size())
        prevPoses_.resize(poses_.size());
    // Manual loop instead of `prevPoses_ = poses_;` to avoid the assignment
    // churn; BodyPose is trivially copyable.
    for (size_t i = 0; i < poses_.size(); ++i)
        prevPoses_[i] = poses_[i];

    // Statistics.
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.lastStepMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    stats_.substepsLastFrame = substeps;
    stats_.bodyCount = static_cast<int>(bodies_.size());
    stats_.shapeCount = static_cast<int>(shapes_.size());

    int activeCount = 0, sleepingCount = 0;
    for (size_t i = 0; i < bodies_.size(); ++i)
    {
        if (!active_[i])
            continue;
        ++activeCount;
        if (bodies_[i].sleeping)
            ++sleepingCount;
    }
    stats_.activeBodyCount = activeCount;
    stats_.sleepingBodyCount = sleepingCount;
    stats_.broadphaseNodeCount = broadphase_.nodeCount();
}

// singleStep - one step: integrate -> detect -> respond -> damp.
void PhysicsWorld::singleStep(float dt)
{
    openPhysLogIfEnabled(debugLogEnabled);
    ++g_physFrame;

    // Refresh broadphase at step start (world AABBs of all active bodies into
    // the tree).
    refreshBroadphase();

    // CCD sweep: compute TOI for fast Dynamic bodies into pendingTOI_ so the
    // following integrate can clamp displacement and avoid tunneling through
    // thin Static objects.
    ccdPhase(dt);

    integrate(dt);

    // Velocity solve: single detect + resolve. No outer multi-pass velocity
    // solve, so restitution targets are not re-applied to the same collision
    // (restitutionTargetVel is recomputed from current velocity on every
    // resolveCollisions call). Stack stability comes from the multi-pass
    // position correction below and the SI inner loop (solverIterations=10).
    detectCollisions();
    if (!contacts_.empty())
        resolveCollisions(dt);

    // Damping runs after collision response with timestep-independent
    // compensation. Per-step decay coefficients for the base timestep (matches
    // legacy behavior at 1/60s).
    const float baseDt = 1.0f / 60.0f;
    const float baseLinearDamping = 0.999f; // fraction kept per base step
    const float baseAngularDamping = 0.98f; // fraction kept per base step
    const float linearDamping = std::pow(baseLinearDamping, dt / baseDt);
    const float angularDamping = std::pow(baseAngularDamping, dt / baseDt);
    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        if (!active_[i])
            continue;
        auto &b = bodies_[i];
        if (b.isStatic() || b.sleeping)
            continue;
        b.velocity *= linearDamping;
        b.angularVelocity *= angularDamping;
    }

    // Position penetration solve (generic split-impulse style):
    //   1. reuse the contacts_ just detected by the velocity solver; no
    //      re-detect, since detect clears contacts_ and would invalidate
    //      accumulated position impulses (standard Bullet/Box2D practice).
    //   2. per contact, iterated SI-style: each round pushes
    //      max(penetration - slop, 0) * percent, accumulated into
    //      posImpulse[ci], clamped >= 0 so bodies are never "pulled back".
    //   3. iterations coordinate between contact pairs, converging toward the
    //      "all penetrations resolved at once" solution.
    //   4. after each push the body center updates; the next round estimates
    //      remaining penetration from the accumulated push, no narrowphase rerun.
    //   5. temporary solver; the real persistent-manifold + split-impulse path
    //      lands later (see the physics roadmap).
    if (!contacts_.empty())
    {
        const int positionIterations = 8;
        const float positionPercent = 0.2f; // push 20% of remaining penetration per round; ~83% after 8 rounds
        const float penetrationSlop = 0.001f;

        // Per-contact accumulated position impulse (non-negative), iteration-
        // local, not kept across frames. Scratch members avoid per-frame heap
        // allocation; assign = clear + resize+fill, no malloc when capacity
        // suffices.
        auto &posImpulse = scratchPosImpulse_;
        posImpulse.assign(contacts_.size(), 0.0f);
        // Per-body accumulated position displacement (estimates remaining
        // penetration = initialPen minus both bodies' displacement projected on
        // the normal).
        auto &posDelta = scratchPosDelta_;
        posDelta.assign(bodies_.size(), glm::vec3(0.0f));
        // Initial penetration snapshot.
        auto &initialPen = scratchInitialPen_;
        initialPen.assign(contacts_.size(), 0.0f);
        for (size_t ci = 0; ci < contacts_.size(); ++ci)
            initialPen[ci] = contacts_[ci].penetration;

        for (int iter = 0; iter < positionIterations; ++iter)
        {
            bool anyApplied = false;
            for (size_t ci = 0; ci < contacts_.size(); ++ci)
            {
                const auto &c = contacts_[ci];
                auto &bodyA = bodies_[c.indexA];
                auto &bodyB = bodies_[c.indexB];
                float invMassA = bodyA.inverseMass();
                float invMassB = bodyB.inverseMass();
                float invMassSum = invMassA + invMassB;
                if (invMassSum <= 0.0f)
                    continue;

                // Estimate remaining penetration: initialPen minus the
                // relative displacement of both bodies along the normal -
                // (posDelta[B] - posDelta[A]) projected on normal is what was
                // already pushed apart.
                glm::vec3 relDelta = posDelta[c.indexB] - posDelta[c.indexA];
                float alreadySeparated = glm::dot(relDelta, c.normal);
                float currentPen = initialPen[ci] - alreadySeparated;
                if (currentPen <= penetrationSlop)
                    continue;

                // Baumgarte-like: push positionPercent of remaining penetration
                // per round.
                float correctionMag = (currentPen - penetrationSlop) * positionPercent / invMassSum;
                // Accumulate with clamp: posImpulse must stay >= 0 (no pulling
                // bodies back).
                float newImpulse = std::max(0.0f, posImpulse[ci] + correctionMag);
                float applied = newImpulse - posImpulse[ci];
                posImpulse[ci] = newImpulse;
                if (applied <= 0.0f)
                    continue;
                anyApplied = true;

                glm::vec3 correction = applied * c.normal;
                if (!bodyA.isStatic() && !bodyA.sleeping)
                {
                    glm::vec3 d = invMassA * correction;
                    poses_[c.indexA].position -= d;
                    posDelta[c.indexA] -= d;
                }
                if (!bodyB.isStatic() && !bodyB.sleeping)
                {
                    glm::vec3 d = invMassB * correction;
                    poses_[c.indexB].position += d;
                    posDelta[c.indexB] += d;
                }
            }
            if (!anyApplied)
                break;
        }
    }

    // Log output (first MAX_LOG_FRAMES steps, only when debugLogEnabled=true).
    if (debugLogEnabled && g_physFrame <= MAX_LOG_FRAMES && g_physLog.is_open())
    {
        g_physLog << "=== Step " << g_physFrame << " ===\n";
        for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
        {
            if (!active_[i] || bodies_[i].isStatic())
                continue;
            auto &b = bodies_[i];
            const glm::vec3 &p = poses_[i].position;
            float linSpeed = glm::length(b.velocity);
            float angSpeed = glm::length(b.angularVelocity);
            g_physLog << "  Body[" << i << "] pos=(" << p.x << "," << p.y << "," << p.z << ")"
                      << " vel=(" << b.velocity.x << "," << b.velocity.y << "," << b.velocity.z << ") |v|=" << linSpeed
                      << " angVel=(" << b.angularVelocity.x << "," << b.angularVelocity.y << "," << b.angularVelocity.z << ") |w|=" << angSpeed
                      << " sleeping=" << (b.sleeping ? 1 : 0) << " sleepFrames=" << b.sleepFrames
                      << "\n";
        }
        if (!contacts_.empty())
        {
            for (size_t ci = 0; ci < contacts_.size(); ++ci)
            {
                auto &c = contacts_[ci];
                g_physLog << "  Contact[" << ci << "] A=" << c.indexA << " B=" << c.indexB
                          << " pen=" << c.penetration
                          << " normal=(" << c.normal.x << "," << c.normal.y << "," << c.normal.z << ")"
                          << " cp=(" << c.contactPoint.x << "," << c.contactPoint.y << "," << c.contactPoint.z << ")"
                          << "\n";
            }
        }
        else
        {
            g_physLog << "  No contacts\n";
        }
        g_physLog.flush();
    }
}

// refreshKinematicVelocities - derive Kinematic velocities from pose deltas.
// ----------------------------------------------------------------------------
// Diffs poses_[i] vs prevPoses_[i] into linear/angular velocity for each
// Kinematic body. Called once by stepSimulation before the substep loop; the
// end of stepSimulation snapshots prevPoses_ = poses_ for the next frame.
//
// linearVel  = (pos - prevPos) / dt
// angularVel = axis-angle(prevOrient^T * orient) / dt
//              (axis-angle vector of the relative rotation between the two
//              orientation matrices; direction = axis, magnitude = angular speed)
//
// Teleport guard: displacement > kKinematicMaxStepSpeed * dt counts as a
// teleport (Gizmo jump / scene reset / script), so both velocities are zeroed
// to avoid injecting huge fake velocities that fling the Dynamic body above.
//
// Generic: branch only on bodyType==Kinematic, no
// per-body/per-scene/per-shape special cases.
void PhysicsWorld::refreshKinematicVelocities(float dt)
{
    if (dt <= 1e-8f)
        return;

    // Defensive: prevPoses_ must match poses_ length (kept in sync by
    // add/remove/clear; belt-and-suspenders here).
    if (prevPoses_.size() != poses_.size())
        prevPoses_.resize(poses_.size(), BodyPose{});

    const float invDt = 1.0f / dt;
    const float maxDisp = kKinematicMaxStepSpeed * dt;

    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        if (!active_[i])
            continue;
        auto &b = bodies_[i];
        if (!b.isKinematic())
            continue;

        // Linear velocity: (pos - prevPos) / dt.
        glm::vec3 dPos = poses_[i].position - prevPoses_[i].position;
        float dispMag = glm::length(dPos);
        if (dispMag > maxDisp)
        {
            // Teleport: zero both velocities (keep the next frame's solver from
            // seeing a huge fake speed).
            b.velocity = glm::vec3(0.0f);
            b.angularVelocity = glm::vec3(0.0f);
            continue;
        }
        b.velocity = dPos * invDt;

        // Angular velocity: extract the axis-angle vector of the relative
        // rotation from prevOrient to orient. R_rel = orient * prevOrient^T
        // (world-frame incremental rotation); quat gives a stable extraction.
        glm::quat qPrev = glm::quat_cast(prevPoses_[i].orientation);
        glm::quat qCurr = glm::quat_cast(poses_[i].orientation);
        // Shortest arc (quaternion double cover).
        if (glm::dot(qPrev, qCurr) < 0.0f)
            qCurr = -qCurr;
        glm::quat qRel = qCurr * glm::inverse(qPrev);
        qRel = glm::normalize(qRel);

        float angle = 2.0f * std::acos(glm::clamp(qRel.w, -1.0f, 1.0f));
        if (angle > 1e-6f)
        {
            float sinHalf = std::sqrt(std::max(0.0f, 1.0f - qRel.w * qRel.w));
            glm::vec3 axis(qRel.x, qRel.y, qRel.z);
            if (sinHalf > 1e-6f)
                axis /= sinHalf;
            else
                axis = glm::vec3(0.0f);
            // Wrap angle > pi to the shortest arc.
            if (angle > 3.14159265f)
                angle = angle - 2.0f * 3.14159265f;
            b.angularVelocity = axis * (angle * invDt);

            // Angular teleport guard: speed beyond maxAngularSpeed counts as
            // teleport.
            float aSpeed = glm::length(b.angularVelocity);
            if (aSpeed > maxAngularSpeed)
            {
                b.angularVelocity = glm::vec3(0.0f);
            }
        }
        else
        {
            b.angularVelocity = glm::vec3(0.0f);
        }
    }
}

// ccdPhase - Conservative Advancement CCD, called at the start of each
// singleStep before integrate. For each active Dynamic body:
//   1. skip slow bodies - trigger only when displacement this step exceeds
//      0.5 * min(shape AABB half-extent) (generic criterion, no special cases)
//   2. sweptAABB (start AABB inflated by linVel*dt) via broadphase_.queryAABB
//      for candidates
//   3. computeTOI only on dynamic-vs-static pairs (dynamic-vs-dynamic is left
//      to the discrete SI solver, avoiding multi-pair CCD sync complexity)
//   4. the smallest TOI across candidates lands in pendingTOI_[bi], consumed
//      by integrate
//
// Output: pendingTOI_ sized like bodies_, defaults 1.0; clamped bodies have
// 0 <= toi < 1.
void PhysicsWorld::ccdPhase(float dt)
{
    // Reset, resized in sync with bodies_ (1.0f = "no clamp").
    pendingTOI_.assign(bodies_.size(), 1.0f);

    if (!ccdEnabled || dt <= 0.0f)
        return;

    const float tolerance = 1e-3f; // gjk_distance tolerance
    const int maxIter = 16;        // max CA iterations
    const float dispFactor = 0.5f; // generic trigger: displacement > 0.5 * minHalfExtent

    for (int bi = 0; bi < static_cast<int>(bodies_.size()); ++bi)
    {
        if (!active_[bi])
            continue;
        const auto &body = bodies_[bi];
        if (body.isStatic() || body.sleeping)
            continue;
        if (bodyShapes_[bi].empty())
            continue;

        // Estimate this body's linear displacement over dt.
        glm::vec3 velPredicted = body.velocity + (gravity + body.forceAccum * body.inverseMass()) * dt;
        float disp = glm::length(velPredicted) * dt;

        // Generic CCD trigger: disp > dispFactor * min(AABB radius over all shapes).
        float minHalf = std::numeric_limits<float>::infinity();
        for (int shIdx : bodyShapes_[bi])
        {
            if (shIdx < 0 || shIdx >= static_cast<int>(shapes_.size()))
                continue;
            if (!shapeActive_[shIdx])
                continue;
            BodyPose shPose = composeShapePose(shIdx);
            AABB a = shape_world_aabb(shapes_[shIdx], shPose);
            glm::vec3 half = (a.max - a.min) * 0.5f;
            float mn = std::min(half.x, std::min(half.y, half.z));
            if (mn < minHalf)
                minHalf = mn;
        }
        if (!std::isfinite(minHalf) || minHalf <= 1e-6f)
            continue;
        if (disp <= dispFactor * minHalf)
            continue;

        // CCD triggered: sweptAABB broadphase coarse pass per shape.
        float minTOI = 1.0f;
        glm::vec3 hitNormal(1, 0, 0);

        for (int shIdx : bodyShapes_[bi])
        {
            if (shIdx < 0 || shIdx >= static_cast<int>(shapes_.size()))
                continue;
            if (!shapeActive_[shIdx] || shapeIsTrigger_[shIdx])
                continue;

            // Current world shape AABB (via member function).
            BodyPose shPose0 = composeShapePose(shIdx);
            AABB a0 = shape_world_aabb(shapes_[shIdx], shPose0);

            // SweptAABB: inflate the AABB along velPredicted*dt.
            AABB swept;
            glm::vec3 disp3 = velPredicted * dt;
            swept.min = glm::min(a0.min, a0.min + disp3);
            swept.max = glm::max(a0.max, a0.max + disp3);

            std::vector<int> candidateBodies;
            broadphase_.queryAABB(swept, [&](int candBody)
                                  {
                if (candBody == bi)
                    return;
                if (candBody < 0 || candBody >= static_cast<int>(active_.size()))
                    return;
                if (!active_[candBody])
                    return;
                // First CCD version: dynamic-vs-static only.
                if (!bodies_[candBody].isStatic())
                    return;
                // Same body-level layer/mask filter as detectCollisions.
                if (!passLayerMask(bi, candBody))
                    return;
                candidateBodies.push_back(candBody); });

            for (int candBi : candidateBodies)
            {
                // computeTOI per shape of candBi, keep the minimum.
                for (int cshIdx : bodyShapes_[candBi])
                {
                    if (cshIdx < 0 || cshIdx >= static_cast<int>(shapes_.size()))
                        continue;
                    if (!shapeActive_[cshIdx] || shapeIsTrigger_[cshIdx])
                        continue;
                    BodyPose cshPose = composeShapePose(cshIdx);

                    glm::vec3 normalOut;
                    // Static linVel = 0; the Dynamic side uses velPredicted
                    // (gravity applied first).
                    float toi = computeTOI(shapes_[shIdx], shPose0, velPredicted,
                                           shapes_[cshIdx], cshPose, glm::vec3(0),
                                           dt, tolerance, maxIter, normalOut);
                    if (toi < minTOI)
                    {
                        minTOI = toi;
                        hitNormal = normalOut;
                    }
                }
            }
        }

        // Write pendingTOI_; < 1 means this step's dt must be clamped. Leave a
        // small margin (0.99) so the clamp does not land exactly at contact and
        // leave penetration for the next frame.
        if (minTOI < 1.0f)
            pendingTOI_[bi] = std::max(0.0f, minTOI * 0.99f);
    }
}

// integrate - semi-implicit Euler integration (translation + rotation).
void PhysicsWorld::integrate(float dt)
{
    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        if (!active_[i])
            continue;
        auto &b = bodies_[i];
        // Kinematic pose/velocity are managed externally by
        // refreshKinematicVelocities; the integrator must never move them.
        if (b.isStatic() || b.isKinematic())
            continue;

        // Sleeping bodies skip integration; external force/torque wakes them.
        if (b.sleeping)
        {
            const float wakeForceThreshold = 1e-4f;
            const float wakeTorqueThreshold = 1e-4f;
            if (glm::length(b.forceAccum) > wakeForceThreshold || glm::length(b.torqueAccum) > wakeTorqueThreshold)
            {
                b.sleeping = false;
                b.sleepFrames = 0;
            }
            else
            {
                b.forceAccum = glm::vec3(0.0f);
                b.torqueAccum = glm::vec3(0.0f);
                b.velocity = glm::vec3(0.0f);
                b.angularVelocity = glm::vec3(0.0f);
                continue;
            }
        }

        // Linear integration.
        glm::vec3 accel = gravity + b.forceAccum * b.inverseMass();
        b.velocity += accel * dt;

        // The hard maxLinearSpeed clamp is gone; CCD (Conservative Advancement
        // TOI) sweeps fast bodies at the start of singleStep and clamps dt, so
        // thin objects are not tunneled through without a magic speed cap.
        //
        // pendingTOI_[i] in [0, 1]: when CCD found a Static-body hit within
        // this step, TOI is the fraction of dt actually integrated; otherwise
        // 1.0 (full step).
        float toiFrac = (i < static_cast<int>(pendingTOI_.size())) ? pendingTOI_[i] : 1.0f;
        float effectiveDt = dt * toiFrac;

        poses_[i].position += b.velocity * effectiveDt;

        // Angular integration: angular acceleration = I^-1 * torque.
        glm::vec3 angAccel = b.inverseInertiaWorld * b.torqueAccum;
        b.angularVelocity += angAccel * dt;

        // Angular velocity clamp: dynamic cap of 90 deg per step (pi/dt*0.5),
        // absolute cap from the member field maxAngularSpeed.
        const float dtAngularCap = 3.14159265f / std::max(dt, 1e-4f) * 0.5f;
        const float angularCap = std::min(dtAngularCap, maxAngularSpeed);
        float aSpeed = glm::length(b.angularVelocity);
        if (aSpeed > angularCap)
        {
            b.angularVelocity *= angularCap / aSpeed;
        }

        // Damping moved into singleStep after resolveCollisions so collision
        // response never sees damping-polluted normal velocity, keeping
        // restitution consistent with the UI values.

        // Advance orientation with angular velocity using effectiveDt (in sync
        // with translation).
        float angle = glm::length(b.angularVelocity);
        if (angle > 1e-6f)
        {
            glm::vec3 axis = b.angularVelocity / angle;
            float halfAngle = angle * effectiveDt * 0.5f;
            // Incremental quaternion: q = cos(theta/2) + sin(theta/2) * axis.
            glm::quat dq(std::cos(halfAngle),
                         std::sin(halfAngle) * axis.x,
                         std::sin(halfAngle) * axis.y,
                         std::sin(halfAngle) * axis.z);
            // Apply rotation and orthonormalize.
            glm::mat3 dR = glm::mat3_cast(glm::normalize(dq));
            glm::mat3 &R = poses_[i].orientation;
            R = dR * R;

            // Re-orthonormalize against drift - the third column must be
            // renormalized too, else float accumulation scales orientation
            // slightly.
            R[0] = glm::normalize(R[0]);
            R[1] = glm::normalize(R[1] - glm::dot(R[1], R[0]) * R[0]);
            R[2] = glm::normalize(glm::cross(R[0], R[1]));
        } // clear force/torque accumulators
        b.forceAccum = glm::vec3(0.0f);
        b.torqueAccum = glm::vec3(0.0f);

        // Refresh the world inertia tensor.
        updateInertia(i);
    }
}

// detectCollisions - broadphase selfOverlap + event production.
// ----------------------------------------------------------------------------
//   1. broadphase_.selfOverlap returns candidate body pairs (AABB overlap,
//      O(n log n) amortized)
//   2. per candidate pair:
//       - skip same body, both-static, or layer/mask-filtered pairs
//       - narrowphase per body.shape x shape: testOBBOverlap + computeContact
//       - trigger: not added to contacts_, but sampled for events
//       - non-trigger: added to contacts_ and sampled
//   3. this frame's contacting pair keys (long long = min<<32 | max) sorted
//      into currContactPairs_; a linear merge with prevContactPairs_ yields
//      Enter/Stay/Exit
void PhysicsWorld::detectCollisions()
{
    // Persistent-manifold management:
    //   - mark all old manifolds alive=false and matched=false first
    //   - narrowphase produces new manifolds -> mergeManifoldWithPersistent
    //     (match + inherit impulses)
    //   - manifolds not alive this round map to Exit events; fresh alive ones
    //     to Enter/Stay
    //   - drop alive=false manifolds at the end, ready for the next frame
    for (auto &m : manifolds_)
    {
        m.alive = false;
        for (int k = 0; k < m.pointCount; ++k)
            m.points[k].matched = false;
    }

    contacts_.clear();
    currContactPairs_.clear();

    // Cache per-pair trigger flag and representative shape/normal/point for
    // events.
    struct PairSample
    {
        long long key;
        int bodyA, bodyB;
        int shapeA, shapeB;
        glm::vec3 point;
        glm::vec3 normal;
        bool isTrigger;
    };
    std::vector<PairSample> samples;
    samples.reserve(64);

    int broadPairCount = 0;

    auto processBodyPair = [&](int bi, int bj)
    {
        if (bi == bj)
            return;
        if (bi > bj)
            std::swap(bi, bj);
        if (!active_[bi] || !active_[bj])
            return;
        if (bodies_[bi].isStatic() && bodies_[bj].isStatic())
            return;
        if (!passLayerMask(bi, bj))
            return;

        ++broadPairCount;

        // Whether this body pair already produced a sample/manifold (keep only
        // the first hitting shape pair as representative). A pair may carry
        // several colliders; manifolds are processed per shape-pair but merged
        // into a (bodyA, bodyB)-level persistent slot, so different shape-pairs
        // overwrite each other - equivalent to earlier phases in the single-
        // collider-per-body case; per-shape manifold refinement is deferred
        // (kept functional and drift-free here).
        bool pairRecorded = false;
        long long pairKey = ((long long)bi << 32) | (unsigned)bj;

        const auto &listA = bodyShapes_[bi];
        const auto &listB = bodyShapes_[bj];

        for (int si : listA)
        {
            if (si < 0 || !shapeActive_[si])
                continue;
            const Shape &sA = shapes_[si];
            for (int sj : listB)
            {
                if (sj < 0 || !shapeActive_[sj])
                    continue;
                const Shape &sB = shapes_[sj];

                // Build this shape-pair's BodyPose via composeShapePose, the
                // same helper used by ccd / computeBodyAABB.
                BodyPose poseI = composeShapePose(si);
                BodyPose poseJ = composeShapePose(sj);

                ContactManifold fresh;
                bool gotContact = false;

                if (sA.type == ShapeType::Box && sB.type == ShapeType::Box)
                {
                    // Box-Box: keep the SAT fast path (4-point manifold +
                    // Sutherland-Hodgman clipping). CP-4.1 removed the
                    // `testOBBOverlap` pre-check here: computeBoxBoxManifold
                    // already runs the full 15-axis SAT and returns false on
                    // separation, so the pre-check was pure duplicate work.
                    // Numerics unchanged (SAT logic untouched); one SAT saved
                    // per Box-Box narrowphase. testOBBOverlap remains a public
                    // API (used by test_sat_overlap).
                    OBB obbI = composeShapeOBB(si);
                    OBB obbJ = composeShapeOBB(sj);
                    if (!computeBoxBoxManifold(obbI, obbJ, bi, bj, si, sj, fresh))
                        continue;
                    gotContact = fresh.pointCount > 0;
                }
                else
                {
                    // Any non-Box -> GJK/EPA generic path.
                    // AABB pre-check first so far-apart pairs never call GJK.
                    AABB aI = shape_world_aabb(sA, poseI);
                    AABB aJ = shape_world_aabb(sB, poseJ);
                    bool aabbOverlap = (aI.min.x <= aJ.max.x && aI.max.x >= aJ.min.x &&
                                        aI.min.y <= aJ.max.y && aI.max.y >= aJ.min.y &&
                                        aI.min.z <= aJ.max.z && aI.max.z >= aJ.min.z);
                    if (!aabbOverlap)
                        continue;

                    Simplex simp;
                    if (!gjk_intersect(sA, poseI, sB, poseJ, simp))
                        continue;
                    if (!epa_manifold(sA, poseI, sB, poseJ, simp, fresh))
                        continue;
                    // Fill the fields EPA leaves unset.
                    fresh.bodyA = bi;
                    fresh.bodyB = bj;
                    fresh.shapeA = si;
                    fresh.shapeB = sj;
                    gotContact = fresh.pointCount > 0;
                }

                if (!gotContact)
                    continue;

                bool isTrigger = shapeIsTrigger_[si] || shapeIsTrigger_[sj];
                fresh.isTrigger = isTrigger;

                if (!isTrigger)
                {
                    // Cross-frame match -> persistent manifold.
                    mergeManifoldWithPersistent(fresh);

                    // Expand the merged manifold into ContactInfo (for external
                    // UI/log and later body-level support stats in
                    // resolveCollisions). Locate the just-merged manifold.
                    for (auto &m : manifolds_)
                    {
                        if (m.bodyA == bi && m.bodyB == bj && m.alive)
                        {
                            for (int k = 0; k < m.pointCount; ++k)
                            {
                                ContactInfo ci;
                                ci.indexA = bi;
                                ci.indexB = bj;
                                ci.shapeIndexA = m.shapeA;
                                ci.shapeIndexB = m.shapeB;
                                ci.normal = m.normal;
                                ci.penetration = m.points[k].penetration;
                                ci.contactPoint = m.points[k].worldPoint;
                                // CP-3.1: no Jn/Jt snapshot copied here -
                                // consumers read the live values from
                                // getManifolds()[mi].points[k].
                                contacts_.push_back(ci);
                            }
                            break;
                        }
                    }
                }

                if (!pairRecorded)
                {
                    PairSample ps;
                    ps.key = pairKey;
                    ps.bodyA = bi;
                    ps.bodyB = bj;
                    ps.shapeA = si;
                    ps.shapeB = sj;
                    ps.point = fresh.points[0].worldPoint;
                    ps.normal = fresh.normal;
                    ps.isTrigger = isTrigger;
                    samples.push_back(ps);
                    pairRecorded = true;
                }
            }
        }
    };

    // broadphase selfOverlap yields all AABB-overlapping body pairs.
    broadphase_.selfOverlap([&](int a, int b)
                            { processBodyPair(a, b); });

    // Drop dead manifolds (not alive this frame = the pair separated).
    manifolds_.erase(std::remove_if(manifolds_.begin(), manifolds_.end(),
                                    [](const ContactManifold &m)
                                    { return !m.alive; }),
                     manifolds_.end());

    // Cross-frame differencing -> events_.
    events_.clear();
    currContactPairs_.reserve(samples.size());
    for (auto &s : samples)
        currContactPairs_.push_back(s.key);
    std::sort(currContactPairs_.begin(), currContactPairs_.end());

    // key -> sample map (currContactPairs_ deduped/sorted; one sample per key).
    auto findSampleByKey = [&](long long key) -> const PairSample *
    {
        for (auto &s : samples)
            if (s.key == key)
                return &s;
        return nullptr;
    };

    // Ordered merge diff of the two ascending arrays.
    size_t i = 0, j = 0;
    while (i < prevContactPairs_.size() && j < currContactPairs_.size())
    {
        long long pk = prevContactPairs_[i];
        long long ck = currContactPairs_[j];
        if (pk < ck)
        {
            ContactEvent ev;
            ev.bodyA = (int)(pk >> 32);
            ev.bodyB = (int)(pk & 0xFFFFFFFF);
            ev.phase = ContactPhase::Exit;
            ev.impulse = 0.0f;
            events_.push_back(ev);
            ++i;
        }
        else if (pk > ck)
        {
            const PairSample *s = findSampleByKey(ck);
            ContactEvent ev;
            ev.bodyA = (int)(ck >> 32);
            ev.bodyB = (int)(ck & 0xFFFFFFFF);
            ev.phase = ContactPhase::Enter;
            if (s)
            {
                ev.shapeA = s->shapeA;
                ev.shapeB = s->shapeB;
                ev.point = s->point;
                ev.normal = s->normal;
                ev.isTrigger = s->isTrigger;
            }
            events_.push_back(ev);
            ++j;
        }
        else
        {
            const PairSample *s = findSampleByKey(ck);
            ContactEvent ev;
            ev.bodyA = (int)(ck >> 32);
            ev.bodyB = (int)(ck & 0xFFFFFFFF);
            ev.phase = ContactPhase::Stay;
            if (s)
            {
                ev.shapeA = s->shapeA;
                ev.shapeB = s->shapeB;
                ev.point = s->point;
                ev.normal = s->normal;
                ev.isTrigger = s->isTrigger;
            }
            events_.push_back(ev);
            ++i;
            ++j;
        }
    }
    while (i < prevContactPairs_.size())
    {
        long long pk = prevContactPairs_[i++];
        ContactEvent ev;
        ev.bodyA = (int)(pk >> 32);
        ev.bodyB = (int)(pk & 0xFFFFFFFF);
        ev.phase = ContactPhase::Exit;
        events_.push_back(ev);
    }
    while (j < currContactPairs_.size())
    {
        long long ck = currContactPairs_[j++];
        const PairSample *s = findSampleByKey(ck);
        ContactEvent ev;
        ev.bodyA = (int)(ck >> 32);
        ev.bodyB = (int)(ck & 0xFFFFFFFF);
        ev.phase = ContactPhase::Enter;
        if (s)
        {
            ev.shapeA = s->shapeA;
            ev.shapeB = s->shapeB;
            ev.point = s->point;
            ev.normal = s->normal;
            ev.isTrigger = s->isTrigger;
        }
        events_.push_back(ev);
    }

    // This frame becomes the previous; swap avoids copying.
    prevContactPairs_.swap(currContactPairs_);

    // Stats.
    stats_.broadphasePairs = broadPairCount;
    stats_.contactCount = static_cast<int>(contacts_.size());

    // Build island structures from this frame's alive manifolds (used by
    // debugIslandIds, the island-scoped solver, and sleep checks).
    buildIslands();

    // One ContactConstraint per manifold this frame, indices aligned with
    // manifolds_. Rebuilt every frame (manifold positions may change; pooling
    // deferred). Objects are built even for !alive / trigger / both-static
    // manifolds: ContactConstraint::prepare sets pointCache_[k].skip there,
    // numerically equivalent to the old resolveCollisions PointCtx.skip.
    //
    // Invariant (CP-1.2 / CP-1.3 depend on it): after this loop,
    //   contactConstraints_.size() == manifolds_.size()
    //   and every element != nullptr (make_unique never returns null).
    // Any change that breaks this invariant (e.g. skipping some manifolds)
    // must also update the solve-loop guards in warmStartManifolds /
    // resolveCollisions.
    contactConstraints_.clear();
    contactConstraints_.reserve(manifolds_.size());
    for (int mi = 0; mi < static_cast<int>(manifolds_.size()); ++mi)
        contactConstraints_.emplace_back(std::make_unique<ContactConstraint>(mi));
    assert(contactConstraints_.size() == manifolds_.size());
}

// testOBBOverlap - OBB-OBB 15-axis SAT separation test.
bool PhysicsWorld::testOBBOverlap(const OBB &a, const OBB &b) const
{
    const float EPSILON = 1e-6f;

    glm::vec3 d = b.center - a.center;

    float R[3][3], AbsR[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            R[i][j] = glm::dot(a.orientation[i], b.orientation[j]);
            AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
        }

    float ra, rb, sep;

    // A's 3 local axes.
    for (int i = 0; i < 3; ++i)
    {
        ra = a.halfExtents[i];
        rb = b.halfExtents[0] * AbsR[i][0] + b.halfExtents[1] * AbsR[i][1] + b.halfExtents[2] * AbsR[i][2];
        sep = std::abs(glm::dot(d, a.orientation[i]));
        if (sep > ra + rb)
            return false;
    }

    // B's 3 local axes.
    for (int j = 0; j < 3; ++j)
    {
        ra = a.halfExtents[0] * AbsR[0][j] + a.halfExtents[1] * AbsR[1][j] + a.halfExtents[2] * AbsR[2][j];
        rb = b.halfExtents[j];
        sep = std::abs(glm::dot(d, b.orientation[j]));
        if (sep > ra + rb)
            return false;
    }

    // 9 cross-product axes.
    // A0 x B0
    ra = a.halfExtents[1] * AbsR[2][0] + a.halfExtents[2] * AbsR[1][0];
    rb = b.halfExtents[1] * AbsR[0][2] + b.halfExtents[2] * AbsR[0][1];
    sep = std::abs(glm::dot(d, a.orientation[2]) * R[1][0] - glm::dot(d, a.orientation[1]) * R[2][0]);
    if (sep > ra + rb)
        return false;

    // A0 x B1
    ra = a.halfExtents[1] * AbsR[2][1] + a.halfExtents[2] * AbsR[1][1];
    rb = b.halfExtents[0] * AbsR[0][2] + b.halfExtents[2] * AbsR[0][0];
    sep = std::abs(glm::dot(d, a.orientation[2]) * R[1][1] - glm::dot(d, a.orientation[1]) * R[2][1]);
    if (sep > ra + rb)
        return false;

    // A0 x B2
    ra = a.halfExtents[1] * AbsR[2][2] + a.halfExtents[2] * AbsR[1][2];
    rb = b.halfExtents[0] * AbsR[0][1] + b.halfExtents[1] * AbsR[0][0];
    sep = std::abs(glm::dot(d, a.orientation[2]) * R[1][2] - glm::dot(d, a.orientation[1]) * R[2][2]);
    if (sep > ra + rb)
        return false;

    // A1 x B0
    ra = a.halfExtents[0] * AbsR[2][0] + a.halfExtents[2] * AbsR[0][0];
    rb = b.halfExtents[1] * AbsR[1][2] + b.halfExtents[2] * AbsR[1][1];
    sep = std::abs(glm::dot(d, a.orientation[0]) * R[2][0] - glm::dot(d, a.orientation[2]) * R[0][0]);
    if (sep > ra + rb)
        return false;

    // A1 x B1
    ra = a.halfExtents[0] * AbsR[2][1] + a.halfExtents[2] * AbsR[0][1];
    rb = b.halfExtents[0] * AbsR[1][2] + b.halfExtents[2] * AbsR[1][0];
    sep = std::abs(glm::dot(d, a.orientation[0]) * R[2][1] - glm::dot(d, a.orientation[2]) * R[0][1]);
    if (sep > ra + rb)
        return false;

    // A1 x B2
    ra = a.halfExtents[0] * AbsR[2][2] + a.halfExtents[2] * AbsR[0][2];
    rb = b.halfExtents[0] * AbsR[1][1] + b.halfExtents[1] * AbsR[1][0];
    sep = std::abs(glm::dot(d, a.orientation[0]) * R[2][2] - glm::dot(d, a.orientation[2]) * R[0][2]);
    if (sep > ra + rb)
        return false;

    // A2 x B0
    ra = a.halfExtents[0] * AbsR[1][0] + a.halfExtents[1] * AbsR[0][0];
    rb = b.halfExtents[1] * AbsR[2][2] + b.halfExtents[2] * AbsR[2][1];
    sep = std::abs(glm::dot(d, a.orientation[1]) * R[0][0] - glm::dot(d, a.orientation[0]) * R[1][0]);
    if (sep > ra + rb)
        return false;

    // A2 x B1
    ra = a.halfExtents[0] * AbsR[1][1] + a.halfExtents[1] * AbsR[0][1];
    rb = b.halfExtents[0] * AbsR[2][2] + b.halfExtents[2] * AbsR[2][0];
    sep = std::abs(glm::dot(d, a.orientation[1]) * R[0][1] - glm::dot(d, a.orientation[0]) * R[1][1]);
    if (sep > ra + rb)
        return false;

    // A2 x B2
    ra = a.halfExtents[0] * AbsR[1][2] + a.halfExtents[1] * AbsR[0][2];
    rb = b.halfExtents[0] * AbsR[2][1] + b.halfExtents[1] * AbsR[2][0];
    sep = std::abs(glm::dot(d, a.orientation[1]) * R[0][2] - glm::dot(d, a.orientation[0]) * R[1][2]);
    if (sep > ra + rb)
        return false;

    return true;
}

// computeBoxBoxManifold - OBB-OBB manifold entry (multi-point, up to 4 points).
// ----------------------------------------------------------------------------
//   1. 15-axis SAT finds the minimum-penetration axis and bestAxis (normal
//      convention A -> B). Also does the separation test: separated -> return
//      false, so callers need no testOBBOverlap pre-check (removed, CP-4.1).
//   2. Reference/incident face choice: the face most parallel to bestAxis is
//      the reference, the other box's facing face is incident.
//   3. Sutherland-Hodgman side clipping: the incident face's 4-vertex polygon
//      is clipped against the reference face's 4 side planes, up to 8 vertices.
//   4. Penetration filter + reduction: drop vertices with signed distance > 0;
//      when > 4 candidates remain, pick the 4 most spread points (deepest +
//      farthest + max/min signed-area, per Box2D/Bullet); otherwise keep all.
//   5. Degenerate protection: if clipping leaves no candidates (edge-cross axis
//      picked / extreme geometry), fall back to the single deepest incident
//      support point and the standard formula.
//
// Cross-frame warm start, the two-tangent friction cone and island sleep are
// handled outside by the persistent-manifold mechanism
// (mergeManifoldWithPersistent). This function only produces a stable,
// symmetric, up-to-4-point manifold for the current frame.
bool PhysicsWorld::computeBoxBoxManifold(const OBB &a, const OBB &b,
                                         int idxA, int idxB,
                                         int shapeA, int shapeB,
                                         ContactManifold &out) const
{
    const float EPSILON = 1e-6f;

    // 1. 15-axis SAT for the minimum-penetration axis. Follows Ericson's
    // "Real-Time Collision Detection" sec. 4.4.1 without explicitly normalizing
    // cross axes: sep/ra/rb per edge-cross axis come straight from the rotation
    // matrix R. When axes are aligned (or nearly so), some edge-cross axes are
    // degenerate in ra/rb, so `faceParallelEps` skips them and hands the pen
    // candidacy to the face axes.
    //   - faceParallelEps = 1 - 1e-4: |cos| above this threshold counts as
    //     "almost parallel"
    glm::vec3 d = b.center - a.center;

    // R[i][j] = a.orient[i] . b.orient[j]; AbsR's EPSILON is Ericson's numeric
    // guard against degenerate sep==0 / ra+rb==0 branches.
    float R[3][3], AbsR[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            R[i][j] = glm::dot(a.orientation[i], b.orientation[j]);
            AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
        }

    // "Almost parallel" test between corresponding axes: any |R[i][j]| near 1.
    // When A's and B's local frames differ only by an axis permutation, one of
    // the 9 edge-crosses is near the zero vector - Ericson's formula then
    // derives ra/rb from two EPSILON terms, pen ~ 2*eps, and this fake positive
    // hijacks minPen. Skip edge-cross axes with |R| > 1 - 1e-4.
    const float faceParallelEps = 1e-4f;
    auto edgeDegenerate = [&](int i, int j) -> bool
    {
        // |cross(a.orient[i], b.orient[j])| = sqrt(1 - R[i][j]^2);
        // |R[i][j]| > 1 - faceParallelEps => magnitude < sqrt(2*faceParallelEps)
        // ~= 0.014.
        return std::abs(R[i][j]) > 1.0f - faceParallelEps;
    };

    float minPen = std::numeric_limits<float>::max();
    glm::vec3 bestAxis(0.0f);
    bool separated = false;
    // Face axes win ties (natural for axis-aligned scenes).
    bool bestIsFace = false;

    // testFaceAxis: face axes are unit vectors a.orient[i] / b.orient[j], used
    // directly.
    auto testFaceAxis = [&](const glm::vec3 &axis, float ra, float rb)
    {
        if (separated)
            return;
        float sep = std::abs(glm::dot(d, axis));
        float pen = (ra + rb) - sep;
        if (pen < 0.0f)
        {
            separated = true;
            return;
        }
        if (pen < minPen)
        {
            minPen = pen;
            bestAxis = axis;
            bestIsFace = true;
        }
    };

    // testEdgeAxis: |cross(a.orient[i], b.orient[j])| may be << 1; normalize
    // before comparing pen so its scale matches the face axes (sep on a unit
    // vector). Degenerate (near-parallel) edge-crosses are skipped.
    auto testEdgeAxis = [&](int i, int j, const glm::vec3 &axisRaw, float ra, float rb)
    {
        if (separated)
            return;
        if (edgeDegenerate(i, j))
            return; // skip degenerate edge-cross so it cannot hijack minPen
        float len = glm::length(axisRaw);
        if (len < 1e-4f)
            return;
        glm::vec3 axis = axisRaw / len;
        // ra/rb are projection-radius sums derived from the non-normalized
        // axisRaw; divide by len to compare pen at face-axis scale.
        float raN = ra / len;
        float rbN = rb / len;
        float sep = std::abs(glm::dot(d, axis));
        float pen = (raN + rbN) - sep;
        if (pen < 0.0f)
        {
            separated = true;
            return;
        }
        // Edge axes replace only when pen is strictly below the face-held
        // minPen (face priority keeps float jitter from flipping bestAxis
        // between face and edge at equal pen).
        if (pen < minPen - 1e-4f)
        {
            minPen = pen;
            bestAxis = axis;
            bestIsFace = false;
        }
    };

    // Face axes (6).
    for (int i = 0; i < 3; ++i)
    {
        float ra = a.halfExtents[i];
        float rb = b.halfExtents[0] * AbsR[i][0] + b.halfExtents[1] * AbsR[i][1] + b.halfExtents[2] * AbsR[i][2];
        testFaceAxis(a.orientation[i], ra, rb);
    }
    for (int j = 0; j < 3; ++j)
    {
        float ra = a.halfExtents[0] * AbsR[0][j] + a.halfExtents[1] * AbsR[1][j] + a.halfExtents[2] * AbsR[2][j];
        float rb = b.halfExtents[j];
        testFaceAxis(b.orientation[j], ra, rb);
    }

    // Edge-cross axes (9).
    {
        glm::vec3 axis = glm::cross(a.orientation[0], b.orientation[0]);
        float ra = a.halfExtents[1] * AbsR[2][0] + a.halfExtents[2] * AbsR[1][0];
        float rb = b.halfExtents[1] * AbsR[0][2] + b.halfExtents[2] * AbsR[0][1];
        testEdgeAxis(0, 0, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[0], b.orientation[1]);
        float ra = a.halfExtents[1] * AbsR[2][1] + a.halfExtents[2] * AbsR[1][1];
        float rb = b.halfExtents[0] * AbsR[0][2] + b.halfExtents[2] * AbsR[0][0];
        testEdgeAxis(0, 1, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[0], b.orientation[2]);
        float ra = a.halfExtents[1] * AbsR[2][2] + a.halfExtents[2] * AbsR[1][2];
        float rb = b.halfExtents[0] * AbsR[0][1] + b.halfExtents[1] * AbsR[0][0];
        testEdgeAxis(0, 2, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[1], b.orientation[0]);
        float ra = a.halfExtents[0] * AbsR[2][0] + a.halfExtents[2] * AbsR[0][0];
        float rb = b.halfExtents[1] * AbsR[1][2] + b.halfExtents[2] * AbsR[1][1];
        testEdgeAxis(1, 0, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[1], b.orientation[1]);
        float ra = a.halfExtents[0] * AbsR[2][1] + a.halfExtents[2] * AbsR[0][1];
        float rb = b.halfExtents[0] * AbsR[1][2] + b.halfExtents[2] * AbsR[1][0];
        testEdgeAxis(1, 1, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[1], b.orientation[2]);
        float ra = a.halfExtents[0] * AbsR[2][2] + a.halfExtents[2] * AbsR[0][2];
        float rb = b.halfExtents[0] * AbsR[1][1] + b.halfExtents[1] * AbsR[1][0];
        testEdgeAxis(1, 2, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[2], b.orientation[0]);
        float ra = a.halfExtents[0] * AbsR[1][0] + a.halfExtents[1] * AbsR[0][0];
        float rb = b.halfExtents[1] * AbsR[2][2] + b.halfExtents[2] * AbsR[2][1];
        testEdgeAxis(2, 0, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[2], b.orientation[1]);
        float ra = a.halfExtents[0] * AbsR[1][1] + a.halfExtents[1] * AbsR[0][1];
        float rb = b.halfExtents[0] * AbsR[2][2] + b.halfExtents[2] * AbsR[2][0];
        testEdgeAxis(2, 1, axis, ra, rb);
    }
    {
        glm::vec3 axis = glm::cross(a.orientation[2], b.orientation[2]);
        float ra = a.halfExtents[0] * AbsR[1][2] + a.halfExtents[1] * AbsR[0][2];
        float rb = b.halfExtents[0] * AbsR[2][1] + b.halfExtents[1] * AbsR[2][0];
        testEdgeAxis(2, 2, axis, ra, rb);
    }

    if (separated || minPen <= 0.0f)
        return false;

    // Normal convention: A -> B.
    if (glm::dot(bestAxis, d) < 0.0f)
        bestAxis = -bestAxis;

    out.bodyA = idxA;
    out.bodyB = idxB;
    out.shapeA = shapeA;
    out.shapeB = shapeB;
    out.normal = bestAxis;
    out.pointCount = 0;

    // 2. Reference/incident face selection (standard Bullet/Box2D/Jolt):
    //   2a) the OBB whose orientation column is most aligned with bestAxis
    //       (largest |dot|) is the reference; its axis direction (sign chosen
    //       against refNormal) is the reference face normal.
    //   2b) among the incident box's 6 faces, pick the one whose normal best
    //       opposes refNormal; its 4 world vertices expand along the orient basis.
    //   2c) incident vertices with signed distance to the ref plane < 0 (into
    //       the ref side) become contact points (up to 4). Each worldPoint is
    //       the midpoint of the original vertex and its projection on the ref
    //       plane, keeping penetration/2 of depth, dimensionally consistent
    //       with the single-point path.
    auto maxAbsDotWithOrient = [](const OBB &obb, const glm::vec3 &dir) -> int
    {
        // Column index (0/1/2) with max |dot|; used for both refFaceAxis and
        // incFaceAxis.
        int best = 0;
        float m = std::abs(glm::dot(obb.orientation[0], dir));
        for (int k = 1; k < 3; ++k)
        {
            float v = std::abs(glm::dot(obb.orientation[k], dir));
            if (v > m)
            {
                m = v;
                best = k;
            }
        }
        return best;
    };

    const int aFaceIdx = maxAbsDotWithOrient(a, bestAxis);
    const int bFaceIdx = maxAbsDotWithOrient(b, bestAxis);
    const float aParallel = std::abs(glm::dot(a.orientation[aFaceIdx], bestAxis));
    const float bParallel = std::abs(glm::dot(b.orientation[bFaceIdx], bestAxis));
    const bool aIsRef = aParallel >= bParallel;

    const OBB &refBox = aIsRef ? a : b;
    const OBB &incBox = aIsRef ? b : a;
    const int refFaceIdx = aIsRef ? aFaceIdx : bFaceIdx;

    // refNormal points at the incident side: bestAxis is A->B, so when A is the
    // reference refNormal = +bestAxis.
    glm::vec3 refNormal = aIsRef ? bestAxis : -bestAxis;

    // Ref face plane: normal = refNormal, constant d = refNormal . refFaceCenter,
    // refFaceCenter = refBox.center + refBox.orient[refFaceIdx] * (+-halfExtents[refFaceIdx]),
    // the sign matching refNormal's direction.
    float refAxisSign = (glm::dot(refBox.orientation[refFaceIdx], refNormal) >= 0.0f) ? 1.0f : -1.0f;
    glm::vec3 refFaceCenter = refBox.center +
                              refAxisSign * refBox.halfExtents[refFaceIdx] *
                                  refBox.orientation[refFaceIdx];
    float refPlaneD = glm::dot(refNormal, refFaceCenter);

    // 2b. Find the incident face: among the inc box's 6 faces (3 orient axes x
    // +/-), the one whose normal has the largest dot with -refNormal (the face
    // most facing the ref box).
    int incFaceIdx = 0;
    float incFaceSign = 1.0f;
    float bestDot = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < 3; ++k)
    {
        // positive face normal = +orient[k]
        float dp = glm::dot(incBox.orientation[k], -refNormal);
        if (dp > bestDot)
        {
            bestDot = dp;
            incFaceIdx = k;
            incFaceSign = 1.0f;
        }
        // negative face normal = -orient[k]
        if (-dp > bestDot)
        {
            bestDot = -dp;
            incFaceIdx = k;
            incFaceSign = -1.0f;
        }
    }

    // Incident face center + the two in-plane tangent axes (the non-incFaceIdx
    // axes of inc box).
    glm::vec3 incFaceCenter = incBox.center +
                              incFaceSign * incBox.halfExtents[incFaceIdx] *
                                  incBox.orientation[incFaceIdx];
    int incT1 = (incFaceIdx + 1) % 3;
    int incT2 = (incFaceIdx + 2) % 3;
    glm::vec3 incE1 = incBox.halfExtents[incT1] * incBox.orientation[incT1];
    glm::vec3 incE2 = incBox.halfExtents[incT2] * incBox.orientation[incT2];

    // Incident 4 vertices (from +-incE1, +-incE2 combinations), the
    // Sutherland-Hodgman input polygon.
    glm::vec3 polyBuf0[8] = {
        incFaceCenter - incE1 - incE2,
        incFaceCenter + incE1 - incE2,
        incFaceCenter + incE1 + incE2,
        incFaceCenter - incE1 + incE2,
    };
    int polyCount0 = 4;

    // 2b'. Sutherland-Hodgman side clipping.
    //
    // Clip the incident face's 4-vertex polygon against the ref face's 4 side
    // planes in sequence. Each side plane:
    //   - planeNormal: unit vector pointing inward from the ref face (= -refT,
    //     back toward the ref face center)
    //   - planeOffset: dot(planeNormal, P) for P = side origin
    //     (refFaceCenter + refT*ht)
    //   `dot(planeNormal, v) - planeOffset >= 0` counts as inside (kept).
    //
    // ClipSegmentToLine handles one polygon edge [v0, v1] by in/out combos:
    //   - in, in   -> emit v1
    //   - in, out  -> emit the intersection p of v0-v1 with the plane
    //   - out, in  -> emit p, then v1
    //   - out, out -> emit nothing
    // Each side plane yields at most original vertex count + 1; after 4 planes
    // the polygon has at most 8 vertices.
    //
    // Clipping runs in world 3D, but every side plane is normal-inward on the
    // ref face plane, orthogonal to refNormal, so the vertices' refNormal
    // component (the penetration depth used by the signed-distance filter
    // below) is preserved.
    auto clipPolyByPlane = [](const glm::vec3 *inBuf, int inCount,
                              glm::vec3 *outBuf,
                              const glm::vec3 &planeN, float planeOffset) -> int
    {
        int outCount = 0;
        if (inCount == 0)
            return 0;
        glm::vec3 v0 = inBuf[inCount - 1];
        float d0 = glm::dot(planeN, v0) - planeOffset;
        for (int i = 0; i < inCount; ++i)
        {
            glm::vec3 v1 = inBuf[i];
            float d1 = glm::dot(planeN, v1) - planeOffset;
            const bool in0 = d0 >= 0.0f;
            const bool in1 = d1 >= 0.0f;
            if (in0 && in1)
            {
                outBuf[outCount++] = v1;
            }
            else if (in0 && !in1)
            {
                float t = d0 / (d0 - d1); // d0 - d1 != 0 guaranteed (otherwise both in or both out)
                outBuf[outCount++] = v0 + t * (v1 - v0);
            }
            else if (!in0 && in1)
            {
                float t = d0 / (d0 - d1);
                outBuf[outCount++] = v0 + t * (v1 - v0);
                outBuf[outCount++] = v1;
            }
            // out,out -> emit nothing
            v0 = v1;
            d0 = d1;
            if (outCount >= 8)
                break; // safety cap (8 = 4-vertex polygon expanded by 4 side planes at most)
        }
        return outCount;
    };

    // Build the ref face's 4 side planes and clip in sequence.
    int refT1 = (refFaceIdx + 1) % 3;
    int refT2 = (refFaceIdx + 2) % 3;
    float refHt1 = refBox.halfExtents[refT1];
    float refHt2 = refBox.halfExtents[refT2];
    glm::vec3 refAxis1 = refBox.orientation[refT1];
    glm::vec3 refAxis2 = refBox.orientation[refT2];

    glm::vec3 polyBuf1[8];
    int polyCount1 = 0;

    // +refAxis1 side (boundary at refC + refAxis1*ht1; inward = -refAxis1)
    polyCount1 = clipPolyByPlane(polyBuf0, polyCount0, polyBuf1,
                                 -refAxis1, glm::dot(-refAxis1, refFaceCenter + refAxis1 * refHt1));
    // -refAxis1 side
    polyCount0 = clipPolyByPlane(polyBuf1, polyCount1, polyBuf0,
                                 refAxis1, glm::dot(refAxis1, refFaceCenter - refAxis1 * refHt1));
    // +refAxis2 side
    polyCount1 = clipPolyByPlane(polyBuf0, polyCount0, polyBuf1,
                                 -refAxis2, glm::dot(-refAxis2, refFaceCenter + refAxis2 * refHt2));
    // -refAxis2 side
    polyCount0 = clipPolyByPlane(polyBuf1, polyCount1, polyBuf0,
                                 refAxis2, glm::dot(refAxis2, refFaceCenter - refAxis2 * refHt2));

    // Final clipped polygon lives in polyBuf0 with polyCount0 in [0, 8].

    // 2c. Vertex projection + penetration filter + manifold write.
    out.bodyA = idxA;
    out.bodyB = idxB;
    out.shapeA = shapeA;
    out.shapeB = shapeB;
    out.normal = bestAxis; // always A->B, separate from refNormal above
    out.pointCount = 0;

    // Filter clipped vertices by penetration; collect up to 4 (pointCount cap).
    // When more than 4 candidates, reduce to the 4 most spread points instead
    // of the first 4.
    //
    // Filter: vertices with sd > 0 (strictly outside the ref plane, not sunk)
    // are dropped. sd == 0 (exactly on the ref plane, zero penetration) is
    // kept as a contact point - the solver computes zero impulse there, so
    // dynamics are unaffected, but it stabilizes the frame's contact set and
    // next frame's persistent matching.
    //
    // Reduction (Box2D b2Collision + Bullet b3ReduceContacts):
    //   P0 = deepest point (max |sd|)
    //   P1 = point farthest from P0
    //   P2 = candidate maximizing triangle P0-P1-cand signed area
    //        (counter-clockwise triangle)
    //   P3 = candidate minimizing it (most negative; clockwise triangle)
    //   The 4 points envelope the largest area, symmetric, for stable support.
    struct Cand
    {
        glm::vec3 world;
        glm::vec3 localA;
        glm::vec3 localB;
        float penetration;
    };
    Cand cands[8];
    int candCount = 0;

    for (int vi = 0; vi < polyCount0; ++vi)
    {
        float sd = glm::dot(refNormal, polyBuf0[vi]) - refPlaneD;
        if (sd > 0.0f)
            continue;
        glm::vec3 onPlane = polyBuf0[vi] - refNormal * sd;
        glm::vec3 world = (polyBuf0[vi] + onPlane) * 0.5f;

        Cand &c = cands[candCount++];
        c.world = world;
        c.localA = glm::transpose(a.orientation) * (world - a.center);
        c.localB = glm::transpose(b.orientation) * (world - b.center);
        c.penetration = -sd;
        if (candCount >= 8)
            break;
    }

    auto pushCand = [&](const Cand &c)
    {
        ContactPoint cp;
        cp.worldPoint = c.world;
        cp.penetration = c.penetration;
        cp.localA = c.localA;
        cp.localB = c.localB;
        out.points[out.pointCount++] = cp;
    };

    if (candCount <= 4)
    {
        // Write all directly (no reduction needed at <= 4).
        for (int i = 0; i < candCount; ++i)
            pushCand(cands[i]);
    }
    else
    {
        // Reduce to 4 points.
        bool used[8] = {false, false, false, false, false, false, false, false};

        // P0: deepest penetration.
        int i0 = 0;
        for (int i = 1; i < candCount; ++i)
            if (cands[i].penetration > cands[i0].penetration)
                i0 = i;
        used[i0] = true;
        pushCand(cands[i0]);

        // P1: farthest from P0 (world-space squared Euclidean distance).
        int i1 = -1;
        float bestD = -1.0f;
        for (int i = 0; i < candCount; ++i)
        {
            if (used[i])
                continue;
            glm::vec3 d = cands[i].world - cands[i0].world;
            float d2 = glm::dot(d, d);
            if (d2 > bestD)
            {
                bestD = d2;
                i1 = i;
            }
        }
        if (i1 >= 0)
        {
            used[i1] = true;
            pushCand(cands[i1]);
        }

        // P2/P3: pick by max/min signed triangle area on the P0-P1 edge.
        //   signedArea(cand) = dot(refNormal, cross(P1-P0, cand-P0)) - the
        //   signed area projected onto the ref plane.
        if (i1 >= 0)
        {
            glm::vec3 e = cands[i1].world - cands[i0].world;
            int iPos = -1, iNeg = -1;
            float bestPos = 0.0f, bestNeg = 0.0f;
            for (int i = 0; i < candCount; ++i)
            {
                if (used[i])
                    continue;
                glm::vec3 f = cands[i].world - cands[i0].world;
                float area = glm::dot(refNormal, glm::cross(e, f));
                if (area > bestPos)
                {
                    bestPos = area;
                    iPos = i;
                }
                if (area < bestNeg)
                {
                    bestNeg = area;
                    iNeg = i;
                }
            }
            if (iPos >= 0)
            {
                used[iPos] = true;
                pushCand(cands[iPos]);
            }
            if (iNeg >= 0 && iNeg != iPos && out.pointCount < 4)
            {
                used[iNeg] = true;
                pushCand(cands[iNeg]);
            }
            // If one side has no candidate (all collinear), fill up to 4 with
            // any remaining points.
            for (int i = 0; i < candCount && out.pointCount < 4; ++i)
            {
                if (!used[i])
                {
                    used[i] = true;
                    pushCand(cands[i]);
                }
            }
        }
    }

    // Degenerate protection: no valid contact after clipping (SAT picked an
    // edge-cross axis, or clipping removed the whole incident face) -> fall
    // back to the incident's deepest support point (single-point path, keeping
    // the original behavior).
    if (out.pointCount == 0)
    {
        glm::vec3 incSupport = incBox.center;
        for (int i = 0; i < 3; ++i)
        {
            float s = glm::dot(incBox.orientation[i], -refNormal) > 0.0f ? 1.0f : -1.0f;
            incSupport += s * incBox.halfExtents[i] * incBox.orientation[i];
        }
        float sd = glm::dot(refNormal, incSupport) - refPlaneD;
        glm::vec3 onPlane = incSupport - refNormal * sd;
        glm::vec3 world = (incSupport + onPlane) * 0.5f;

        ContactPoint cp;
        cp.worldPoint = world;
        cp.penetration = minPen; // fall back to SAT's minPen
        cp.localA = glm::transpose(a.orientation) * (world - a.center);
        cp.localB = glm::transpose(b.orientation) * (world - b.center);
        out.points[0] = cp;
        out.pointCount = 1;
    }

    return true;
}

// mergeManifoldWithPersistent - match the fresh manifold against the old one
// by local-space coordinates.
// 1) find the old (bodyA, bodyB) manifold in manifolds_; if absent, push fresh.
// 2) if present: pair fresh.points with old points within a distance threshold
//    (localA + localB); matched new points inherit the old accumNormalImpulse /
//    accumTangentImpulse; unmatched old points drop. Threshold = 5% of the
//    smaller shape's min half-extent (empirical, no special cases).
// 3) overwrite the old slot with fresh (impulses inherited), keeping indices
//    stable.
void PhysicsWorld::mergeManifoldWithPersistent(ContactManifold &fresh)
{
    // Threshold: 5% of the smaller shape's min half-extent, floor 0.005m
    // (generic).
    float minHalfA = std::min(shapes_[fresh.shapeA].halfExtents.x,
                              std::min(shapes_[fresh.shapeA].halfExtents.y, shapes_[fresh.shapeA].halfExtents.z));
    float minHalfB = std::min(shapes_[fresh.shapeB].halfExtents.x,
                              std::min(shapes_[fresh.shapeB].halfExtents.y, shapes_[fresh.shapeB].halfExtents.z));
    float matchThresh = std::max(0.005f, 0.05f * std::min(minHalfA, minHalfB));
    float matchThreshSq = matchThresh * matchThresh;

    int oldIdx = -1;
    for (size_t i = 0; i < manifolds_.size(); ++i)
    {
        if (manifolds_[i].bodyA == fresh.bodyA && manifolds_[i].bodyB == fresh.bodyB)
        {
            oldIdx = static_cast<int>(i);
            break;
        }
    }

    if (oldIdx < 0)
    {
        fresh.alive = true;
        manifolds_.push_back(fresh);
        return;
    }

    ContactManifold &old = manifolds_[oldIdx];

    // For each new point, find the nearest old point (localA distance + localB
    // distance both below the threshold squared).
    //
    // History: the single-point manifold had a face-face contact flaw -
    // `supportPoint(box, -Y)` chose among the bottom 4 corners by float dot
    // sign, jittering between +-0.5 each frame, so localA/localB distances
    // across frames approached the diagonal (far above matchThresh) and warm
    // start failed. The multi-point manifold fixed this: 4 stable corners each
    // frame, no cross-frame drift. This matching loop is agnostic to point
    // count and needs no change.
    for (int i = 0; i < fresh.pointCount; ++i)
    {
        auto &np = fresh.points[i];
        int bestJ = -1;
        float bestScore = std::numeric_limits<float>::max();
        for (int j = 0; j < old.pointCount; ++j)
        {
            auto &op = old.points[j];
            if (op.matched)
                continue;
            float dA = glm::dot(np.localA - op.localA, np.localA - op.localA);
            float dB = glm::dot(np.localB - op.localB, np.localB - op.localB);
            if (dA > matchThreshSq || dB > matchThreshSq)
                continue;
            float score = dA + dB;
            if (score < bestScore)
            {
                bestScore = score;
                bestJ = j;
            }
        }
        if (bestJ >= 0)
        {
            np.accumNormalImpulse = old.points[bestJ].accumNormalImpulse;
            np.accumTangentImpulse[0] = old.points[bestJ].accumTangentImpulse[0];
            np.accumTangentImpulse[1] = old.points[bestJ].accumTangentImpulse[1];
            old.points[bestJ].matched = true;
        }
    }

    fresh.alive = true;
    // Overwrite the old manifold: bodyA/bodyB already match; new
    // normal/shape/points replace.
    old = fresh;
}

// Island construction + union-find helpers. buildIslands() runs once at the
// end of each detectCollisions: build connected components of the contact
// graph from alive && !trigger manifolds and assign body/manifold indices into
// islands_[].
//
// Union-find conventions:
//   - path compression: every node on the find path attaches directly to root
//   - union by rank: lower rank hangs under higher rank
//   - amortized O(alpha(N)); constant for physics scale (N <= 1e4)
// Static bodies join the union-find normally (no special case); the
// island-sleep accumulation filters by isStatic later,
// so statics connecting islands lets two Dynamics share one island through a
// Static floor.
int PhysicsWorld::unionFind(int i)
{
    // Path compression until parent[i] == i.
    while (unionParent_[i] != i)
    {
        unionParent_[i] = unionParent_[unionParent_[i]]; // half path compression
        i = unionParent_[i];
    }
    return i;
}

void PhysicsWorld::unionUnite(int i, int j)
{
    int ri = unionFind(i);
    int rj = unionFind(j);
    if (ri == rj)
        return;
    // Union by rank: lower rank hangs under higher rank.
    if (unionRank_[ri] < unionRank_[rj])
        std::swap(ri, rj);
    unionParent_[rj] = ri;
    if (unionRank_[ri] == unionRank_[rj])
        ++unionRank_[ri];
}

void PhysicsWorld::buildIslands()
{
    const int N = static_cast<int>(bodies_.size());

    // 1. Init union-find (reflexive).
    unionParent_.assign(N, 0);
    unionRank_.assign(N, 0);
    for (int i = 0; i < N; ++i)
        unionParent_[i] = i;

    // 2. Union bodyA/bodyB across alive non-trigger manifolds.
    for (const auto &m : manifolds_)
    {
        if (!m.alive || m.isTrigger)
            continue;
        if (m.bodyA < 0 || m.bodyB < 0)
            continue;
        if (m.bodyA >= N || m.bodyB >= N)
            continue; // defensive
        unionUnite(m.bodyA, m.bodyB);
    }

    // 3. Assign islandId: each active body's root is its island representative;
    //    rootToIslandIdx maps root body indices to islands_ indices.
    islandId_.assign(N, -1);
    islands_.clear();
    std::vector<int> rootToIslandIdx(N, -1);
    for (int i = 0; i < N; ++i)
    {
        if (!active_[i])
            continue;
        int r = unionFind(i);
        int idx = rootToIslandIdx[r];
        if (idx < 0)
        {
            idx = static_cast<int>(islands_.size());
            rootToIslandIdx[r] = idx;
            islands_.emplace_back();
        }
        islandId_[i] = idx;
        islands_[idx].bodies.push_back(i);
    }

    // 4. Assign alive non-trigger manifolds to their island.
    for (int mi = 0; mi < static_cast<int>(manifolds_.size()); ++mi)
    {
        const auto &m = manifolds_[mi];
        if (!m.alive || m.isTrigger)
            continue;
        if (m.bodyA < 0 || m.bodyB < 0)
            continue;
        int idA = islandId_[m.bodyA];
        int idB = islandId_[m.bodyB];
        if (idA < 0 || idA != idB)
            continue; // defensive: both ends must share one island
        islands_[idA].manifoldIndices.push_back(mi);
    }

    // 5. Init island.sleepFrames as the MINIMUM over the island's Dynamic
    //    bodies. Persistence: body.sleepFrames is the single source of truth;
    //    resolveCollisions writes the final value back each frame, and the
    //    rebuilt island reads it back. Min semantics: when membership changes
    //    (a new body joins), the newcomer's fresh sleepFrames dilutes the
    //    island's accumulation, so a colliding newcomer cannot put the whole
    //    island straight to sleep.
    for (auto &isl : islands_)
    {
        int minFrames = std::numeric_limits<int>::max();
        bool anyDynamic = false;
        for (int bi : isl.bodies)
        {
            if (bodies_[bi].isStatic())
                continue;
            anyDynamic = true;
            if (bodies_[bi].sleepFrames < minFrames)
                minFrames = bodies_[bi].sleepFrames;
        }
        isl.sleepFrames = anyDynamic ? minFrames : 0;
    }

    // 6. Stats: total islands / islands with at least one non-sleeping body.
    stats_.islandCount = static_cast<int>(islands_.size());
    int activeIslands = 0;
    for (const auto &isl : islands_)
    {
        bool hasActive = false;
        for (int bi : isl.bodies)
        {
            if (!bodies_[bi].isStatic() && !bodies_[bi].sleeping)
            {
                hasActive = true;
                break;
            }
        }
        if (hasActive)
            ++activeIslands;
    }
    stats_.activeIslandCount = activeIslands;
}

// Static-stability criterion, the 4th necessary condition for sleep.
// ----------------------------------------------------------------------------
// Project all contact points touching this Dynamic body onto the gravity-
// perpendicular plane, build a 2D hull, and check the COM projection lies
// inside it.
//
//   - zero gravity (|gravity| < 1e-4) -> always stable (return true)
//   - single contact -> COM projection within eps of the point
//   - two contacts -> COM projection within eps of the segment
//   - many contacts -> generic hull containment
//
// eps = 5% of the contact-to-COM 2D distance + absolute floor of 1mm - a
// generic scale, independent of shape type.
bool PhysicsWorld::isStaticallyStable(int bodyIndex,
                                      const std::vector<int> &manifoldIndices) const
{
    const float gMag2 = glm::dot(gravity, gravity);
    if (gMag2 < 1e-8f)
        return true; // zero gravity: no tipping torque, always stable

    const glm::vec3 gHat = gravity / std::sqrt(gMag2);

    // Build a 2D basis (u, v) perpendicular to gravity (Erin Catto's stable
    // orthonormal-basis method).
    glm::vec3 u, v;
    if (std::abs(gHat.x) >= 0.57735f)
        u = glm::vec3(gHat.y, -gHat.x, 0.0f);
    else
        u = glm::vec3(0.0f, gHat.z, -gHat.y);
    u = glm::normalize(u);
    v = glm::cross(gHat, u);

    const glm::vec3 com = poses_[bodyIndex].position;

    // Collect every contact point's 2D projection (relative to COM).
    std::vector<glm::vec2> pts2d;
    pts2d.reserve(16);
    float maxRadius = 0.0f;
    for (int mi : manifoldIndices)
    {
        if (mi < 0 || mi >= (int)manifolds_.size())
            continue;
        const auto &m = manifolds_[mi];
        if (!m.alive || m.isTrigger)
            continue;
        if (m.bodyA != bodyIndex && m.bodyB != bodyIndex)
            continue;
        for (int k = 0; k < m.pointCount; ++k)
        {
            glm::vec3 d = m.points[k].worldPoint - com;
            // Drop the gravity-direction component; keep the perpendicular 2D
            // coordinates.
            float du = glm::dot(d, u);
            float dv = glm::dot(d, v);
            pts2d.emplace_back(du, dv);
            float r = std::sqrt(du * du + dv * dv);
            if (r > maxRadius)
                maxRadius = r;
        }
    }

    if (pts2d.empty())
        return false; // no contact -> unstable (sleep should not trigger here anyway; defensive)

    // Stability margin: the COM projection must not just be inside the hull -
    // it must sit > margin from the hull edge. Ideal symmetric edge/corner
    // balance (the margin == 0 boundary case) therefore counts as unstable,
    // matching reality where any perturbation tips it over.
    //
    // margin = max(1cm absolute floor, 2% of max contact radius). The floor
    // keeps small bodies from getting a too-tight threshold; the relative term
    // lets large bodies scale up. Measured data:
    //   - flat resting / stacks: margin >= 0.49m (ample)
    //   - symmetric edge balance: margin == 0 (must reject)
    // The 1cm floor separates both cases cleanly.
    const float stabilityMargin = std::max(0.01f, 0.02f * maxRadius);

    // Build the 2D hull.
    auto hull = geometry2d::convexHull2D(pts2d);

    // Require the origin (COM projection) to stay stabilityMargin inside the
    // hull. A negative eps to pointInConvexPolygon2D means "signed distance to
    // every edge >= stabilityMargin" - the hull shrunk by stabilityMargin must
    // contain the origin.
    return geometry2d::pointInConvexPolygon2D(hull, glm::vec2(0.0f, 0.0f), -stabilityMargin);
}

bool PhysicsWorld::isStaticallyStableIsland(const Island &island) const
{
    // Every Dynamic body in the island must be statically stable for the
    // island to sleep.
    for (int bi : island.bodies)
    {
        if (bodies_[bi].isStatic())
            continue;
        if (!isStaticallyStable(bi, island.manifoldIndices))
            return false;
    }
    return true;
}

// buildTangentBasis - Erin Catto's stable orthonormal tangent basis. For a
// unit normal, emit two unit vectors t1/t2 orthogonal to it and to each other;
// avoids cross(normal, arbitrary) degenerating when arbitrary is parallel to
// the normal.
void PhysicsWorld::buildTangentBasis(const glm::vec3 &normal, glm::vec3 &t1, glm::vec3 &t2)
{
    if (std::abs(normal.x) >= 0.57735f)
        t1 = glm::vec3(normal.y, -normal.x, 0.0f);
    else
        t1 = glm::vec3(0.0f, normal.z, -normal.y);
    t1 = glm::normalize(t1);
    t2 = glm::cross(normal, t1);
}

// warmStartManifolds - apply last frame's accumulated impulses at once (warm
// start). Applied only to this frame's alive manifolds with at least one
// non-static, not-both-sleeping pair. Tangent bases refresh first (the normal
// may have changed with narrowphase re-solve).
// Do NOT overwrite cp.penetration/cp.worldPoint here - narrowphase wrote the
// authoritative clipping depth in computeBoxBoxManifold; warm start only
// applies last frame's (Jn, Jt1, Jt2) to current velocity/angular velocity.
// (Historical bug: overwriting penetration with dot(worldA - worldB, normal)
// zeroed freshly generated first-frame penetration, so the split-impulse
// position solver never pushed and boxes sank into each other.)
void PhysicsWorld::warmStartManifolds()
{
    // With warmStartEnabled=false, zero every alive manifold's accumulated
    // impulse. Cannot early-return: ContactConstraint::warmStart also rebuilds
    // the tangent basis each frame (buildTangentBasis), which decouples the
    // tangents from the narrowphase normal and must run every frame. Zeroed
    // impulses applied as 0 are a no-op, but the basis still refreshes -
    // matching Bullet/Box2D's `solverInfo.m_warmstartingFactor=0` semantics.
    if (!warmStartEnabled)
    {
        for (auto &m : manifolds_)
        {
            if (!m.alive)
                continue;
            for (int k = 0; k < m.pointCount; ++k)
            {
                m.points[k].accumNormalImpulse = 0.0f;
                m.points[k].accumTangentImpulse[0] = 0.0f;
                m.points[k].accumTangentImpulse[1] = 0.0f;
            }
        }
    }

    // Traverse manifolds grouped by island. Numerically equivalent: every alive
    // non-trigger manifold belongs to exactly one island (guaranteed by
    // buildIslands), so the visited manifold set equals the old
    // `for (auto& m : manifolds_)`, only the order is island-grouped.
    //
    // Per-manifold work moved into ContactConstraint::warmStart; this function
    // only provides the island-scoped outer grouping.
    //
    // CP-1.2: the former `else` fallback (linear scan of contactConstraints_
    // when islands_ is empty) could never fire - buildIslands() is only called
    // inside detectCollisions before contactConstraints_ is rebuilt, so an
    // empty islands_ implies an empty contactConstraints_ and both paths are
    // no-ops. The `if (!islands_.empty())` wrapper was removed as noise;
    // islands_ may still be empty (e.g. detectCollisions never ran), in which
    // case the loop naturally iterates zero times.
    for (const auto &isl : islands_)
    {
        for (int mi : isl.manifoldIndices)
        {
            if (mi < 0 || mi >= (int)manifolds_.size())
                continue;
            // Invariant (see the assert at the end of detectCollisions):
            // contactConstraints_.size() == manifolds_.size() && all non-null.
            contactConstraints_[mi]->warmStart(*this);
        }
    }
}

// resolveCollisions - manifold + warm start + two-tangent friction cone.
// ----------------------------------------------------------------------------
// Flow (based on manifolds_, no longer consuming contacts_ directly):
//   1) prepare: per manifold point compute rA/rB/denominator/restitutionTarget;
//      tangent bases from buildTangentBasis; both-static / both-sleeping pairs
//      get skip.
//   2) warm start: apply inherited (Jn, Jt1, Jt2).
//   3) iterative solver (solverIterations rounds):
//        - normal: dJn = (restitutionTarget - velAlongNormal) / denomN,
//          accumulated and clamped >= 0;
//        - two tangents: dJt1/dJt2 per axis, accumulated then disk-projected
//          with |J_t| <= mu*Jn.
//   4) write each point's latest worldPoint + penetration back into contacts_
//      (for the split-impulse position solver).
//   5) resting/rolling friction (no -gravity angle test, no heuristic params).
//   6) sleep frame accumulation.
void PhysicsWorld::resolveCollisions(float /*dt*/)
{
    if (manifolds_.empty())
        return;

    const int solverIterations = 10;
    // CP-2.1 / CP-2.2: these 8 thresholds were promoted to PhysicsWorld public
    // members (restingNormal/Tangent/Angular/CenterLockThreshold,
    // sleepLinear/AngularThreshold, sleepFramesRequired, wakeImpactThreshold);
    // accessed by name here so the Editor UI / unit tests can tune them.

    // Reuse PhysicsWorld's scratch members to avoid per-frame heap allocation.
    // assign = clear -> resize(N) -> fill(init); no malloc when capacity
    // suffices.
    auto &hadContact = scratchHadContact_;
    auto &supportNormals = scratchSupportNormals_;
    auto &supportPoints = scratchSupportPoints_;
    auto &supportCounts = scratchSupportCounts_;
    // Accumulated tangential support velocity (opposite body's world velocity
    // at the contact point); applyRestingStaticFriction locks resting to
    // vSurface instead of 0 - the core mechanism letting an upper body follow
    // a moving Kinematic base.
    auto &supportVelocities = scratchSupportVelocities_;
    hadContact.assign(bodies_.size(), false);
    supportNormals.assign(bodies_.size(), glm::vec3(0.0f));
    supportPoints.assign(bodies_.size(), glm::vec3(0.0f));
    supportCounts.assign(bodies_.size(), 0);
    supportVelocities.assign(bodies_.size(), glm::vec3(0.0f));

    // Prepare + warm start: refresh each ContactPoint's world pose / tangent
    // basis / apply inherited impulses.
    warmStartManifolds();

    // The old PointCtx / ctxs[mi][k] is replaced by
    // ContactConstraint::pointCache_; SI iterations read the same data via
    // contactConstraints_[mi]->point(k).

    // Prepare + wake-up decisions. The per-point prep (rA/rB/denom/
    // restitutionTarget) lives in ContactConstraint::prepare, which also
    // writes the skip flags into pointCache_[k].skip; this loop keeps only the
    // body-level side effects: wake-up + hadContact marks.
    for (size_t mi = 0; mi < manifolds_.size(); ++mi)
    {
        contactConstraints_[mi]->prepare(*this, 0.0f);

        auto &m = manifolds_[mi];
        if (!m.alive || m.isTrigger || m.pointCount <= 0)
            continue;
        auto &bodyA = bodies_[m.bodyA];
        auto &bodyB = bodies_[m.bodyB];
        bool bothStatic = bodyA.isStatic() && bodyB.isStatic();
        bool bothSleeping = bodyA.sleeping && bodyB.sleeping;
        if (bothStatic || bothSleeping)
            continue;
        float invMassA = bodyA.inverseMass();
        float invMassB = bodyB.inverseMass();
        if (invMassA + invMassB <= 0.0f)
            continue;

        // Neighbor wake: judge by the manifold's max initial impact (computed
        // in prepare).
        auto *cc = static_cast<ContactConstraint *>(contactConstraints_[mi].get());
        float maxInitialImpact = cc->maxInitialImpact();

        // Neighbor-wake condition extended:
        //   Old: only "opposite body is Kinematic && |v_at_cp| > threshold".
        //   Problem: Dynamic-Dynamic layers (stack with dragged bottom) never
        //   matched, so a sleeping upper Dynamic could not be woken by the
        //   tangential motion of an awake lower Dynamic.
        //   New (generic): "opposite body not sleeping && |v_at_cp| > threshold"
        //     * Kinematic with velocity -> hits (matches old behavior)
        //     * awake Dynamic with velocity -> hits (new; fixes layer wake)
        //     * Static velocity is always 0 -> never hits (Static semantics
        //       intact)
        //     * two sleeping Dynamics -> never hits (stack sleep convergence
        //       unaffected; guarded by StackSleepConvergenceUnaffected)
        //   Generic condition (only !sleeping + velocity magnitude), no
        //   per-bodyType/per-scene special cases.
        bool wakeByNeighborMotion = false;
        if (m.pointCount > 0)
        {
            const glm::vec3 &cp = m.points[0].worldPoint;
            if (!bodyA.sleeping)
            {
                glm::vec3 vAtCpA = bodyA.velocity + glm::cross(bodyA.angularVelocity, cp - poses_[m.bodyA].position);
                if (glm::length(vAtCpA) > sleepLinearThreshold)
                    wakeByNeighborMotion = true;
            }
            if (!wakeByNeighborMotion && !bodyB.sleeping)
            {
                glm::vec3 vAtCpB = bodyB.velocity + glm::cross(bodyB.angularVelocity, cp - poses_[m.bodyB].position);
                if (glm::length(vAtCpB) > sleepLinearThreshold)
                    wakeByNeighborMotion = true;
            }
        }

        if (maxInitialImpact > wakeImpactThreshold || wakeByNeighborMotion)
        {
            if (bodyA.sleeping)
            {
                bodyA.sleeping = false;
                bodyA.sleepFrames = 0;
            }
            if (bodyB.sleeping)
            {
                bodyB.sleeping = false;
                bodyB.sleepFrames = 0;
            }
        }

        hadContact[m.bodyA] = true;
        hadContact[m.bodyB] = true;
    }

    // SI iteration: normal + two tangents (friction disk). Per-manifold body
    // moved into ContactConstraint::solveVelocity; the outer layer keeps only
    // the solverIterations loop + manifold traversal (original order preserved,
    // numerically equivalent).
    for (int iter = 0; iter < solverIterations; ++iter)
    {
        // CP-1.3: the old `mi < size && ptr` double guard is always true - see
        // the invariant assert at the end of detectCollisions
        // (contactConstraints_.size() == manifolds_.size() && non-null).
        for (size_t mi = 0; mi < manifolds_.size(); ++mi)
            contactConstraints_[mi]->solveVelocity(*this);
    }

    // contacts_ was expanded once from alive manifolds in detectCollisions,
    // with penetration/worldPoint/normal/shapeIndex being narrowphase's
    // authoritative values; resolveCollisions does NOT write contacts_ back.
    //   - latest accumulated impulses live in
    //     manifolds_[mi].points[k].accumNormalImpulse / accumTangentImpulse,
    //     consumed by the persistent-manifold warm start; external consumers
    //     (tests / Inspector) read getManifolds(), not the contacts_ snapshot.
    //     (Before CP-3.1, contacts_ carried 3 snapshot fields like
    //     accumulatedNormalImpulse written once at detection and never updated;
    //     removed in CP-3.1.)
    //   - the position solver reads contacts_[ci].penetration - the true
    //     geometric depth written at detection; resolve need not modify it.

    // Support stats + rolling friction (iterating contacts_, already expanded
    // from manifolds).
    for (auto &c : contacts_)
    {
        auto &bodyA = bodies_[c.indexA];
        auto &bodyB = bodies_[c.indexB];
        float invMassA = bodyA.inverseMass();
        float invMassB = bodyB.inverseMass();
        if (invMassA + invMassB <= 0.0f)
            continue;

        if (!bodyA.isStatic())
        {
            supportNormals[c.indexA] -= c.normal;
            supportPoints[c.indexA] += c.contactPoint;
            supportCounts[c.indexA]++;
            // Accumulate the opposite body (B)'s world velocity at the contact
            // point: v_at_point(B) = bodyB.velocity + angVel_B x (cp - posB).
            // Works for Static/Kinematic too: Static speed = 0, Kinematic
            // comes from pose deltas.
            const glm::vec3 rB = c.contactPoint - poses_[c.indexB].position;
            supportVelocities[c.indexA] += bodyB.velocity + glm::cross(bodyB.angularVelocity, rB);
        }
        if (!bodyB.isStatic())
        {
            supportNormals[c.indexB] += c.normal;
            supportPoints[c.indexB] += c.contactPoint;
            supportCounts[c.indexB]++;
            // Symmetric: accumulate the opposite body (A)'s contact-point world
            // velocity.
            const glm::vec3 rA = c.contactPoint - poses_[c.indexA].position;
            supportVelocities[c.indexB] += bodyA.velocity + glm::cross(bodyA.angularVelocity, rA);
        }

        // rollingFriction comes straight from material.rollingFriction
        // (combined with max) - no more `mu * 0.3f` fallback and no
        // restingCenterLockThreshold heuristic branch. Set rollingFriction
        // explicitly in PhysicsMaterial to get rolling resistance.
        CombinedMaterial cmRoll = combineMaterial(shapeMaterial_[c.shapeIndexA], shapeMaterial_[c.shapeIndexB]);
        float rollingMu = cmRoll.rollingFriction;
        if (rollingMu <= 0.0f)
            continue;

        auto applyRollingFriction = [&](RigidBody &body, int /*bodyIndex*/)
        {
            if (body.isStatic() || body.sleeping)
                return;
            float angSpeed = glm::length(body.angularVelocity);
            if (angSpeed < 1e-6f)
                return;

            glm::vec3 wHat = body.angularVelocity / angSpeed;
            glm::vec3 invIw = body.inverseInertiaWorld * wHat;
            float invIalong = glm::dot(wHat, invIw);
            float Ialong = (invIalong > 1e-8f) ? (1.0f / invIalong) : 1e-6f;

            // Generic rolling resistance: reduction =
            // rollingMu * m * g * dt / I_along_omega, clamped to the current
            // angular speed (never reverses rotation).
            float frictionTorqueMag = rollingMu * body.mass * glm::length(gravity) * fixedTimeStep;
            float reduction = std::min(frictionTorqueMag / Ialong, angSpeed);
            body.angularVelocity -= wHat * reduction;
        };

        applyRollingFriction(bodyA, c.indexA);
        applyRollingFriction(bodyB, c.indexB);
    }

    // Resting friction (the -gravity angle gate was removed; generic support
    // normal only).
    auto applyRestingStaticFriction = [&](int bodyIndex)
    {
        if (!active_[bodyIndex])
            return;
        auto &body = bodies_[bodyIndex];
        if (body.isStatic() || body.sleeping)
            return;
        if (supportCounts[bodyIndex] <= 0)
            return;

        // Statically unstable pose -> skip resting lock. Resting lock zeroes
        // sub-threshold tangent/angular velocity; on an unstable tilted pose
        // (edge/corner balance) it keeps clipping the angular velocity gravity
        // torque just built up, producing a slow-motion tip-over (measured
        // 1.7s to fall from edge to face). As with sleep, isStaticallyStable
        // is the unified necessary check: only stable poses may zero motion;
        // unstable poses must keep it so gravity torque acts freely.
        {
            int islandIdx = (bodyIndex < static_cast<int>(islandId_.size()))
                                ? islandId_[bodyIndex]
                                : -1;
            if (islandIdx >= 0 && islandIdx < static_cast<int>(islands_.size()))
            {
                if (!isStaticallyStable(bodyIndex, islands_[islandIdx].manifoldIndices))
                    return;
            }
        }

        glm::vec3 supportNormal = supportNormals[bodyIndex];
        float supportNormalLen = glm::length(supportNormal);
        if (supportNormalLen < 1e-6f)
            return;
        supportNormal /= supportNormalLen;
        // No `dot(supportNormal, -gravity) < 0.85` direction special case:
        // resting-lock triggers on "contact exists + speed small enough"
        // generically, so slopes / rotating platforms / zero-gravity scenes
        // all rest correctly.

        glm::vec3 avgSupportPoint = supportPoints[bodyIndex] / static_cast<float>(supportCounts[bodyIndex]);
        glm::vec3 supportOffset = avgSupportPoint - poses_[bodyIndex].position;
        glm::vec3 supportLeverVec = supportOffset - glm::dot(supportOffset, supportNormal) * supportNormal;
        float supportLever = glm::length(supportLeverVec);

        // Average support-surface tangential velocity (mean of the opposite
        // body's world velocity at contact points).
        //   - vSurface ~ 0 (static base / Static floor) -> classic "lock
        //     tangent to 0" branch, behavior identical to before, preserving
        //     stack sleep convergence.
        //   - nonzero vSurface (moving Kinematic base) -> skip tangent lock
        //     entirely and let the friction disk (mu*Jn clamp) in
        //     ContactConstraint::solveVelocity push the body - mu physically
        //     controls "follow vs slip":
        //       * high mu -> larger tangent impulse cap, pulls v toward
        //         vSurface each substep -> follows
        //       * low mu -> smaller cap, small pull per substep -> slips
        //   Generic: branch only on vSurface magnitude, no bodyType/scene
        //   dependence.
        glm::vec3 vSurface = supportVelocities[bodyIndex] /
                             static_cast<float>(supportCounts[bodyIndex]);
        float vSurfaceSpeed = glm::length(vSurface);

        float normalSpeed = glm::dot(body.velocity, supportNormal);
        if (normalSpeed > restingNormalLockThreshold)
            return;

        glm::vec3 tangentVel = body.velocity - normalSpeed * supportNormal;
        float tangentSpeed = glm::length(tangentVel);

        // Generic rule: a base with tangential motion -> drop tangent/angular
        // lock and hand friction back to the solver's disk clamp (mu*Jn).
        // Threshold is sleepLinearThreshold, aligned with the sleep criterion:
        // once the base moves at "would keep the other awake" speed, do not
        // preempt the solver here.
        if (vSurfaceSpeed > sleepLinearThreshold)
            return;

        if (tangentSpeed < restingTangentLockThreshold)
        {
            body.velocity -= tangentVel;
            if (supportLever < restingCenterLockThreshold &&
                std::abs(normalSpeed) < restingNormalLockThreshold)
            {
                body.velocity -= normalSpeed * supportNormal;
            }
        }

        if (supportLever < restingCenterLockThreshold &&
            tangentSpeed < sleepLinearThreshold &&
            glm::length(body.angularVelocity) < restingAngularLockThreshold)
        {
            body.angularVelocity = glm::vec3(0.0f);
        }
    };

    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        applyRestingStaticFriction(i);
    }

    // Sleep accumulation (island-level + impulse-change criterion):
    // 1) total normal impulse each Dynamic body received this frame (over all
    //    alive manifolds touching it) for the "contact-force change <
    //    threshold" test - a truly resting stack keeps this nearly constant
    //    across frames. Scratch members avoid per-frame allocation (equivalent
    //    to the former std::vector<float>(N, 0)).
    auto &thisFrameNormalImpulse = scratchThisFrameNormalImpulse_;
    thisFrameNormalImpulse.assign(bodies_.size(), 0.0f);
    for (const auto &m : manifolds_)
    {
        if (!m.alive || m.isTrigger)
            continue;
        if (m.bodyA < 0 || m.bodyB < 0)
            continue;
        float sum = 0.0f;
        for (int k = 0; k < m.pointCount; ++k)
            sum += m.points[k].accumNormalImpulse;
        // Both Dynamic ends accumulate; Static does not (it never participates
        // in sleep).
        if (!bodies_[m.bodyA].isStatic())
            thisFrameNormalImpulse[m.bodyA] += sum;
        if (!bodies_[m.bodyB].isStatic())
            thisFrameNormalImpulse[m.bodyB] += sum;
    }

    // 2) Island-level sleep (iterate this frame's islands_; each Dynamic island
    //    accumulates independently). Criteria - every Dynamic body in the
    //    island must satisfy:
    //   (a) linearSpeed  < sleepLinearThreshold
    //   (b) angularSpeed < sleepAngularThreshold
    //   (c) |thisFrame - lastFrame normal impulse| < impulseDeltaThreshold
    //    All three -> island.sleepFrames++; any failure resets to 0 and wakes
    //    every Dynamic body of the island.
    //
    // impulseDeltaThreshold ~ 10% of body.mass * gravity * dt - a fraction of
    // the one-frame impulse needed to hold gravity; a smaller two-frame impulse
    // delta means the contact-force scale is stable (not still bouncing).
    const float gMag = glm::length(gravity);
    const float impulseDeltaBaseScale = 0.1f * fixedTimeStep * gMag;
    for (auto &isl : islands_)
    {
        // Collect the island's Dynamic bodies (Static stays as a connection
        // node but never judged).
        bool anyDynamic = false;
        bool allLow = true;
        float maxImpulseDelta = 0.0f;
        for (int bi : isl.bodies)
        {
            const auto &body = bodies_[bi];
            if (body.isStatic())
                continue;
            anyDynamic = true;
            float linearSpeed = glm::length(body.velocity);
            float angularSpeed = glm::length(body.angularVelocity);
            if (linearSpeed >= sleepLinearThreshold || angularSpeed >= sleepAngularThreshold)
            {
                allLow = false;
                break;
            }
            float impulseDelta = std::fabs(thisFrameNormalImpulse[bi] - body.lastFrameNormalImpulse);
            // Threshold scales with body mass (generic: mass determines the
            // one-frame gravity impulse).
            float impulseDeltaThreshold = std::max(body.mass * impulseDeltaBaseScale, 1e-4f);
            if (impulseDelta >= impulseDeltaThreshold)
            {
                allLow = false;
                break;
            }
            if (impulseDelta > maxImpulseDelta)
                maxImpulseDelta = impulseDelta;
        }

        if (!anyDynamic)
        {
            // All-Static island: no sleep accumulation needed; zero
            // sleepFrames defensively.
            isl.sleepFrames = 0;
            continue;
        }

        // 4th necessary condition - static stability: every Dynamic body's COM
        // projected along gravity must fall inside its contact support polygon;
        // otherwise the pose tips over under gravity torque even with
        // near-zero instantaneous velocity/impulse change, and sleep is
        // refused. Needed because velocity/impulse-only criteria misjudge
        // "balanced on an edge/corner".
        if (allLow && isStaticallyStableIsland(isl))
        {
            ++isl.sleepFrames;
            if (isl.sleepFrames >= sleepFramesRequired)
            {
                // Island sleeps: all Dynamic bodies set sleeping together,
                // velocities zeroed.
                for (int bi : isl.bodies)
                {
                    auto &body = bodies_[bi];
                    if (body.isStatic())
                        continue;
                    body.sleeping = true;
                    body.sleepFrames = sleepFramesRequired; // compatibility field
                    body.velocity = glm::vec3(0.0f);
                    body.angularVelocity = glm::vec3(0.0f);
                }
            }
            else
            {
                // Not at the threshold yet: sync body-level sleepFrames
                // (compatibility field).
                for (int bi : isl.bodies)
                {
                    if (bodies_[bi].isStatic())
                        continue;
                    bodies_[bi].sleepFrames = isl.sleepFrames;
                }
            }
        }
        else
        {
            // Any criterion failed: reset island sleepFrames and wake sleeping
            // Dynamic bodies.
            isl.sleepFrames = 0;
            for (int bi : isl.bodies)
            {
                auto &body = bodies_[bi];
                if (body.isStatic())
                    continue;
                body.sleepFrames = 0;
                // Wake only when speed/angular speed are clearly above
                // threshold (avoids threshold-edge jitter).
                float linearSpeed = glm::length(body.velocity);
                float angularSpeed = glm::length(body.angularVelocity);
                if (body.sleeping &&
                    (linearSpeed > sleepLinearThreshold ||
                     angularSpeed > sleepAngularThreshold))
                {
                    body.sleeping = false;
                }
            }
        }
    }

    // 3) Record this frame's normal impulse as next frame's lastFrame
    //    reference (written for all bodies incl. Static for simplicity; Static
    //    never consumes it, no side effect).
    for (int i = 0; i < static_cast<int>(bodies_.size()); ++i)
    {
        bodies_[i].lastFrameNormalImpulse = thisFrameNormalImpulse[i];
    }
}
