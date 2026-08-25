#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include "physics/Shape.h"
#include "physics/PhysicsMaterial.h"
#include "physics/DynamicAABBTree.h"
#include "physics/ContactConstraint.h"

// OBB - plain value type for external interfaces and narrowphase input
// (testOBBOverlap / computeContact); internal storage uses BodyPose + Shape.
struct OBB
{
    glm::vec3 center = glm::vec3(0.0f);      // world-space center
    glm::vec3 halfExtents = glm::vec3(0.0f); // half-extents per axis (local space)
    glm::mat3 orientation = glm::mat3(1.0f); // rotation (local axes -> world axes)
};

// BodyType - rigid body type. Kinematic poses are written externally
// (Gizmo/animation/script); the body ignores forces and impulses, yet its
// pose-derived velocity still contributes nonzero relVel to the opposite
// Dynamic body during solving (elevators, conveyor bases, animated skeletons).
// Static and Kinematic have inverseMass 0; only the Dynamic side gets impulse.
enum class BodyType : uint8_t
{
    Static = 0,
    Dynamic = 1,
    Kinematic = 2
};

// RigidBody - runtime rigid-body data. friction/restitution now live entirely
// in shape-level PhysicsMaterial (CP-3.2): pass a material via attachShape or
// the addBody convenience parameter, never a body-level fallback. This removes
// the dual-parameter ambiguity; Inspector edits land on ColliderDesc.material.
struct RigidBody
{
    BodyType bodyType = BodyType::Dynamic;
    float mass = 1.0f;

    glm::vec3 velocity = glm::vec3(0.0f);
    glm::vec3 forceAccum = glm::vec3(0.0f);
    glm::vec3 angularVelocity = glm::vec3(0.0f); // angular velocity (world space, rad/s)
    glm::vec3 torqueAccum = glm::vec3(0.0f);     // accumulated torque

    // local-space inertia tensor; 3x3 symmetric, usually non-diagonal after
    // combining multiple colliders via the parallel-axis theorem
    glm::mat3 inertiaLocalMat = glm::mat3(1.0f);

    // world-space inverse inertia tensor, updated per frame from orientation
    glm::mat3 inverseInertiaWorld = glm::mat3(0.0f);

    bool sleeping = false; // sleeping: skip gravity/integration, avoids ground jitter
    int sleepFrames = 0;   // frames the rest condition held consecutively

    // last frame's accumulated normal impulse (sum of accumNormalImpulse over
    // all alive manifolds touching this body); feeds the island sleep check
    // "contact-force change < threshold". Persists across frames.
    float lastFrameNormalImpulse = 0.0f;

    float inverseMass() const
    {
        // Only Dynamic receives impulse. Static (immovable floor/wall) and
        // Kinematic (externally authored pose) return 0, though kinematic
        // velocity still acts on the opposite Dynamic through contacts
        // (see ContactConstraint).
        return (bodyType != BodyType::Dynamic || mass <= 0.0f) ? 0.0f : 1.0f / mass;
    }
    bool isStatic() const { return bodyType == BodyType::Static; }
    bool isKinematic() const { return bodyType == BodyType::Kinematic; }
};

// ContactPoint / ContactManifold - persistent manifold + warm start.
// Up to 4 contact points per (bodyA, bodyB) pair kept across frames:
//   - localA/localB: point in each body's local space, for cross-frame matching
//   - accumNormalImpulse / accumTangentImpulse[2]: warm-start carry-over
//   - tangent[2]: two tangent basis vectors (friction cone)
// New points match last frame's manifold by local-space distance (threshold =
// 5% of the smallest contact shape half-extent); matches inherit impulse.
struct ContactPoint
{
    glm::vec3 localA = glm::vec3(0.0f);     // contact point in bodyA local space
    glm::vec3 localB = glm::vec3(0.0f);     // contact point in bodyB local space
    glm::vec3 worldPoint = glm::vec3(0.0f); // world-space point, current-frame value only
    float penetration = 0.0f;

    // two tangent basis vectors, orthogonal to the manifold-level normal
    glm::vec3 tangent[2] = {glm::vec3(0.0f), glm::vec3(0.0f)};

    float accumNormalImpulse = 0.0f;
    float accumTangentImpulse[2] = {0.0f, 0.0f};

    bool matched = false; // matched by a new contact point this frame (lifetime management)
};

struct ContactManifold
{
    int bodyA = -1;
    int bodyB = -1;
    int shapeA = -1; // representative shape (event dispatch, rolling friction)
    int shapeB = -1;
    glm::vec3 normal = glm::vec3(0.0f); // points from A to B
    int pointCount = 0;
    ContactPoint points[4];
    bool isTrigger = false;
    bool alive = false; // still touching this frame (Enter/Stay/Exit detection)
};

// ContactInfo - single-point contact info for events/debug. The solver no
// longer reads contacts_; it walks manifolds_ directly, and contacts_ is
// expanded from manifolds_ each frame for UI/log/events. ContactInfo is a pure
// geometric snapshot (pos/normal/penetration/indices): the old accumulated
// impulse/tangent fields were written once at detection and never followed the
// solver, which was misleading. Consumers needing true Jn/Jt read
// getManifolds() -> ContactManifold::points[k].accumNormalImpulse.
struct ContactInfo
{
    int indexA = -1;
    int indexB = -1;
    int shapeIndexA = -1;
    int shapeIndexB = -1;
    glm::vec3 normal = glm::vec3(0.0f); // collision normal, A -> B
    float penetration = 0.0f;
    glm::vec3 contactPoint = glm::vec3(0.0f);
};

// Raycast / overlap query return types.
struct Ray
{
    glm::vec3 origin = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
};

struct RaycastHit
{
    int bodyIndex = -1;
    int shapeIndex = -1;
    glm::vec3 point = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);
    float t = 0.0f;
};

// ContactEvent - contact event. Enter: pair new this frame (absent last frame);
// Stay: present both frames; Exit: gone this frame. Trigger events share the
// same stream, decided by shape.isTrigger (whether impulse is produced is up to
// the solver, but both kinds dispatch events).
enum class ContactPhase : uint8_t
{
    Enter = 0,
    Stay = 1,
    Exit = 2,
};

struct ContactEvent
{
    int bodyA = -1;
    int bodyB = -1;
    int shapeA = -1;
    int shapeB = -1;
    ContactPhase phase = ContactPhase::Enter;
    bool isTrigger = false;
    glm::vec3 point = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);
    float impulse = 0.0f; // 0 on Exit
};

// PhysicsStats - runtime statistics.
struct PhysicsStats
{
    int bodyCount = 0;
    int activeBodyCount = 0;
    int sleepingBodyCount = 0;
    int shapeCount = 0;
    int broadphaseNodeCount = 0;
    int broadphasePairs = 0; // candidate pairs from last broadphase step
    int contactCount = 0;    // contacts solved in the last step
    int substepsLastFrame = 0;
    float lastStepMs = 0.0f;

    // island statistics
    int islandCount = 0;       // total islands this frame (all Dynamic bodies)
    int activeIslandCount = 0; // islands with at least one non-sleeping body
};

// Island - connected component of the contact graph: bodies joined through
// manifolds form one island. Static bodies connect nodes but never accumulate
// sleep frames. Fields:
//   bodies          - all body indices in the island (incl. Static; unordered)
//   manifoldIndices - manifold indices joining body pairs (alive && !trigger)
//   sleepFrames     - consecutive frames below the low-speed/low-impulse bars
//   allLowLastFrame - previous frame met the sleep-candidate condition
//   allSleeping     - all Dynamic bodies sleeping; lets warmStart/resolve skip
struct Island
{
    std::vector<int> bodies;
    std::vector<int> manifoldIndices;
    int sleepFrames = 0;
    bool allLowLastFrame = false;
    bool allSleeping = false;
};

// PhysicsWorld - the physics world.
class PhysicsWorld
{
    // ContactConstraint needs private data (manifolds_/bodies_/poses_/shapeMaterial_);
    // friend is clearer than public getters and costs nothing at runtime.
    friend class ContactConstraint;

public:
    glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    float fixedTimeStep = 1.0f / 60.0f;
    int maxSubSteps = 8;

    // warm-starting toggle (same-named option in Bullet/Box2D):
    //   - true (default): seed each frame's solve with last frame's accumulated
    //     impulses, so sequential-impulse iterations converge faster and stacks
    //     stay stable
    //   - false: start from accumulatedImpulse=0 each frame, to compare the
    //     effect on stack stability
    bool warmStartEnabled = true;

    // CCD toggle (Bullet-style). true (default): at the start of singleStep,
    // run Conservative Advancement TOI for Dynamic bodies whose displacement
    // this step exceeds 0.5 * min(shape AABB half-extent); if contact with a
    // Static body is imminent, clamp this body's dt to the TOI and carry the
    // remainder to the next frame. false: discrete detection only (fast bodies
    // may tunnel through thin objects). First version handles dynamic-vs-static
    // only, the standard practice.
    bool ccdEnabled = true;

    float maxAngularSpeed = 30.0f; // rad/s - CCD ignores rotation; numeric fallback

    // Debug logs off by default; UI can enable at runtime.
    bool debugLogEnabled = false;

    // Resting-lock thresholds (promoted from resolveCollisions locals).
    // While contact exists, zero out small residual velocities to kill
    // long-term jitter. Orthogonal to sleep:
    //   - resting-lock: applied every frame contact exists (even pre-sleep)
    //   - sleep: whole island skips integrate after enough frames
    // Units: linear m/s; angular rad/s; centerLock = support-point spread m.
    float restingNormalLockThreshold = 0.12f;
    float restingTangentLockThreshold = 0.18f;
    float restingAngularLockThreshold = 0.20f;
    float restingCenterLockThreshold = 0.18f;

    // Sleep thresholds (promoted from resolveCollisions locals). An island
    // sleeps when all its Dynamic bodies stay below the linear/angular
    // thresholds for sleepFramesRequired consecutive frames; any impulse change
    // beyond wakeImpactThreshold wakes it. Same solver-parameter tier as
    // warmStartEnabled/ccdEnabled.
    float sleepLinearThreshold = 0.08f;
    float sleepAngularThreshold = 0.12f;
    int sleepFramesRequired = 12;
    float wakeImpactThreshold = 0.25f;

    // =====================================================================
    // Body API
    // =====================================================================
    // addBody(body, obb, material={}) - single-collider convenience that calls
    // addEmptyBody + attachShape(Box); material passed explicitly (default =
    // friction 0.5 / restitution 0.3, matching historical behavior).
    int addBody(const RigidBody &body, const OBB &obb,
                const PhysicsMaterial &material = {});
    int addEmptyBody(const RigidBody &body, const BodyPose &pose);
    void removeBody(int bodyIndex);
    void clear();

    void updateOBB(int bodyIndex, const OBB &obb);
    void setBodyPose(int bodyIndex, const BodyPose &pose);
    void updateBody(int bodyIndex, const RigidBody &body);
    void updateShapeExtents(int bodyIndex, const glm::vec3 &halfExtents);

    void stepSimulation(float deltaTime);

    const RigidBody &getBody(int bodyIndex) const { return bodies_[bodyIndex]; }

    OBB getOBB(int bodyIndex) const;

    int bodyCount() const { return static_cast<int>(bodies_.size()); }
    bool isActive(int bodyIndex) const { return active_[bodyIndex]; }

    glm::vec3 getPosition(int bodyIndex) const;
    const std::vector<ContactInfo> &getContacts() const { return contacts_; }

    // Read-only manifolds_ for tests/Inspector to read the true per-point
    // Jn/Jt: the solver's latest accumulatedImpulse lives in
    // manifolds_[mi].points[k]; the old impulse fields on contacts_ were
    // removed because they were detection-time snapshots that never tracked
    // the solver.
    const std::vector<ContactManifold> &getManifolds() const { return manifolds_; }

    // =====================================================================
    // Force / wake API (minimal; extended to a full interface later)
    // ---------------------------------------------------------------------
    // Every entry point auto-wakes the island: all Dynamic members of the
    // body's island wake the same frame, no per-frame neighbor contagion.
    // Static / sleeping-only branches are handled internally.
    // =====================================================================
    void applyImpulse(int bodyIndex, const glm::vec3 &impulse,
                      const glm::vec3 &worldPoint);
    void applyForce(int bodyIndex, const glm::vec3 &force,
                    const glm::vec3 &worldPoint);
    void applyTorque(int bodyIndex, const glm::vec3 &torque);
    // Set linear/angular velocity directly (scripts/tests/init). Auto-wakes
    // the island so the new speed is not swallowed by sleep state.
    void setLinearVelocity(int bodyIndex, const glm::vec3 &v);
    void setAngularVelocity(int bodyIndex, const glm::vec3 &w);
    // Explicit wake: mark every Dynamic body of the island non-sleeping and
    // reset island.sleepFrames (islands_ is rebuilt each frame, so body-level
    // sleepFrames is reset too).
    void wakeBody(int bodyIndex);

    // =====================================================================
    // Shape API
    // =====================================================================
    int attachShape(int bodyIndex,
                    const Shape &shape,
                    const BodyPose &localPose,
                    const PhysicsMaterial &material,
                    uint32_t layer,
                    uint32_t mask,
                    bool isTrigger);
    void detachShape(int bodyIndex, int shapeIndex);

    int shapeCount() const { return static_cast<int>(shapes_.size()); }
    bool isShapeActive(int shapeIndex) const { return shapeActive_[shapeIndex]; }

    // =====================================================================
    // ConvexHull registration
    // ---------------------------------------------------------------------
    // Registers local vertices as a ConvexMeshCache, returns hullIndex;
    // reusable across attachShape calls (bodies share one cache, zero vertex
    // copying). attachShape backfills shape.hullCache to hulls_[hullIndex]
    // when type==ConvexHull && hullIndex>=0.
    // =====================================================================
    int registerConvexHull(const std::vector<glm::vec3> &localVertices);
    const ConvexMeshCache *getConvexHull(int hullIndex) const;

    // =====================================================================
    // Layer / mask API
    // =====================================================================
    void setBodyLayerMask(int bodyIndex, uint32_t layer, uint32_t mask);

    // =====================================================================
    // Query API
    // =====================================================================
    // Ray: nearest hit, broadphase coarse filter + OBB slab test per candidate.
    bool raycast(const Ray &ray, float maxT, uint32_t mask, RaycastHit &out) const;

    // Broadphase AABB overlap query (deduped body indices); no narrowphase.
    int overlapAABB(const AABB &aabb, uint32_t mask,
                    int *outBodies, int maxOut) const;

    // Broadphase sphere overlap: AABB (center - r, center + r) coarse pass,
    // then closest-point-to-OBB distance <= r per candidate body.
    int overlapSphere(const glm::vec3 &center, float radius, uint32_t mask,
                      int *outBodies, int maxOut) const;

    // =====================================================================
    // Event stream
    // =====================================================================
    const std::vector<ContactEvent> &events() const { return events_; }
    void clearEvents() { events_.clear(); }

    // =====================================================================
    // Stats
    // =====================================================================
    const PhysicsStats &stats() const { return stats_; }

    // =====================================================================
    // Island debug accessors
    // ---------------------------------------------------------------------
    // Rebuilt by buildIslands() at the end of each detectCollisions;
    // islandId_[bi] is the body's island index in [0, islands_.size()), or -1
    // if the body is inactive. Read by unit tests, the Editor panel, and the
    // island-scoped solver.
    // =====================================================================
    const std::vector<int> &debugIslandIds() const { return islandId_; }
    int debugIslandCount() const { return static_cast<int>(islands_.size()); }
    const Island &debugIsland(int idx) const { return islands_[idx]; }

    // =====================================================================
    // Narrowphase pure functions, callable from unit tests
    // ---------------------------------------------------------------------
    // Idempotent, side-effect-free OBB-OBB tools:
    //   - testOBBOverlap: SAT separation test (fast reject)
    //   - computeBoxBoxManifold: contact manifold (normal + up to 4 points)
    // Public so tests/physics can do numeric regression; no member state read
    // or written (parameters are the only inputs).
    // =====================================================================
    bool testOBBOverlap(const OBB &a, const OBB &b) const;
    bool computeBoxBoxManifold(const OBB &a, const OBB &b,
                               int idxA, int idxB,
                               int shapeA, int shapeB,
                               ContactManifold &out) const;

private:
    void singleStep(float dt);
    void integrate(float dt);
    void detectCollisions();

    // Conservative-Advancement CCD sweep, run before integrate. Dynamic bodies
    // with displacement > own min AABB radius get sweptAABB broadphase coarse
    // filter + computeTOI; the TOI lands in pendingTOI_[bodyIdx] and integrate
    // clamps that body's dt. Dynamic-vs-static pairs only (standard first pass).
    void ccdPhase(float dt);

    // Kinematic velocities derived implicitly from external pose deltas.
    // Called once per frame at the top of stepSimulation (not per substep);
    // diff poses_[i] vs prevPoses_[i] into bodies_[i].velocity/.angularVelocity:
    //   - linVel = (pos - prevPos) / dt
    //   - angVel = axis-angle(prevOrient^T * orient) / dt
    // If displacement exceeds kKinematicMaxStepSpeed*dt (teleport), both are
    // zeroed so a Gizmo jump or scene reset cannot inject a huge fake velocity
    // that corrupts the next frame's contact solve.
    //
    // Snapshots prevPoses_ = poses_ at the end for the next frame. Frame-constant
    // velocity keeps the same relVel driving tangent impulses across substeps
    // in ContactConstraint::solveVelocity (kinematic speed never resets mid-frame).
    //
    // Generic: branch only on bodyType==Kinematic, no
    // per-body/per-scene/per-shape special cases.
    static constexpr float kKinematicMaxStepSpeed = 100.0f;
    void refreshKinematicVelocities(float dt);

    // Narrowphase emits manifolds directly (up to 4 points), replacing the old
    // single-point ContactInfo. Returns true when contact is valid
    // (pointCount > 0 && penetration > 0).
    // Cross-frame matching: find the old manifold for (bodyA, bodyB), match new
    // points to old ones by local-space distance (threshold = 5% of the smaller
    // shape's min half-extent) and inherit accumulated impulses; unmatched old
    // points are dropped.
    void mergeManifoldWithPersistent(ContactManifold &fresh);
    void resolveCollisions(float dt);
    void warmStartManifolds();

    // -------------------------------------------------------------------
    // Island construction (once per frame, end of detectCollisions)
    // -------------------------------------------------------------------
    // Union alive && !trigger manifolds into body groups, then fill islands_[]
    // by find results. Static bodies connect but never accumulate sleep.
    // O((N + M)*alpha(N)), N = bodies, M = alive manifolds.
    void buildIslands();
    // Union-find helpers (path compression + union by rank; used inside buildIslands)
    int unionFind(int i);
    void unionUnite(int i, int j);

    // -------------------------------------------------------------------
    // Static-stability criterion, a necessary condition for island sleep
    // -------------------------------------------------------------------
    // After the velocity/angular-velocity/impulse-change "dynamically low" bars
    // pass, additionally require: each Dynamic body's COM, projected along
    // gravity, must fall inside the convex hull of its contact-point
    // projections - otherwise gravity torque tips the pose over and sleep is
    // refused.
    //
    // Pure geometry: uses only pos / contact worldPoint / gravity / the body's
    // first shape halfExtents (as an eps scale reference); valid for any shape
    // and any gravity direction.
    //
    // Near-zero gravity (|gravity| < 1e-4) counts as stable (returns true).
    // -------------------------------------------------------------------
    bool isStaticallyStable(int bodyIndex,
                            const std::vector<int> &manifoldIndices) const;
    bool isStaticallyStableIsland(const Island &island) const;

    // Two tangent basis vectors per point (Erin Catto's stabilized method),
    // orthonormal with the normal.
    static void buildTangentBasis(const glm::vec3 &normal,
                                  glm::vec3 &t1, glm::vec3 &t2);

    void updateInertia(int bodyIndex);

    OBB composeShapeOBB(int shapeIndex) const;

    // Compose the shape's body-local pose with the body's world pose into one
    // world BodyPose; reused on hot paths (detect/ccd/broadphase).
    //   world.position    = body.pos + body.orient * shape.local.pos
    //   world.orientation = body.orient * shape.local.orient
    BodyPose composeShapePose(int shapeIndex) const;

    bool passLayerMask(int bodyIndexA, int bodyIndexB) const;

    // Body's world AABB: union of all its active shapes' AABBs.
    AABB computeBodyAABB(int bodyIndex) const;

    // Refresh every active body's broadphase proxy (once at the start of each
    // singleStep). const because broadphase_/broadphaseProxy_/stats_ are
    // mutable caches/statistics, not logical world state; const query APIs may
    // refresh before querying.
    void refreshBroadphase() const;

    // body-level, 1:1 with bodies_
    std::vector<RigidBody> bodies_;
    std::vector<BodyPose> poses_;
    // Last frame's pose snapshot for Kinematic velocity derivation (same length
    // as poses_; kept in sync on add/remove/clear). refreshKinematicVelocities
    // diffs poses_[i] - prevPoses_[i], then snapshots prevPoses_ = poses_.
    // Dynamic/Static bodies are maintained too (no extra cost); readers decide
    // by bodyType.
    std::vector<BodyPose> prevPoses_;
    std::vector<bool> active_;
    std::vector<uint32_t> bodyLayer_;
    std::vector<uint32_t> bodyMask_;
    std::vector<std::vector<int>> bodyShapes_;

    // CCD: each Dynamic body's time-of-impact this frame (fraction of dt in
    // [0, 1]). 1.0 = no penetration risk this step (no clamp); < 1.0 = sweep
    // found a Static-body contact within the step, integrate clamps dt to that
    // fraction and the remainder goes to the next frame's solver + CCD.
    // Reset and filled by ccdPhase at the start of each singleStep.
    std::vector<float> pendingTOI_;

    // broadphase: one proxy per body (-1 = not attached to the tree).
    // mutable so const query APIs can refresh before querying.
    mutable std::vector<int> broadphaseProxy_;
    mutable DynamicAABBTree broadphase_;

    // shape-level, 1:1 (slots reusable)
    std::vector<Shape> shapes_;
    std::vector<BodyPose> shapeLocal_;
    std::vector<PhysicsMaterial> shapeMaterial_;
    std::vector<uint32_t> shapeLayer_;
    std::vector<uint32_t> shapeMask_;
    std::vector<bool> shapeIsTrigger_;
    std::vector<int> shapeBody_;
    std::vector<bool> shapeActive_;

    std::vector<ContactInfo> contacts_;

    // Shared ConvexHull data pool. std::vector keeps pointers stable as long as
    // it does not reallocate, so registration reserves room up front;
    // Shape.hullCache points into this vector.
    // TODO: switch to std::list/std::deque for address stability if
    // multi-world or migration support lands. reserve(32) suffices today.
    std::vector<ConvexMeshCache> hulls_;

    // Cross-frame contact-pair cache for Enter/Stay/Exit differencing. Packs
    // (bodyA, bodyB) into an ordered long long (low 32 bits = min, high =
    // max); sorting each frame and linearly merging with the previous frame's
    // sorted set yields Enter/Stay/Exit with no hash table or cross-frame table.
    std::vector<long long> prevContactPairs_;
    std::vector<long long> currContactPairs_;
    std::vector<ContactEvent> events_;

    // Persistent-manifold storage. Key = (min(bodyA,bodyB)<<32)|max; each
    // singleStep starts with detectCollisions:
    //   1) mark all alive old manifolds alive=false;
    //   2) after narrowphase produces points, match them against old localA/
    //      localB and inherit accumNormalImpulse/accumTangentImpulse (warm start);
    //   3) old manifolds untouched this frame stay alive=false and drop next frame.
    // vector + linear scan because the pair count is bounded by broadphase and
    // the warm-start hit rate is >90%, making linear scans faster and more
    // cache-friendly than a hash table.
    std::vector<ContactManifold> manifolds_;

    // One ContactConstraint per alive && !trigger manifold each frame; indices
    // align with manifolds_ (contactConstraints_[mi] <-> manifolds_[mi]).
    // Skipped manifolds (static-static / both-sleeping / !alive) get all
    // pointCache_[k].skip = true in prepare, so solveVelocity bypasses them.
    // unique_ptr keeps Constraint* polymorphic (joints will share this vector).
    std::vector<std::unique_ptr<Constraint>> contactConstraints_;

    // Island structures (rebuilt by buildIslands at the end of each
    // detectCollisions). unionParent_/unionRank_: temporary union-find data,
    // sized bodies_.size(), resized + self-initialized at the top of each
    // buildIslands; never kept across frames (body sets may change).
    // islandId_[bi]: body bi's island index in [0, islands_.size()), -1 if
    // inactive. islands_: this frame's full island list (see Island).
    std::vector<int> unionParent_;
    std::vector<int> unionRank_;
    std::vector<int> islandId_;
    std::vector<Island> islands_;

    // ---------------------------------------------------------------------
    // Scratch buffers - per-frame work memory reused by solver / position
    // solver / sleep checks
    // ---------------------------------------------------------------------
    // These fields do NOT represent world state; they just avoid heap
    // allocating vectors sized to bodies_.size()/contacts_.size() every
    // singleStep. Convention: assign(size, init_value) to reset at each
    // singleStep, then consume. Versus local std::vector<T>(N, v) this only
    // saves malloc/free; numerics are identical.
    std::vector<bool> scratchHadContact_;
    std::vector<glm::vec3> scratchSupportNormals_;
    std::vector<glm::vec3> scratchSupportPoints_;
    std::vector<int> scratchSupportCounts_;
    // Per-body accumulated support-surface tangential velocity, same
    // denominator as supportCounts. Accumulates
    // otherBody.velocity + otherBody.angularVelocity x (contactPoint - otherBody.pos)
    // during the support loop so applyRestingStaticFriction locks resting to
    // vSurface's tangential component (not to 0): the upper body follows a
    // moving/rotating Kinematic base, and reverts to resting when it stops.
    std::vector<glm::vec3> scratchSupportVelocities_;
    std::vector<float> scratchThisFrameNormalImpulse_;
    std::vector<float> scratchPosImpulse_;   // sized contacts_.size()
    std::vector<glm::vec3> scratchPosDelta_; // sized bodies_.size()
    std::vector<float> scratchInitialPen_;   // sized contacts_.size()

    PhysicsStats stats_;
    float accumulator_ = 0.0f;
};
