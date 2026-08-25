// ============================================================================
// Contact constraint implementation.
// ============================================================================

#include "physics/ContactConstraint.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <glm/glm.hpp>

void ContactConstraint::prepare(PhysicsWorld &world, float /*dt*/)
{
    // Reset all point caches (defaults; defensive overwrite).
    for (int k = 0; k < 4; ++k)
        pointCache_[k] = PointCache{};
    maxInitialImpact_ = 0.0f;

    if (manifoldIndex_ < 0 || manifoldIndex_ >= static_cast<int>(world.manifolds_.size()))
        return;

    auto &m = world.manifolds_[manifoldIndex_];

    // Start every point as skipped; any early return then leaves the manifold
    // out of the SI loop.
    for (int k = 0; k < m.pointCount; ++k)
        pointCache_[k].skip = true;

    if (!m.alive || m.isTrigger || m.pointCount <= 0)
        return;

    auto &bodyA = world.bodies_[m.bodyA];
    auto &bodyB = world.bodies_[m.bodyB];
    bool bothStatic = bodyA.isStatic() && bodyB.isStatic();
    bool bothSleeping = bodyA.sleeping && bodyB.sleeping;
    if (bothStatic || bothSleeping)
        return;

    float invMassA = bodyA.inverseMass();
    float invMassB = bodyB.inverseMass();
    float invMassSum = invMassA + invMassB;
    if (invMassSum <= 0.0f)
        return;

    // Combine materials from both shapes.
    CombinedMaterial cm = combineMaterial(world.shapeMaterial_[m.shapeA], world.shapeMaterial_[m.shapeB]);
    const float bounceVelThreshold = cm.restitutionThreshold;

    // All points take part in solving (clear skip).
    for (int k = 0; k < m.pointCount; ++k)
    {
        auto &cp = m.points[k];
        auto &pc = pointCache_[k];
        pc.skip = false;

        glm::vec3 rA = cp.worldPoint - world.poses_[m.bodyA].position;
        glm::vec3 rB = cp.worldPoint - world.poses_[m.bodyB].position;

        // Normal denominator.
        glm::vec3 rAxN = glm::cross(rA, m.normal);
        glm::vec3 rBxN = glm::cross(rB, m.normal);
        float angN_A = glm::dot(glm::cross(bodyA.inverseInertiaWorld * rAxN, rA), m.normal);
        float angN_B = glm::dot(glm::cross(bodyB.inverseInertiaWorld * rBxN, rB), m.normal);
        pc.denomN = invMassSum + angN_A + angN_B;

        // Tangent denominators (both axes).
        for (int ti = 0; ti < 2; ++ti)
        {
            glm::vec3 t = cp.tangent[ti];
            glm::vec3 rAxT = glm::cross(rA, t);
            glm::vec3 rBxT = glm::cross(rB, t);
            float angT_A = glm::dot(glm::cross(bodyA.inverseInertiaWorld * rAxT, rA), t);
            float angT_B = glm::dot(glm::cross(bodyB.inverseInertiaWorld * rBxT, rB), t);
            float denomT = invMassSum + angT_A + angT_B;
            if (ti == 0)
                pc.denomT1 = denomT;
            else
                pc.denomT2 = denomT;
        }

        // Restitution target uses the post-warm-start velAlongNormal, but impactSpeed
        // uses the pre-warm-start velocity: warm starting only re-applies last frame's
        // accumulated impulse and must not mask the real bounce.
        glm::vec3 velA = bodyA.velocity + glm::cross(bodyA.angularVelocity, rA);
        glm::vec3 velB = bodyB.velocity + glm::cross(bodyB.angularVelocity, rB);
        float velAlongNormal = glm::dot(velB - velA, m.normal);
        float impactSpeed = -velAlongNormal;
        if (impactSpeed > maxInitialImpact_)
            maxInitialImpact_ = impactSpeed;
        if (impactSpeed > 0.0f)
        {
            float e = cm.restitution;
            if (impactSpeed < bounceVelThreshold)
                e *= impactSpeed / bounceVelThreshold;
            pc.restitutionTargetVel = e * impactSpeed;
        }
        else
        {
            pc.restitutionTargetVel = 0.0f;
        }
    }
}

void ContactConstraint::warmStart(PhysicsWorld &world)
{
    if (manifoldIndex_ < 0 || manifoldIndex_ >= static_cast<int>(world.manifolds_.size()))
        return;

    auto &m = world.manifolds_[manifoldIndex_];
    if (!m.alive || m.isTrigger)
        return;
    if (m.bodyA < 0 || m.bodyB < 0)
        return;
    auto &bodyA = world.bodies_[m.bodyA];
    auto &bodyB = world.bodies_[m.bodyB];
    if (bodyA.isStatic() && bodyB.isStatic())
        return;
    if (bodyA.sleeping && bodyB.sleeping)
        return;
    float invMassA = bodyA.inverseMass();
    float invMassB = bodyB.inverseMass();
    if (invMassA + invMassB <= 0.0f)
        return;

    for (int k = 0; k < m.pointCount; ++k)
    {
        auto &cp = m.points[k];
        // This frame's tangent basis (decoupled from the narrowphase normal).
        PhysicsWorld::buildTangentBasis(m.normal, cp.tangent[0], cp.tangent[1]);

        // Apply last frame's accumulated impulse: P = Jn*n + Jt1*t1 + Jt2*t2.
        glm::vec3 P = cp.accumNormalImpulse * m.normal +
                      cp.accumTangentImpulse[0] * cp.tangent[0] +
                      cp.accumTangentImpulse[1] * cp.tangent[1];

        glm::vec3 rA = cp.worldPoint - world.poses_[m.bodyA].position;
        glm::vec3 rB = cp.worldPoint - world.poses_[m.bodyB].position;
        bodyA.velocity -= invMassA * P;
        bodyB.velocity += invMassB * P;
        bodyA.angularVelocity -= bodyA.inverseInertiaWorld * glm::cross(rA, P);
        bodyB.angularVelocity += bodyB.inverseInertiaWorld * glm::cross(rB, P);
    }
}

void ContactConstraint::solveVelocity(PhysicsWorld &world)
{
    if (manifoldIndex_ < 0 || manifoldIndex_ >= static_cast<int>(world.manifolds_.size()))
        return;

    auto &m = world.manifolds_[manifoldIndex_];
    if (!m.alive || m.isTrigger || m.pointCount <= 0)
        return;
    auto &bodyA = world.bodies_[m.bodyA];
    auto &bodyB = world.bodies_[m.bodyB];
    float invMassA = bodyA.inverseMass();
    float invMassB = bodyB.inverseMass();

    // Friction is constant per manifold (combine result).
    CombinedMaterial cmF = combineMaterial(world.shapeMaterial_[m.shapeA], world.shapeMaterial_[m.shapeB]);
    float mu = cmF.friction;

    for (int k = 0; k < m.pointCount; ++k)
    {
        auto &cp = m.points[k];
        auto &pc = pointCache_[k];
        if (pc.skip)
            continue;
        if (pc.denomN <= 0.0f)
            continue;

        glm::vec3 rA = cp.worldPoint - world.poses_[m.bodyA].position;
        glm::vec3 rB = cp.worldPoint - world.poses_[m.bodyB].position;

        // ---- Normal ----
        {
            glm::vec3 velA = bodyA.velocity + glm::cross(bodyA.angularVelocity, rA);
            glm::vec3 velB = bodyB.velocity + glm::cross(bodyB.angularVelocity, rB);
            float velAlongNormal = glm::dot(velB - velA, m.normal);
            float dJn = (pc.restitutionTargetVel - velAlongNormal) / pc.denomN;
            float newJn = std::max(0.0f, cp.accumNormalImpulse + dJn);
            float appliedJn = newJn - cp.accumNormalImpulse;
            cp.accumNormalImpulse = newJn;
            if (appliedJn != 0.0f)
            {
                glm::vec3 P = appliedJn * m.normal;
                bodyA.velocity -= invMassA * P;
                bodyB.velocity += invMassB * P;
                bodyA.angularVelocity -= bodyA.inverseInertiaWorld * glm::cross(rA, P);
                bodyB.angularVelocity += bodyB.inverseInertiaWorld * glm::cross(rB, P);
            }
        }

        // ---- Tangents (friction-disk clamp) ----
        // Solve dJt1/dJt2 from the same relVel decomposition, then sum and project
        // the pair onto the Coulomb disk.
        glm::vec3 velA = bodyA.velocity + glm::cross(bodyA.angularVelocity, rA);
        glm::vec3 velB = bodyB.velocity + glm::cross(bodyB.angularVelocity, rB);
        glm::vec3 relVel = velB - velA;

        float vt1 = glm::dot(relVel, cp.tangent[0]);
        float vt2 = glm::dot(relVel, cp.tangent[1]);

        float dJt1 = (pc.denomT1 > 0.0f) ? (-vt1 / pc.denomT1) : 0.0f;
        float dJt2 = (pc.denomT2 > 0.0f) ? (-vt2 / pc.denomT2) : 0.0f;

        float newJt1 = cp.accumTangentImpulse[0] + dJt1;
        float newJt2 = cp.accumTangentImpulse[1] + dJt2;

        // Coulomb disk: |(Jt1, Jt2)| <= mu * Jn.
        float maxJt = mu * cp.accumNormalImpulse;
        float magJt = std::sqrt(newJt1 * newJt1 + newJt2 * newJt2);
        if (magJt > maxJt && magJt > 1e-8f)
        {
            float scale = maxJt / magJt;
            newJt1 *= scale;
            newJt2 *= scale;
        }

        float appliedJt1 = newJt1 - cp.accumTangentImpulse[0];
        float appliedJt2 = newJt2 - cp.accumTangentImpulse[1];
        cp.accumTangentImpulse[0] = newJt1;
        cp.accumTangentImpulse[1] = newJt2;

        if (appliedJt1 != 0.0f || appliedJt2 != 0.0f)
        {
            glm::vec3 P = appliedJt1 * cp.tangent[0] + appliedJt2 * cp.tangent[1];
            bodyA.velocity -= invMassA * P;
            bodyB.velocity += invMassB * P;
            bodyA.angularVelocity -= bodyA.inverseInertiaWorld * glm::cross(rA, P);
            bodyB.angularVelocity += bodyB.inverseInertiaWorld * glm::cross(rB, P);
        }
    }
}

float ContactConstraint::solvePosition(PhysicsWorld & /*world*/)
{
    // Position solving intentionally stays in PhysicsWorld::singleStep rather than
    // moving into ContactConstraint: the position solver is keyed on contacts_ (not
    // manifolds_), relies on shared cross-constraint state (posDelta[]), and
    // remapping would risk index mismatches and float reordering. Revisit once
    // contacts_ is properly aligned with the manifold layout. Returning 0 keeps this
    // constraint out of the position pass, which still runs over contacts_.
    return 0.0f;
}
