// ============================================================================
// Conservative Advancement TOI implementation. See ccd_toi.h for details.
// ============================================================================

#include "physics/ccd_toi.h"
#include "physics/narrowphase_gjk.h"
#include <algorithm>
#include <cmath>

float computeTOI(const Shape &shA, const BodyPose &poseA0, const glm::vec3 &linVelA,
                 const Shape &shB, const BodyPose &poseB0, const glm::vec3 &linVelB,
                 float dt, float tolerance, int maxIter,
                 glm::vec3 &outNormal)
{
    outNormal = glm::vec3(1, 0, 0); // fallback

    if (dt <= 0.0f)
        return 1.0f;

    glm::vec3 vRel = linVelA - linVelB;
    float vRelMag = glm::length(vRel);
    if (vRelMag < 1e-6f)
    {
        // Effectively at rest: no relative displacement within dt, so no new penetration.
        return 1.0f;
    }

    // Remember the normal from the last gjk_distance call that reported separation;
    // penetration normals are unreliable. Falling back to the last good one is
    // correct for head-on cases: the separating direction cannot flip as we advance.
    glm::vec3 lastGoodNormal(1, 0, 0);
    bool hasGoodNormal = false;

    float t = 0.0f;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        // Pose at time t: pose0 + linVel * t * dt.
        BodyPose pA = poseA0;
        BodyPose pB = poseB0;
        pA.position = poseA0.position + linVelA * (t * dt);
        pB.position = poseB0.position + linVelB * (t * dt);

        float d;
        glm::vec3 n, ptA, ptB;
        bool sep = gjk_distance(shA, pA, shB, pB, d, n, ptA, ptB);

        if (!sep)
        {
            // Penetrating or too close: return t, with lastGoodNormal if available
            // (otherwise an initial-direction guess).
            if (hasGoodNormal)
                outNormal = lastGoodNormal;
            else
            {
                // Initial guess: direction from B0 to A0.
                glm::vec3 init = poseA0.position - poseB0.position;
                float l = glm::length(init);
                outNormal = (l > 1e-6f) ? (init / l) : glm::vec3(1, 0, 0);
            }
            return t;
        }

        lastGoodNormal = n;
        hasGoodNormal = true;

        if (d < tolerance)
        {
            // Close enough to count as contact.
            outNormal = n;
            return t;
        }

        // Approach speed: vRel projected along A→B. n points B→A, so A→B is -n.
        float approachSpeed = glm::dot(vRel, -n);
        if (approachSpeed <= 1e-6f)
        {
            // Separating or sliding tangentially: no contact within dt.
            return 1.0f;
        }

        // Conservative advance: dt_safe = d / approachSpeed, normalized to a fraction.
        float advanceFraction = d / (approachSpeed * dt);

        // Avoid tiny advances that would make the loop iterate forever.
        const float minAdvance = tolerance / std::max(vRelMag * dt, 1e-12f);
        if (advanceFraction < minAdvance)
        {
            // Advance below tolerance: the contact point is already close enough.
            outNormal = n;
            return t;
        }

        t += advanceFraction;
        if (t >= 1.0f)
            return 1.0f;
    }

    // No convergence within maxIter: stop conservatively at the last known safe t.
    if (hasGoodNormal)
        outNormal = lastGoodNormal;
    return t;
}
