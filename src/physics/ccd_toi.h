// ============================================================================
// Conservative Advancement TOI for CCD: the fraction t ∈ [0, 1] of dt at which
// two shapes first touch. Linear velocity only; angular velocity is ignored
// (the industry default, e.g. Bullet), since rotation-induced tunneling is rare.
// ============================================================================
#pragma once

#include <glm/glm.hpp>
#include "physics/Shape.h"

// Time fraction t ∈ [0, 1] of dt at which A and B first touch; 1 means no contact
// this step, 0 means penetration already present (handed to the discrete solver).
//   shA/poseA0/linVelA — A's shape, start-of-step pose, linear velocity; B likewise
//   dt                 — step duration (s)
//   tolerance          — distance below which contact is assumed (suggest 1e-3 m)
//   maxIter            — CA main-loop cap (suggest 16)
//   outNormal          — separating normal at first contact (B → A; undefined at t == 1)
float computeTOI(const Shape &shA, const BodyPose &poseA0, const glm::vec3 &linVelA,
                 const Shape &shB, const BodyPose &poseB0, const glm::vec3 &linVelB,
                 float dt, float tolerance, int maxIter,
                 glm::vec3 &outNormal);
