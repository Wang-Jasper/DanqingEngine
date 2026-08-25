# ============================================================================
# spin_cube.py - Phase 13.6 per-entity script (Unity MonoBehaviour-style)
# ----------------------------------------------------------------------------
# Attach this script to any Entity that has a TransformComponent (drop a
# ScriptComponent on it via the Inspector or set scriptPath in the scene
# JSON / SceneSetup). The default scene attaches it to the "Cube" entity.
#
# Lifecycle (driven by ScriptEngine::callOnStartAll/UpdateAll/StopAll):
#   * on_start(self)       : called once when Play begins. We snapshot the
#                            starting Y rotation so on_stop can restore it.
#   * on_update(self, dt)  : called every frame while simulation runs. We
#                            advance the Y rotation by 90 deg/s.
#   * on_stop(self)        : called once when Play stops, before the
#                            transform-snapshot restore in PhysicsSystem.
#
# Two cubes carrying the same script run in fully isolated namespaces:
# `_starting_rotation` is module-private to this entity's globals dict
# (ScriptEngine builds a fresh dict per attach), so multiple instances
# don't trip over each other's state.
#
# `self` is a live PyEntity handle injected by ScriptEngine immediately
# before each callback. It stays valid as long as the entity exists in
# the registry; if the entity is destroyed mid-simulation the handle
# evaluates to False and we bail out.
#
# `engine` is also injected (the embedded engine module), so we can use
# engine.log / engine.vec3 without an explicit `import engine` line. We
# still keep the import below for editor IDE autocomplete + type hints.
# ============================================================================

import engine
from engine import log, vec3

# Per-instance state lives at module level. ScriptEngine assigns each
# attach a brand-new globals dict, so this `None` is the starting value
# for every entity independently. There is intentionally no class wrapper
# here — Unity-style "one script = one component" maps cleanly onto
# module-level globals + on_*() free functions.
_starting_rotation = None  # vec3 of starting Euler angles (degrees)
_rotation_speed_dps = 900.0  # degrees per second around the Y axis


def on_start(self):
    """Called once when Play begins. `self` is the PyEntity we're attached to."""
    global _starting_rotation

    if not self:
        log("spin_cube: on_start - self handle is invalid, skipping.")
        return
    if not self.has_TransformComponent():
        log("spin_cube: on_start - entity has no TransformComponent, skipping.")
        return

    t = self.get_TransformComponent()
    # Defensive copy: store as a fresh vec3 so mutations to t.rotation
    # later don't retroactively change our stored snapshot.
    _starting_rotation = vec3(t.rotation.x, t.rotation.y, t.rotation.z)
    log(f"spin_cube: on_start - entity #{self.id}, start_rot={_starting_rotation}")


def on_update(self, dt):
    """Called every frame while simulationRunning && !simulationPaused."""
    if not self:
        return
    if not self.has_TransformComponent():
        return

    t = self.get_TransformComponent()
    rot = t.rotation
    # Only Y axis spins; X and Z stay at their current values so other
    # scripts / physics adjustments don't get clobbered.
    new_y = rot.y + _rotation_speed_dps * dt
    # Wrap around 360 to keep the angle bounded (avoids float blow-up
    # over multi-minute Play sessions).
    if new_y > 360.0:
        new_y -= 360.0
    elif new_y < -360.0:
        new_y += 360.0
    t.rotation = vec3(rot.x, new_y, rot.z)


def on_stop(self):
    """Called once when Play stops, BEFORE PhysicsSystem::exitPlay restores
    transform snapshots. We restore the starting rotation explicitly so the
    visual state is identical to what the editor showed before Play. The
    snapshot path will overwrite this in the default Stop branch, but the
    Load-Scene path uses exitPlay(restoreSnapshots=false), so we still
    behave correctly there.
    """
    global _starting_rotation
    if self and self.has_TransformComponent() and _starting_rotation is not None:
        self.get_TransformComponent().rotation = _starting_rotation
        log(f"spin_cube: on_stop - entity #{self.id}, restored Y={_starting_rotation.y}")
    _starting_rotation = None


# Top-level marker logged once per attach. Two entities sharing this
# script will print this twice (once per attach), confirming they got
# independent globals dicts.
log("spin_cube: ready (per-entity attach)")
