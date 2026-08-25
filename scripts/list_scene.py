# ============================================================================
# list_scene.py - Phase 13.3 verification for Entity / Scene / Camera handles.
# ----------------------------------------------------------------------------
# Exercises the API the way a real user script would: walk the scene, read
# components, demonstrate find_by_name, and probe Camera read/write.
#
# Auto-loaded by DeferredRenderer::init() after smoke.py + verify_bindings.py.
# Failure raises and is captured by ScriptEngine.capturePythonError; the
# console will show a full Python traceback.
# ============================================================================

from engine import (
    log,
    scene,
    camera,
    delta_time,
    Entity,
    BodyType,
    vec3,
)


# ---------------------------------------------------------------------------
# Sanity: scene / camera should be bound by setSceneContext()
# ---------------------------------------------------------------------------
if not scene:
    raise RuntimeError("engine.scene is not bound — setSceneContext() was not called.")
if not camera:
    raise RuntimeError("engine.camera is not bound — setSceneContext() was not called.")

log(f"engine.delta_time at load = {delta_time}")
log(f"camera.position = {camera.position}")
log(f"camera.fov      = {camera.fov}")
log(f"camera.near_plane = {camera.near_plane}")
log(f"camera.far_plane  = {camera.far_plane}")

# Round-trip a Camera setter and restore.
old_fov = camera.fov
camera.fov = 60.0
if abs(camera.fov - 60.0) > 1e-4:
    raise RuntimeError("camera.fov setter did not stick")
camera.fov = old_fov
log("camera.fov setter round-trip OK")


# ---------------------------------------------------------------------------
# Walk every entity in the scene
# ---------------------------------------------------------------------------
all_entities = scene.view_all_entities()
log(f"scene has {len(all_entities)} named entities:")
for e in all_entities:
    name = e.get_NameComponent().name if e.has_NameComponent() else "<unnamed>"
    parts = [f"  - id={e.id} name='{name}'"]

    if e.has_TransformComponent():
        t = e.get_TransformComponent()
        parts.append(f"pos={t.position}")

    if e.has_PointLightComponent():
        pl = e.get_PointLightComponent()
        parts.append(f"PointLight(intensity={pl.intensity:.2f})")

    if e.has_DirectionalLightComponent():
        dl = e.get_DirectionalLightComponent()
        parts.append(f"DirLight(intensity={dl.intensity:.2f})")

    if e.has_SpotLightComponent():
        sl = e.get_SpotLightComponent()
        parts.append(f"SpotLight(intensity={sl.intensity:.2f})")

    if e.has_RigidBodyComponent():
        rb = e.get_RigidBodyComponent()
        type_name = {BodyType.Static: "Static", BodyType.Dynamic: "Dynamic",
                     BodyType.Kinematic: "Kinematic"}.get(rb.body_type, "?")
        parts.append(f"RigidBody({type_name}, mass={rb.mass:.2f})")

    log(" | ".join(parts))


# ---------------------------------------------------------------------------
# Root entity / parent walk — exercises HierarchyComponent indirectly
# ---------------------------------------------------------------------------
roots = scene.root_entities()
log(f"scene has {len(roots)} root entities")
for r in roots:
    nm = r.get_NameComponent().name if r.has_NameComponent() else "<unnamed>"
    children = r.children
    log(f"  root '{nm}' -> {len(children)} child(ren)")


# ---------------------------------------------------------------------------
# Read/write round-trip via Entity handle, then revert
# ---------------------------------------------------------------------------
# Pick the first entity that has both Name and Transform — guaranteed to exist
# in the default scene (every editor entity is created with both).
target = None
for e in all_entities:
    if e.has_NameComponent() and e.has_TransformComponent():
        target = e
        break

if target is not None:
    t = target.get_TransformComponent()
    saved = vec3(t.position.x, t.position.y, t.position.z)
    t.position = vec3(saved.x + 1.0, saved.y, saved.z)
    if abs(target.get_TransformComponent().position.x - (saved.x + 1.0)) > 1e-4:
        raise RuntimeError("TransformComponent.position write did not propagate via reference")
    # Revert so the on-screen scene is untouched after this script runs.
    target.get_TransformComponent().position = saved
    log(f"position write round-trip OK on '{target.get_NameComponent().name}'")
else:
    log("no entity with Name+Transform found — skipping write round-trip")


# ---------------------------------------------------------------------------
# find_by_name on a guaranteed-missing name returns None (not raise)
# ---------------------------------------------------------------------------
ghost = scene.find_by_name("__definitely_not_an_entity__")
if ghost is not None:
    raise RuntimeError("find_by_name returned a hit for a bogus name")
log("find_by_name(missing) -> None OK")


# ---------------------------------------------------------------------------
# Create + destroy round-trip (does NOT leave residue in the scene)
# ---------------------------------------------------------------------------
tmp = scene.create_entity("__phase_13_3_temp__")
if not tmp:
    raise RuntimeError("create_entity returned an invalid handle")
if not tmp.has_TransformComponent():
    raise RuntimeError("create_entity did not auto-attach TransformComponent")
tmp_id = tmp.id
tmp.destroy()
if tmp:
    raise RuntimeError("Entity.destroy() did not invalidate the handle")
if scene.find_by_name("__phase_13_3_temp__") is not None:
    raise RuntimeError("Destroyed entity still discoverable via find_by_name")
log(f"create_entity / destroy round-trip OK (id was {tmp_id})")


log("Phase 13.3 ECS handle verification PASSED")
