# ============================================================================
# verify_bindings.py - Phase 13.2 smoke test for glm / Component bindings.
# ----------------------------------------------------------------------------
# Exercises every type registered by Bindings_Glm.cpp + Bindings_Components.cpp
# and writes the results through engine.log(). Run by replacing smoke.py in
# DeferredRenderer::init() temporarily, or load it from the future Script
# Console panel (Phase 13.6).
# ============================================================================

from engine import (
    log,
    vec3,
    vec4,
    quat,
    mat4,
    NameComponent,
    TransformComponent,
    PointLightComponent,
    DirectionalLightComponent,
    SpotLightComponent,
    MaterialComponent,
    PhysicsMaterial,
    RigidBodyComponent,
    BoxColliderComponent,
    BodyType,
)


def assert_close(name, got, expected, eps=1e-4):
    """Tiny helper - we don't want to depend on the unittest module."""
    diff = abs(got - expected)
    if diff > eps:
        log(f"FAIL  {name}: got {got}, expected {expected} (diff={diff})")
        raise AssertionError(name)


# ---------------------------------------------------------------------------
# vec3 arithmetic + helpers
# ---------------------------------------------------------------------------
a = vec3(1.0, 2.0, 3.0)
b = vec3(4.0, 5.0, 6.0)

assert_close("a.x", a.x, 1.0)
assert_close("a[1]", a[1], 2.0)
assert_close("dot",  a.dot(b),    1*4 + 2*5 + 3*6)
assert_close("len",  vec3(3, 4, 0).length(), 5.0)

c = a + b
assert_close("add.x", c.x, 5.0)
assert_close("add.y", c.y, 7.0)

d = a * 2.0
assert_close("scale.z", d.z, 6.0)

n = vec3(0.0, 0.0, 0.0).normalize()
assert_close("normalize-zero.x", n.x, 0.0)  # safe path: zero -> zero, not NaN

log(f"vec3 OK: a={a} b={b} a+b={c} 2*a={d}")

# ---------------------------------------------------------------------------
# vec4 / quat / mat4 read-only smoke
# ---------------------------------------------------------------------------
v4 = vec4(1.0, 2.0, 3.0, 4.0)
assert_close("vec4[3]", v4[3], 4.0)
log(f"vec4 OK: {v4}")

q = quat(1.0, 0.0, 0.0, 0.0)  # identity (w, x, y, z)
assert_close("quat.w", q.w, 1.0)
log(f"quat OK: {q}")

m = mat4(1.0)  # identity
assert_close("mat4 diag (0,0)", m.get(0, 0), 1.0)
assert_close("mat4 off (0,1)",  m.get(0, 1), 0.0)
log("mat4 OK: identity diagonal verified")

# ---------------------------------------------------------------------------
# Components — construct + round-trip
# ---------------------------------------------------------------------------
nc = NameComponent()
nc.name = "PythonCube"
log(f"NameComponent OK: {nc}")

t = TransformComponent()
t.position = vec3(1.0, 2.0, 3.0)
t.rotation = vec3(0.0, 90.0, 0.0)  # setter must sync orientation
t.scale = vec3(2.0, 2.0, 2.0)
assert_close("transform.position.y", t.position.y, 2.0)
assert_close("transform.rotation.y", t.rotation.y, 90.0)
# orientation is read-only; just confirm we can read it without crashing
_ = t.orientation
_ = t.world_matrix
log(f"TransformComponent OK: {t}")

pl = PointLightComponent()
pl.color = vec3(1.0, 0.5, 0.25)
pl.intensity = 5.0
pl.cast_shadows = True
log(f"PointLightComponent OK: {pl}")

dl = DirectionalLightComponent()
dl.intensity = 1.5
dl.cast_shadows = False
log(f"DirectionalLightComponent OK: {dl}")

sl = SpotLightComponent()
sl.inner_angle = 10.0
sl.outer_angle = 25.0
log(f"SpotLightComponent OK: {sl}")

mat = MaterialComponent()
mat.albedo = vec3(0.7, 0.2, 0.2)
mat.metallic = 0.0
mat.roughness = 0.4
log(f"MaterialComponent OK: {mat}")

pm = PhysicsMaterial()
pm.friction = 0.8
pm.restitution = 0.1
log(f"PhysicsMaterial OK: {pm}")

rb = RigidBodyComponent()
rb.body_type = BodyType.Dynamic
rb.mass = 2.5
rb.velocity = vec3(0.0, 0.0, 1.0)
log(f"RigidBodyComponent OK: {rb}")

bc = BoxColliderComponent()
bc.center = vec3(0.0, 0.5, 0.0)
bc.half_extents = vec3(0.5, 0.5, 0.5)
bc.material = pm  # full assignment of nested PhysicsMaterial
log(f"BoxColliderComponent OK: {bc}")

log("Phase 13.2 binding verification PASSED")
