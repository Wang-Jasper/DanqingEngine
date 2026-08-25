// Bindings_Components — pybind11 bindings for the ECS component types.
// All numeric fields are read-write; texture handles and internal physics
// indices are intentionally not exposed (see the individual classes below).

#include <pybind11/pybind11.h>

#include "ecs/Components.h"
#include "physics/PhysicsComponents.h"
#include "physics/PhysicsMaterial.h"
#include "physics/PhysicsWorld.h" // BodyType enum

#include <sstream>

namespace py = pybind11;

namespace
{
    // Used by the Light components' __repr__ helpers — we keep it private
    // here rather than sharing across TUs because the formatting choices
    // are component-specific.
    template <typename T>
    std::string fmtF(T v)
    {
        std::ostringstream oss;
        oss.precision(3);
        oss << std::fixed << v;
        return oss.str();
    }
} // namespace

// ============================================================================
// registerBindings_Components — called from ScriptEngine's embedded module
// ============================================================================
void registerBindings_Components(py::module_ &m)
{
    // ------------------------------------------------------------------
    // BodyType enum
    // ------------------------------------------------------------------
    py::enum_<BodyType>(m, "BodyType",
                        "Rigid body type. Matches physics/PhysicsWorld.h.")
        .value("Static", BodyType::Static)
        .value("Dynamic", BodyType::Dynamic)
        .value("Kinematic", BodyType::Kinematic);

    // ------------------------------------------------------------------
    // NameComponent
    // ------------------------------------------------------------------
    py::class_<NameComponent>(m, "NameComponent",
                              "Editor-visible name string for an entity.")
        .def(py::init<>())
        .def_readwrite("name", &NameComponent::name)
        .def("__repr__",
             [](const NameComponent &n)
             {
                 return "NameComponent('" + n.name + "')";
             });

    // ------------------------------------------------------------------
    // TransformComponent — the most-used scriptable component
    // ------------------------------------------------------------------
    // Setter for `rotation` MUST keep `orientation` in sync, otherwise the
    // next TransformSystem pass will use a stale quaternion (the ECS uses
    // orientation as the source of truth when computing worldMatrix).
    // We expose `rotation` as a property so the setter runs every assignment.
    py::class_<TransformComponent>(m, "TransformComponent",
                                   "Position / rotation (Euler degrees) / scale + cached worldMatrix.")
        .def(py::init<>())
        .def_readwrite("position", &TransformComponent::position)
        .def_property(
            "rotation",
            [](const TransformComponent &t)
            { return t.rotation; },
            [](TransformComponent &t, const glm::vec3 &eulerDegrees)
            {
                // Funnel through the existing helper so orientation stays
                // consistent. Doing `t.rotation = v` directly from C++
                // would have skipped this — we don't expose that footgun.
                t.setEulerRotation(eulerDegrees);
            },
            "Euler XYZ rotation in degrees. Setter syncs the underlying quaternion.")
        .def_readwrite("scale", &TransformComponent::scale)
        // worldMatrix is computed by Systems::updateTransforms from TRS;
        // exposing it writable would let scripts produce a stale matrix
        // that survives exactly until the next transform pass. Read-only.
        .def_readonly("world_matrix", &TransformComponent::worldMatrix)
        .def_readonly("orientation", &TransformComponent::orientation,
                      "Read-only quaternion. Edit via 'rotation' (Euler).")
        .def("set_euler_rotation",
             &TransformComponent::setEulerRotation,
             py::arg("euler_degrees"),
             "Same as assigning to .rotation; provided for explicit calls.")
        .def("__repr__",
             [](const TransformComponent &t)
             {
                 std::ostringstream oss;
                 oss << "TransformComponent(pos=("
                     << fmtF(t.position.x) << "," << fmtF(t.position.y) << ","
                     << fmtF(t.position.z) << "), rot=("
                     << fmtF(t.rotation.x) << "," << fmtF(t.rotation.y) << ","
                     << fmtF(t.rotation.z) << "), scale=("
                     << fmtF(t.scale.x) << "," << fmtF(t.scale.y) << ","
                     << fmtF(t.scale.z) << "))";
                 return oss.str();
             });

    // ------------------------------------------------------------------
    // Light components — all numeric fields are rw.
    // ------------------------------------------------------------------
    // NOTE: editing these from a script does NOT flush the GPU light SSBO;
    // the renderer re-gathers all light components every frame in
    // updateLightBuffers(), so a Python-side edit is visible the very next
    // frame without manual mark_lights_dirty calls. (mark_lights_dirty is
    // still surfaced for back-compat with old code paths that expect a
    // dirty flag.)
    py::class_<PointLightComponent>(m, "PointLightComponent",
                                    "Point light parameters (position from TransformComponent).")
        .def(py::init<>())
        .def_readwrite("color", &PointLightComponent::color)
        .def_readwrite("intensity", &PointLightComponent::intensity)
        .def_readwrite("radius", &PointLightComponent::radius)
        .def_readwrite("cast_shadows", &PointLightComponent::castShadows)
        .def_readwrite("shadow_slope_bias", &PointLightComponent::shadowSlopeBias)
        .def_readwrite("shadow_constant_bias", &PointLightComponent::shadowConstantBias)
        .def_readwrite("shadow_normal_bias", &PointLightComponent::shadowNormalBias)
        .def_readwrite("shadow_depth_bias", &PointLightComponent::shadowDepthBias)
        .def("__repr__",
             [](const PointLightComponent &l)
             {
                 std::ostringstream oss;
                 oss << "PointLightComponent(intensity=" << fmtF(l.intensity)
                     << ", radius=" << fmtF(l.radius)
                     << ", cast_shadows=" << (l.castShadows ? "True" : "False") << ")";
                 return oss.str();
             });

    py::class_<DirectionalLightComponent>(m, "DirectionalLightComponent",
                                          "Directional light parameters (direction from TransformComponent.rotation).")
        .def(py::init<>())
        .def_readwrite("color", &DirectionalLightComponent::color)
        .def_readwrite("intensity", &DirectionalLightComponent::intensity)
        .def_readwrite("cast_shadows", &DirectionalLightComponent::castShadows)
        .def("__repr__",
             [](const DirectionalLightComponent &l)
             {
                 std::ostringstream oss;
                 oss << "DirectionalLightComponent(intensity=" << fmtF(l.intensity)
                     << ", cast_shadows=" << (l.castShadows ? "True" : "False") << ")";
                 return oss.str();
             });

    py::class_<SpotLightComponent>(m, "SpotLightComponent",
                                   "Spotlight parameters (pos+dir from TransformComponent).")
        .def(py::init<>())
        .def_readwrite("color", &SpotLightComponent::color)
        .def_readwrite("intensity", &SpotLightComponent::intensity)
        .def_readwrite("radius", &SpotLightComponent::radius)
        .def_readwrite("inner_angle", &SpotLightComponent::innerAngle)
        .def_readwrite("outer_angle", &SpotLightComponent::outerAngle)
        .def_readwrite("cast_shadows", &SpotLightComponent::castShadows)
        .def("__repr__",
             [](const SpotLightComponent &l)
             {
                 std::ostringstream oss;
                 oss << "SpotLightComponent(intensity=" << fmtF(l.intensity)
                     << ", inner=" << fmtF(l.innerAngle) << "deg, outer="
                     << fmtF(l.outerAngle) << "deg, cast_shadows="
                     << (l.castShadows ? "True" : "False") << ")";
                 return oss.str();
             });

    // ------------------------------------------------------------------
    // MaterialComponent — only the PBR scalar/colour fields. albedoTexture
    // (and the descriptor set behind it) is deliberately not surfaced:
    //   * scripts have no business poking VkDescriptorSet handles
    //   * texture GUID serialisation is still TODO
    // ------------------------------------------------------------------
    py::class_<MaterialComponent>(m, "MaterialComponent",
                                  "PBR Cook-Torrance material parameters.")
        .def(py::init<>())
        .def_readwrite("albedo", &MaterialComponent::albedo)
        .def_readwrite("metallic", &MaterialComponent::metallic)
        .def_readwrite("roughness", &MaterialComponent::roughness)
        .def("__repr__",
             [](const MaterialComponent &mat)
             {
                 std::ostringstream oss;
                 oss << "MaterialComponent(albedo=("
                     << fmtF(mat.albedo.x) << "," << fmtF(mat.albedo.y) << ","
                     << fmtF(mat.albedo.z) << "), metallic=" << fmtF(mat.metallic)
                     << ", roughness=" << fmtF(mat.roughness) << ")";
                 return oss.str();
             });

    // ------------------------------------------------------------------
    // PhysicsMaterial — embedded value type for BoxColliderComponent.material
    // ------------------------------------------------------------------
    // Cook-style "PBR for friction/restitution" — registering it lets
    // scripts read collider.material.friction etc. We only expose the
    // fields actually used by the solver today; combine_mode / threshold
    // and other future fields stay C++-private until they are stable.
    py::class_<PhysicsMaterial>(m, "PhysicsMaterial",
                                "Per-collider physics material (friction / restitution).")
        .def(py::init<>())
        .def_readwrite("friction", &PhysicsMaterial::friction)
        .def_readwrite("restitution", &PhysicsMaterial::restitution)
        .def("__repr__",
             [](const PhysicsMaterial &pm)
             {
                 std::ostringstream oss;
                 oss << "PhysicsMaterial(friction=" << fmtF(pm.friction)
                     << ", restitution=" << fmtF(pm.restitution) << ")";
                 return oss.str();
             });

    // ------------------------------------------------------------------
    // RigidBodyComponent — body-level state.
    // ------------------------------------------------------------------
    // We deliberately do NOT expose physicsIndex: it's a runtime mapping
    // into PhysicsWorld.bodies_ and a script poking it could desync the
    // ECS<->physics binding. Higher-level helpers (apply_force /
    // set_velocity on Entity) can be added later if needed.
    py::class_<RigidBodyComponent>(m, "RigidBodyComponent",
                                   "ECS rigid body component.")
        .def(py::init<>())
        .def_readwrite("body_type", &RigidBodyComponent::bodyType)
        .def_readwrite("mass", &RigidBodyComponent::mass)
        .def_readwrite("velocity", &RigidBodyComponent::velocity)
        .def_readwrite("layer", &RigidBodyComponent::layer)
        .def_readwrite("mask", &RigidBodyComponent::mask)
        .def("__repr__",
             [](const RigidBodyComponent &rb)
             {
                 const char *typeStr = "Dynamic";
                 if (rb.bodyType == BodyType::Static)
                     typeStr = "Static";
                 else if (rb.bodyType == BodyType::Kinematic)
                     typeStr = "Kinematic";
                 std::ostringstream oss;
                 oss << "RigidBodyComponent(type=" << typeStr
                     << ", mass=" << fmtF(rb.mass)
                     << ", velocity=(" << fmtF(rb.velocity.x) << ","
                     << fmtF(rb.velocity.y) << "," << fmtF(rb.velocity.z) << "))";
                 return oss.str();
             });

    // ------------------------------------------------------------------
    // BoxColliderComponent — single-collider simplified entry point
    // ------------------------------------------------------------------
    py::class_<BoxColliderComponent>(m, "BoxColliderComponent",
                                     "Single-Box collider (PhysicsSystem treats this as a 1-element CollidersComponent).")
        .def(py::init<>())
        .def_readwrite("center", &BoxColliderComponent::center)
        .def_readwrite("half_extents", &BoxColliderComponent::halfExtents)
        .def_readwrite("material", &BoxColliderComponent::material)
        .def("__repr__",
             [](const BoxColliderComponent &c)
             {
                 std::ostringstream oss;
                 oss << "BoxColliderComponent(center=("
                     << fmtF(c.center.x) << "," << fmtF(c.center.y) << ","
                     << fmtF(c.center.z) << "), half_extents=("
                     << fmtF(c.halfExtents.x) << "," << fmtF(c.halfExtents.y) << ","
                     << fmtF(c.halfExtents.z) << "))";
                 return oss.str();
             });
}
