// Bindings_Ecs — pybind11 bindings for Entity / Scene / Camera handles.
//
// Entity wraps {entt::entity, ECSScene*} rather than binding entt::entity
// directly: a raw id loses the scene context, and threading ECSScene through
// every binding lambda would make the API awkward. Handles never cache
// component references; every call routes through the registry, so they stay
// valid across registry reallocations.
//
// HierarchyComponent is only reachable via Scene.set_parent / Entity.parent /
// Entity.children so scripts cannot desync the parent/child invariants
// destroyEntity relies on. MeshComponent (Vulkan buffers via shared_ptr) is
// not bound at all — Python must not touch GPU resources, and sourcePath is
// already covered by serialisation.
//
// The embedded module is registered once at interpreter startup, but the
// `engine.scene` / `engine.camera` attributes are only bound at
// setSceneContext() time, so `from engine import scene` yields None until
// the renderer has finished init().

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "ecs/Components.h"
#include "ecs/ECSScene.h"
#include "physics/PhysicsComponents.h"
#include "scene/Camera.h"

#include <optional>
#include <sstream>
#include <string>

namespace py = pybind11;

// ============================================================================
// Entity — a {entity_id, scene*} pair surfaced to Python
// ----------------------------------------------------------------------------
// We DO NOT cache Component references inside the handle. Every call goes
// through `scene_->registry`, so the handle stays valid across registry
// reallocations and avoids dangling-pointer pitfalls.
// ============================================================================
namespace
{
    struct PyEntity
    {
        entt::entity id = entt::null;
        ECSScene *scene = nullptr;

        bool valid() const
        {
            return scene != nullptr && id != entt::null && scene->registry.valid(id);
        }
    };

    // Resolve PyEntity into a registry reference, throwing a Python-style
    // RuntimeError on invalid handles. Centralised so every component
    // accessor produces a consistent error.
    entt::registry &reg_or_raise(const PyEntity &e)
    {
        if (!e.scene)
            throw std::runtime_error("Entity has no scene (default-constructed handle?)");
        if (e.id == entt::null)
            throw std::runtime_error("Entity is null");
        if (!e.scene->registry.valid(e.id))
            throw std::runtime_error("Entity handle refers to a destroyed entity");
        return e.scene->registry;
    }
} // namespace

// Registers has_/get_/add_/remove_<Name>() for one component type on Entity.
// get_/add_ return T by reference so Python edits propagate into the registry.
// A templated helper is used instead of a preprocessor macro so
// jump-to-definition still works, compiler errors point at the binding site
// rather than a macro expansion, and adding a component is a one-line call.
namespace
{
    template <typename T>
    void registerComponentApi(py::class_<PyEntity> &cls, const char *suffix)
    {
        const std::string suf = suffix;

        cls.def(
            ("has_" + suf).c_str(),
            [](const PyEntity &e)
            {
                return e.valid() && e.scene->registry.all_of<T>(e.id);
            },
            ("Returns True if the entity carries a " + suf + ".").c_str());

        // get_X — return by reference so Python edits propagate back into
        // the registry (return_value_policy::reference with T bound as a
        // non-pointer).
        cls.def(
            ("get_" + suf).c_str(),
            [](PyEntity &e) -> T &
            {
                auto &reg = reg_or_raise(e);
                if (!reg.all_of<T>(e.id))
                    throw std::runtime_error(
                        std::string("Entity does not have a ") +
                        typeid(T).name() + ". Use has_*() to check first.");
                return reg.get<T>(e.id);
            },
            py::return_value_policy::reference_internal,
            ("Return the live " + suf + " reference. Edits propagate to the ECS.").c_str());

        cls.def(
            ("add_" + suf).c_str(),
            [](PyEntity &e, const T &value) -> T &
            {
                auto &reg = reg_or_raise(e);
                // emplace_or_replace mirrors what users expect from
                // "add" semantics in Unity-like APIs (idempotent attach).
                return reg.emplace_or_replace<T>(e.id, value);
            },
            py::arg("value") = T{},
            py::return_value_policy::reference_internal,
            ("Attach (or replace) a " + suf + " on the entity.").c_str());

        cls.def(
            ("remove_" + suf).c_str(),
            [](PyEntity &e)
            {
                auto &reg = reg_or_raise(e);
                if (reg.all_of<T>(e.id))
                    reg.remove<T>(e.id);
            },
            ("Detach the " + suf + " from the entity (no-op if absent).").c_str());
    }
} // namespace

// ============================================================================
// Scene wrapper — operates on a non-owning ECSScene*
// ============================================================================
namespace
{
    struct PyScene
    {
        ECSScene *scene = nullptr;

        ECSScene &raise_if_null() const
        {
            if (!scene)
                throw std::runtime_error("engine.scene is not bound. Call from inside a script load / on_*() hook.");
            return *scene;
        }
    };

    struct PyCamera
    {
        Camera *camera = nullptr;

        Camera &raise_if_null() const
        {
            if (!camera)
                throw std::runtime_error("engine.camera is not bound.");
            return *camera;
        }
    };
} // namespace

// ============================================================================
// registerBindings_Ecs — called from ScriptEngine's embedded module
// ============================================================================
void registerBindings_Ecs(py::module_ &m)
{
    // ------------------------------------------------------------------
    // Entity
    // ------------------------------------------------------------------
    py::class_<PyEntity> entityCls(m, "Entity",
                                   "Lightweight ECS handle. Equivalent to a (registry, entity_id) pair.");

    entityCls
        .def(py::init<>(), "Default-construct an invalid handle (id=null).")
        .def("__bool__", [](const PyEntity &e)
             { return e.valid(); }, "True iff the underlying entt::entity is still alive in its registry.")
        .def("__repr__", [](const PyEntity &e)
             {
                 std::ostringstream oss;
                 oss << "Entity(";
                 if (!e.scene)
                     oss << "no_scene";
                 else if (e.id == entt::null)
                     oss << "null";
                 else if (!e.scene->registry.valid(e.id))
                     oss << "destroyed:#" << static_cast<uint32_t>(e.id);
                 else
                     oss << "id=" << static_cast<uint32_t>(e.id);
                 oss << ")";
                 return oss.str(); })
        .def("__eq__", [](const PyEntity &a, const PyEntity &b)
             { return a.id == b.id && a.scene == b.scene; })
        .def("__ne__", [](const PyEntity &a, const PyEntity &b)
             { return !(a.id == b.id && a.scene == b.scene); })
        .def("__hash__", [](const PyEntity &e)
             { return static_cast<py::ssize_t>(static_cast<uint32_t>(e.id)); })
        .def_property_readonly("id", [](const PyEntity &e)
                               { return static_cast<uint32_t>(e.id); }, "Underlying entt entity id (unsigned 32-bit). Useful for logging only.")
        .def("destroy", [](PyEntity &e)
             {
                if (!e.valid())
                    return;
                // Goes through ECSScene::destroyEntity, which already fires
                // onBeforeDestroyEntity (PhysicsSystem::unregisterEntity hook)
                // and recursively cleans up children. After this call the
                // handle becomes invalid; subsequent get_*() calls will raise.
                e.scene->destroyEntity(e.id);
                e.id = entt::null; }, "Destroy this entity (and its children). Hooks notify physics / audio.")
        .def_property_readonly("parent", [](const PyEntity &e) -> std::optional<PyEntity>
                               {
                if (!e.valid())
                    return std::nullopt;
                if (!e.scene->registry.all_of<HierarchyComponent>(e.id))
                    return std::nullopt;
                const auto &hc = e.scene->registry.get<HierarchyComponent>(e.id);
                if (hc.parent == entt::null)
                    return std::nullopt;
                return PyEntity{hc.parent, e.scene}; }, "Parent entity, or None if this is a root entity.")
        .def_property_readonly("children", [](const PyEntity &e)
                               {
                std::vector<PyEntity> out;
                if (!e.valid())
                    return out;
                if (!e.scene->registry.all_of<HierarchyComponent>(e.id))
                    return out;
                const auto &hc = e.scene->registry.get<HierarchyComponent>(e.id);
                out.reserve(hc.children.size());
                for (auto child : hc.children)
                    out.push_back(PyEntity{child, e.scene});
                return out; }, "List of immediate child entities (may be empty).");

    // Component-by-component method registration. Adding a new Component is
    // exactly one line here. Order kept consistent with Bindings_Components.cpp.
    registerComponentApi<NameComponent>(entityCls, "NameComponent");
    registerComponentApi<TransformComponent>(entityCls, "TransformComponent");
    registerComponentApi<PointLightComponent>(entityCls, "PointLightComponent");
    registerComponentApi<DirectionalLightComponent>(entityCls, "DirectionalLightComponent");
    registerComponentApi<SpotLightComponent>(entityCls, "SpotLightComponent");
    registerComponentApi<MaterialComponent>(entityCls, "MaterialComponent");
    registerComponentApi<RigidBodyComponent>(entityCls, "RigidBodyComponent");
    registerComponentApi<BoxColliderComponent>(entityCls, "BoxColliderComponent");

    // ------------------------------------------------------------------
    // Scene
    // ------------------------------------------------------------------
    py::class_<PyScene> sceneCls(m, "Scene", "ECS scene handle (non-owning view of the engine's ECSScene).");
    sceneCls.def(py::init<>());
    sceneCls.def("__bool__", [](const PyScene &s)
                 { return s.scene != nullptr; });
    sceneCls.def(
        "create_entity",
        [](PyScene &s, const std::string &name)
        {
            auto &scn = s.raise_if_null();
            return PyEntity{scn.createEntity(name), &scn};
        },
        py::arg("name") = std::string("Entity"));
    sceneCls.def(
        "destroy_entity",
        [](PyScene &s, PyEntity &e)
        {
            auto &scn = s.raise_if_null();
            if (e.scene != &scn)
                throw std::runtime_error("Entity belongs to a different scene.");
            if (e.valid())
                scn.destroyEntity(e.id);
            e.id = entt::null;
        },
        py::arg("entity"));
    sceneCls.def(
        "set_parent",
        [](PyScene &s, PyEntity &child, const py::object &maybeParent)
        {
            auto &scn = s.raise_if_null();
            if (!child.valid())
                throw std::runtime_error("Child entity is invalid.");

            entt::entity parentId = entt::null;
            if (!maybeParent.is_none())
            {
                auto p = py::cast<PyEntity>(maybeParent);
                if (!p.valid())
                    throw std::runtime_error("Parent entity is invalid.");
                parentId = p.id;
            }
            scn.setParent(child.id, parentId);
        },
        py::arg("child"), py::arg("parent"));
    sceneCls.def_property(
        "selected_entity",
        [](const PyScene &s) -> std::optional<PyEntity>
        {
            auto &scn = s.raise_if_null();
            if (scn.selectedEntity == entt::null)
                return std::nullopt;
            return PyEntity{scn.selectedEntity, &scn};
        },
        [](PyScene &s, const py::object &val)
        {
            auto &scn = s.raise_if_null();
            if (val.is_none())
            {
                scn.selectedEntity = entt::null;
            }
            else
            {
                auto e = py::cast<PyEntity>(val);
                if (e.scene != &scn)
                    throw std::runtime_error("Entity belongs to a different scene.");
                scn.selectedEntity = e.id;
            }
        });
    sceneCls.def(
        "root_entities",
        [](PyScene &s)
        {
            auto &scn = s.raise_if_null();
            std::vector<PyEntity> out;
            auto roots = scn.getRootEntities();
            out.reserve(roots.size());
            for (auto e : roots)
                out.push_back(PyEntity{e, &scn});
            return out;
        });
    sceneCls.def(
        "view_all_entities",
        [](PyScene &s)
        {
            auto &scn = s.raise_if_null();
            std::vector<PyEntity> out;
            auto view = scn.registry.view<NameComponent>();
            for (auto e : view)
                out.push_back(PyEntity{e, &scn});
            return out;
        });
    sceneCls.def(
        "find_by_name",
        [](PyScene &s, const std::string &name) -> std::optional<PyEntity>
        {
            auto &scn = s.raise_if_null();
            auto view = scn.registry.view<NameComponent>();
            for (auto e : view)
            {
                if (view.get<NameComponent>(e).name == name)
                    return PyEntity{e, &scn};
            }
            return std::nullopt;
        },
        py::arg("name"));
    sceneCls.def("mark_lights_dirty", [](PyScene &) {});
    sceneCls.def("sync_lights_to_gpu", [](PyScene &) {});

    // ------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------
    py::class_<PyCamera> camCls(m, "Camera", "View / projection camera (non-owning).");
    camCls.def(py::init<>());
    camCls.def("__bool__", [](const PyCamera &c)
               { return c.camera != nullptr; });
    camCls.def_property_readonly(
        "position",
        [](const PyCamera &c)
        { return c.raise_if_null().getPosition(); });
    camCls.def_property(
        "fov",
        [](const PyCamera &c)
        { return c.raise_if_null().getFov(); },
        [](PyCamera &c, float v)
        { c.raise_if_null().setFov(v); });
    camCls.def_property(
        "near_plane",
        [](const PyCamera &c)
        { return c.raise_if_null().getNearPlane(); },
        [](PyCamera &c, float v)
        { c.raise_if_null().setNearPlane(v); });
    camCls.def_property(
        "far_plane",
        [](const PyCamera &c)
        { return c.raise_if_null().getFarPlane(); },
        [](PyCamera &c, float v)
        { c.raise_if_null().setFarPlane(v); });
}

// Bindings_Ecs_setContext — host-side helpers used by ScriptEngine to wire
// the singleton `engine.scene` / `engine.camera` / `engine.delta_time`
// attributes. Declared here (same TU as PyScene/PyCamera) so the POD
// wrappers stay out of any header. These are NOT registered with pybind11 —
// they're plain C++ helpers called from ScriptEngine.cpp under the GIL.
namespace pybindings_ecs_internal
{
    // Wrap the raw pointers in our POD wrappers and stash them on the
    // `engine` module. After this returns, scripts can do
    // `from engine import scene` and `scene.find_by_name(...)`.
    void publishContext(py::module_ &engineModule, ECSScene *scene, Camera *camera)
    {
        engineModule.attr("scene") = py::cast(PyScene{scene});
        engineModule.attr("camera") = py::cast(PyCamera{camera});
        // delta_time gets its first concrete value when callOnUpdate runs.
        // We seed it to 0.0 here so `import engine; engine.delta_time` is
        // not undefined at import time.
        engineModule.attr("delta_time") = py::cast(0.0f);
    }

    void publishDeltaTime(py::module_ &engineModule, float dt)
    {
        engineModule.attr("delta_time") = py::cast(dt);
    }

    void clearContext(py::module_ &engineModule)
    {
        // Replace with non-bound wrappers so any lingering user-side
        // reference fails loudly with our raise_if_null message instead of
        // segfaulting on a stale pointer.
        engineModule.attr("scene") = py::cast(PyScene{});
        engineModule.attr("camera") = py::cast(PyCamera{});
        engineModule.attr("delta_time") = py::cast(0.0f);
    }

    // ScriptEngine.cpp needs to inject `self` (a PyEntity) into each
    // per-entity script's globals dict before invoking on_start / on_update /
    // on_stop. PyEntity lives in this TU's anonymous namespace, so this thin
    // wrapper avoids leaking the type through a header. The returned object
    // is a fresh py::cast of a new PyEntity{id, scene}; subsequent registry
    // mutations stay visible because PyEntity's methods route through `scene`.
    py::object makeEntityHandle(ECSScene *scene, entt::entity id)
    {
        return py::cast(PyEntity{id, scene});
    }
} // namespace pybindings_ecs_internal
