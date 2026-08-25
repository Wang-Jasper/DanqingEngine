#pragma once

// Serializes/deserializes ECS scenes to JSON.
// Serialized: Name, Transform, Hierarchy, MeshComponent(sourcePath), lights,
// RigidBody, BoxCollider, Script, Colliders. Not serialized: GPU resource
// handles, worldMatrix (computed at runtime), selectedEntity (editor state).
//
// Asset references are stored as GUIDs, not paths. The JSON field `mesh.source`
// carries one of:
//   - "builtin:cube" / "builtin:plane"   : engine built-in meshes
//   - "guid:<uuid>"                      : GUID of a tracked asset
//   - "guid:<uuid>#<subMeshName>"        : GUID + sub-mesh selector (multi-material OBJ)
// Legacy scene files storing absolute / relative paths are NOT supported.

#include <nlohmann/json.hpp>
#include <entt/entt.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>

#include "core/asset/AssetRegistry.h"
#include "ecs/Components.h"
#include "ecs/ECSScene.h"
#include "physics/PhysicsComponents.h"

using json = nlohmann::json;

namespace SceneSerializer
{

    // Serializes the ECS scene to a JSON file.
    inline bool saveScene(ECSScene &scene, const std::string &filepath)
    {
        auto &reg = scene.registry;
        json root;
        root["version"] = 1;

        // Collect all entities via NameComponent (createEntity always attaches it).
        auto allEntities = reg.view<NameComponent>();
        std::unordered_map<entt::entity, int> entityToId;
        int nextId = 0;
        for (auto e : allEntities)
        {
            entityToId[e] = nextId++;
        }

        json entities = json::array();

        for (auto e : allEntities)
        {
            json entityJson;
            entityJson["id"] = entityToId[e];

            if (reg.all_of<NameComponent>(e))
            {
                entityJson["name"] = reg.get<NameComponent>(e).name;
            }

            if (reg.all_of<TransformComponent>(e))
            {
                auto &tc = reg.get<TransformComponent>(e);
                entityJson["transform"] = {
                    {"position", {tc.position.x, tc.position.y, tc.position.z}},
                    {"rotation", {tc.rotation.x, tc.rotation.y, tc.rotation.z}},
                    {"scale", {tc.scale.x, tc.scale.y, tc.scale.z}}};
            }

            if (reg.all_of<HierarchyComponent>(e))
            {
                auto &hc = reg.get<HierarchyComponent>(e);
                json hierJson;
                if (hc.parent != entt::null && entityToId.count(hc.parent))
                {
                    hierJson["parent"] = entityToId[hc.parent];
                }
                else
                {
                    hierJson["parent"] = nullptr;
                }
                json childIds = json::array();
                for (auto child : hc.children)
                {
                    if (entityToId.count(child))
                    {
                        childIds.push_back(entityToId[child]);
                    }
                }
                hierJson["children"] = childIds;
                entityJson["hierarchy"] = hierJson;
            }

            // sourcePath is GUID-form (builtin:* stays as-is; guid:<uuid>[#sub] is set by the import pipeline).
            if (reg.all_of<MeshComponent>(e))
            {
                auto &mc = reg.get<MeshComponent>(e);
                entityJson["mesh"] = {{"source", mc.sourcePath}};
            }

            if (reg.all_of<PointLightComponent>(e))
            {
                auto &plc = reg.get<PointLightComponent>(e);
                entityJson["pointLight"] = {
                    {"color", {plc.color.x, plc.color.y, plc.color.z}},
                    {"intensity", plc.intensity},
                    {"radius", plc.radius},
                    {"castShadows", plc.castShadows},
                    // Unity-style per-light shadow bias.
                    {"shadowSlopeBias", plc.shadowSlopeBias},
                    {"shadowConstantBias", plc.shadowConstantBias},
                    {"shadowNormalBias", plc.shadowNormalBias},
                    {"shadowDepthBias", plc.shadowDepthBias}};
            }

            if (reg.all_of<DirectionalLightComponent>(e))
            {
                auto &dlc = reg.get<DirectionalLightComponent>(e);
                entityJson["dirLight"] = {
                    {"color", {dlc.color.x, dlc.color.y, dlc.color.z}},
                    {"intensity", dlc.intensity},
                    {"castShadows", dlc.castShadows}};
            }

            if (reg.all_of<SpotLightComponent>(e))
            {
                auto &slc = reg.get<SpotLightComponent>(e);
                entityJson["spotLight"] = {
                    {"color", {slc.color.x, slc.color.y, slc.color.z}},
                    {"intensity", slc.intensity},
                    {"radius", slc.radius},
                    {"innerAngle", slc.innerAngle},
                    {"outerAngle", slc.outerAngle},
                    {"castShadows", slc.castShadows}};
            }

            if (reg.all_of<MaterialComponent>(e))
            {
                auto &mat = reg.get<MaterialComponent>(e);
                entityJson["material"] = {
                    {"albedo", {mat.albedo.x, mat.albedo.y, mat.albedo.z}},
                    {"metallic", mat.metallic},
                    {"roughness", mat.roughness}};
            }

            // CP-3.2: friction/restitution were removed from RigidBody; the actual
            // values are written from the BoxCollider/Colliders material subsection.
            // The reader tolerantly ignores legacy "friction"/"restitution" keys in
            // rigidBody; CP-3.2c is the full migration path to collider material.
            if (reg.all_of<RigidBodyComponent>(e))
            {
                auto &rb = reg.get<RigidBodyComponent>(e);
                entityJson["rigidBody"] = {
                    {"bodyType", static_cast<int>(rb.bodyType)},
                    {"mass", rb.mass}};
            }

            // CP-3.2c: write a material subsection, matching the CollidersComponent format.
            if (reg.all_of<BoxColliderComponent>(e))
            {
                auto &bc = reg.get<BoxColliderComponent>(e);
                entityJson["boxCollider"] = {
                    {"center", {bc.center.x, bc.center.y, bc.center.z}},
                    {"halfExtents", {bc.halfExtents.x, bc.halfExtents.y, bc.halfExtents.z}},
                    {"material", {{"restitution", bc.material.restitution}, {"friction", bc.material.friction}, {"rollingFriction", bc.material.rollingFriction}}}};
            }

            // Only serialized fields (scriptPath / enabled) are written; the runtime
            // cache (loaded / faulted / lastError) is rebuilt by ScriptEngine at attach.
            // scriptPath is written even when empty so tools can distinguish "script
            // component with nothing bound" from "no script component at all".
            if (reg.all_of<ScriptComponent>(e))
            {
                auto &sc = reg.get<ScriptComponent>(e);
                entityJson["script"] = {
                    {"scriptPath", sc.scriptPath},
                    {"enabled", sc.enabled}};
            }

            // Multi-collider serialization; old scenes only have boxCollider, and
            // PhysicsSystem::init's hasMulti dispatch picks the right path.
            if (reg.all_of<CollidersComponent>(e))
            {
                auto &cc = reg.get<CollidersComponent>(e);
                json arr = json::array();
                for (auto &cd : cc.list)
                {
                    json c;
                    const char *typeName = "Box";
                    switch (cd.shape.type)
                    {
                    case ShapeType::Box:
                        typeName = "Box";
                        break;
                    case ShapeType::Sphere:
                        typeName = "Sphere";
                        break;
                    case ShapeType::Capsule:
                        typeName = "Capsule";
                        break;
                    case ShapeType::ConvexHull:
                        typeName = "ConvexHull";
                        break;
                    case ShapeType::Compound:
                        typeName = "Compound";
                        break;
                    }
                    c["type"] = typeName;
                    c["localPos"] = {cd.localPosition.x, cd.localPosition.y, cd.localPosition.z};
                    c["localRot"] = {cd.localRotation.x, cd.localRotation.y, cd.localRotation.z, cd.localRotation.w};
                    if (cd.shape.type == ShapeType::Box)
                    {
                        c["halfExtents"] = {cd.shape.halfExtents.x, cd.shape.halfExtents.y, cd.shape.halfExtents.z};
                    }
                    else if (cd.shape.type == ShapeType::Sphere)
                    {
                        c["radius"] = cd.shape.radius;
                    }
                    else if (cd.shape.type == ShapeType::Capsule)
                    {
                        c["radius"] = cd.shape.radius;
                        c["halfHeight"] = cd.shape.halfHeight;
                    }
                    else if (cd.shape.type == ShapeType::ConvexHull)
                    {
                        // Prefer hullLocalVertices (ECS-native data), then hullCache (post-Play).
                        json verts = json::array();
                        if (!cd.hullLocalVertices.empty())
                        {
                            for (auto &v : cd.hullLocalVertices)
                                verts.push_back({v.x, v.y, v.z});
                        }
                        else if (cd.shape.hullCache)
                        {
                            for (auto &v : cd.shape.hullCache->vertices)
                                verts.push_back({v.x, v.y, v.z});
                        }
                        c["vertices"] = verts;
                    }
                    c["material"] = {
                        {"restitution", cd.material.restitution},
                        {"friction", cd.material.friction},
                        {"rollingFriction", cd.material.rollingFriction}};
                    c["layer"] = cd.layer;
                    c["mask"] = cd.mask;
                    c["isTrigger"] = cd.isTrigger;
                    arr.push_back(c);
                }
                entityJson["colliders"] = arr;
            }

            entities.push_back(entityJson);
        }

        root["entities"] = entities;

        std::ofstream ofs(filepath);
        if (!ofs.is_open())
        {
            std::cerr << "[SceneSerializer] Failed to open file for writing: " << filepath << "\n";
            return false;
        }
        ofs << root.dump(2);
        ofs.close();
        std::cout << "[SceneSerializer] Scene saved to: " << filepath << "\n";
        return true;
    }

    // Deserializes an ECS scene from a JSON file.
    // meshLoader: callback that resolves a sourcePath to a MeshReference
    //   - "builtin:cube" -> built-in cube mesh
    //   - OBJ path       -> loads the OBJ and returns the mesh
    // Returns true on success.
    inline bool loadScene(
        ECSScene &scene,
        const std::string &filepath,
        std::function<std::shared_ptr<MeshReference>(const std::string &sourcePath)> meshLoader)
    {
        std::ifstream ifs(filepath);
        if (!ifs.is_open())
        {
            std::cerr << "[SceneSerializer] Failed to open file for reading: " << filepath << "\n";
            return false;
        }

        json root;
        try
        {
            ifs >> root;
        }
        catch (const json::parse_error &e)
        {
            std::cerr << "[SceneSerializer] JSON parse error: " << e.what() << "\n";
            return false;
        }
        ifs.close();

        if (!root.contains("version") || !root.contains("entities"))
        {
            std::cerr << "[SceneSerializer] Invalid scene file format\n";
            return false;
        }

        auto &reg = scene.registry;

        // Map: JSON id -> entt::entity.
        std::unordered_map<int, entt::entity> idToEntity;

        // Pass 1: create all entities and base components (Name, Transform, Mesh, PointLight).
        for (auto &entityJson : root["entities"])
        {
            int id = entityJson["id"].get<int>();
            std::string name = entityJson.value("name", "Entity");
            auto e = scene.createEntity(name);
            idToEntity[id] = e;

            if (entityJson.contains("transform"))
            {
                auto &tj = entityJson["transform"];
                auto &tc = reg.get<TransformComponent>(e);
                if (tj.contains("position"))
                {
                    auto &p = tj["position"];
                    tc.position = {p[0].get<float>(), p[1].get<float>(), p[2].get<float>()};
                }
                if (tj.contains("rotation"))
                {
                    auto &r = tj["rotation"];
                    tc.setEulerRotation({r[0].get<float>(), r[1].get<float>(), r[2].get<float>()});
                }
                if (tj.contains("scale"))
                {
                    auto &s = tj["scale"];
                    tc.scale = {s[0].get<float>(), s[1].get<float>(), s[2].get<float>()};
                }
            }

            // `source` is builtin:* or guid:<uuid>[#sub]; meshLoader receives the
            // raw string and parses GUID/sub-mesh internally.
            if (entityJson.contains("mesh"))
            {
                std::string source = entityJson["mesh"].value("source", "");
                if (!source.empty() && meshLoader)
                {
                    auto meshRef = meshLoader(source);
                    if (meshRef)
                    {
                        reg.emplace<MeshComponent>(e, MeshComponent{meshRef, source});
                    }
                    else
                    {
                        std::cerr << "[SceneSerializer] Failed to load mesh: " << source << "\n";
                    }
                }
            }

            if (entityJson.contains("pointLight"))
            {
                auto &lj = entityJson["pointLight"];
                PointLightComponent plc;
                if (lj.contains("color"))
                {
                    auto &c = lj["color"];
                    plc.color = {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()};
                }
                plc.intensity = lj.value("intensity", 2.0f);
                plc.radius = lj.value("radius", 5.0f);
                plc.castShadows = lj.value("castShadows", false);
                // Per-light shadow bias; defaults to 1.0 (engine default behavior).
                plc.shadowSlopeBias = lj.value("shadowSlopeBias", 1.0f);
                plc.shadowConstantBias = lj.value("shadowConstantBias", 1.0f);
                plc.shadowNormalBias = lj.value("shadowNormalBias", 1.0f);
                plc.shadowDepthBias = lj.value("shadowDepthBias", 1.0f);
                reg.emplace<PointLightComponent>(e, plc);
            }

            if (entityJson.contains("dirLight"))
            {
                auto &dj = entityJson["dirLight"];
                DirectionalLightComponent dlc;
                if (dj.contains("color"))
                {
                    auto &c = dj["color"];
                    dlc.color = {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()};
                }
                dlc.intensity = dj.value("intensity", 1.0f);
                // Old scenes lack this field; default false (no shadows).
                dlc.castShadows = dj.value("castShadows", false);
                reg.emplace<DirectionalLightComponent>(e, dlc);
            }

            if (entityJson.contains("spotLight"))
            {
                auto &sj = entityJson["spotLight"];
                SpotLightComponent slc;
                if (sj.contains("color"))
                {
                    auto &c = sj["color"];
                    slc.color = {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()};
                }
                slc.intensity = sj.value("intensity", 3.0f);
                slc.radius = sj.value("radius", 10.0f);
                slc.innerAngle = sj.value("innerAngle", 15.0f);
                slc.outerAngle = sj.value("outerAngle", 30.0f);
                slc.castShadows = sj.value("castShadows", false);
                reg.emplace<SpotLightComponent>(e, slc);
            }

            if (entityJson.contains("material"))
            {
                auto &mj = entityJson["material"];
                MaterialComponent mat;
                if (mj.contains("albedo"))
                {
                    auto &a = mj["albedo"];
                    mat.albedo = {a[0].get<float>(), a[1].get<float>(), a[2].get<float>()};
                }
                mat.metallic = mj.value("metallic", 0.0f);
                mat.roughness = mj.value("roughness", 0.5f);
                reg.emplace<MaterialComponent>(e, mat);
            }

            // CP-3.2: legacy "friction"/"restitution" keys in rigidBody are ignored
            // by RigidBody (fields removed); the values are stashed here and used
            // only as a fallback when boxCollider / colliders lack an explicit
            // material subsection (not treated as authoritative).
            float legacyFriction = -1.0f;    // -1 sentinel: legacy scene didn't provide it
            float legacyRestitution = -1.0f; // distinguishes a provided 0
            if (entityJson.contains("rigidBody"))
            {
                auto &rj = entityJson["rigidBody"];
                RigidBodyComponent rb;
                rb.bodyType = static_cast<BodyType>(rj.value("bodyType", 1));
                rb.mass = rj.value("mass", 1.0f);
                if (rj.contains("friction"))
                    legacyFriction = rj.value("friction", 0.5f);
                if (rj.contains("restitution"))
                    legacyRestitution = rj.value("restitution", 0.3f);
                reg.emplace<RigidBodyComponent>(e, rb);
            }

            // CP-3.2c: read the material subsection; if absent (old format), migrate
            // from the legacy rigidBody.friction / restitution (pre-CP-3.2 semantics).
            if (entityJson.contains("boxCollider"))
            {
                auto &bj = entityJson["boxCollider"];
                BoxColliderComponent bc;
                if (bj.contains("center"))
                {
                    auto &c = bj["center"];
                    bc.center = {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()};
                }
                if (bj.contains("halfExtents"))
                {
                    auto &h = bj["halfExtents"];
                    bc.halfExtents = {h[0].get<float>(), h[1].get<float>(), h[2].get<float>()};
                }
                if (bj.contains("material"))
                {
                    auto &mj = bj["material"];
                    bc.material.restitution = mj.value("restitution", 0.3f);
                    bc.material.friction = mj.value("friction", 0.5f);
                    bc.material.rollingFriction = mj.value("rollingFriction", 0.0f);
                }
                else
                {
                    // Old-format migration: override material defaults with the
                    // rigidBody values if present; otherwise keep PhysicsMaterial
                    // defaults (0.5 / 0.3).
                    if (legacyFriction >= 0.0f)
                        bc.material.friction = legacyFriction;
                    if (legacyRestitution >= 0.0f)
                        bc.material.restitution = legacyRestitution;
                }
                reg.emplace<BoxColliderComponent>(e, bc);
            }

            // Old scenes without a "script" field get no ScriptComponent; missing
            // sub-fields fall back to defaults (scriptPath="" / enabled=true).
            // Module loading is deferred to ScriptEngine::syncFromScene; deserialization
            // only fills fields so loadScene never triggers Python exceptions.
            if (entityJson.contains("script"))
            {
                auto &sj = entityJson["script"];
                ScriptComponent sc;
                sc.scriptPath = sj.value("scriptPath", std::string{});
                sc.enabled = sj.value("enabled", true);
                reg.emplace<ScriptComponent>(e, sc);
            }

            // Multi-collider deserialization.
            if (entityJson.contains("colliders") && entityJson["colliders"].is_array())
            {
                CollidersComponent cc;
                for (auto &cj : entityJson["colliders"])
                {
                    ColliderDesc cd;
                    std::string typeStr = cj.value("type", "Box");
                    if (typeStr == "Box")
                        cd.shape.type = ShapeType::Box;
                    else if (typeStr == "Sphere")
                        cd.shape.type = ShapeType::Sphere;
                    else if (typeStr == "Capsule")
                        cd.shape.type = ShapeType::Capsule;
                    else if (typeStr == "ConvexHull")
                        cd.shape.type = ShapeType::ConvexHull;
                    else
                        cd.shape.type = ShapeType::Box;

                    if (cj.contains("localPos"))
                    {
                        auto &p = cj["localPos"];
                        cd.localPosition = {p[0].get<float>(), p[1].get<float>(), p[2].get<float>()};
                    }
                    if (cj.contains("localRot"))
                    {
                        auto &r = cj["localRot"];
                        cd.localRotation = glm::quat(r[3].get<float>(),
                                                     r[0].get<float>(),
                                                     r[1].get<float>(),
                                                     r[2].get<float>());
                    }
                    if (cd.shape.type == ShapeType::Box && cj.contains("halfExtents"))
                    {
                        auto &h = cj["halfExtents"];
                        cd.shape.halfExtents = {h[0].get<float>(), h[1].get<float>(), h[2].get<float>()};
                    }
                    else if (cd.shape.type == ShapeType::Sphere)
                    {
                        cd.shape.radius = cj.value("radius", 0.5f);
                    }
                    else if (cd.shape.type == ShapeType::Capsule)
                    {
                        cd.shape.radius = cj.value("radius", 0.3f);
                        cd.shape.halfHeight = cj.value("halfHeight", 0.5f);
                    }
                    else if (cd.shape.type == ShapeType::ConvexHull)
                    {
                        cd.shape.hullIndex = -1;
                        cd.shape.hullCache = nullptr;
                        if (cj.contains("vertices") && cj["vertices"].is_array())
                        {
                            for (auto &v : cj["vertices"])
                                cd.hullLocalVertices.emplace_back(
                                    v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
                        }
                    }
                    if (cj.contains("material"))
                    {
                        auto &mj = cj["material"];
                        cd.material.restitution = mj.value("restitution", 0.3f);
                        cd.material.friction = mj.value("friction", 0.5f);
                        cd.material.rollingFriction = mj.value("rollingFriction", 0.0f);
                    }
                    cd.layer = cj.value("layer", 0u);
                    cd.mask = cj.value("mask", 0u);
                    cd.isTrigger = cj.value("isTrigger", false);
                    cc.list.push_back(cd);
                }
                if (!cc.list.empty())
                    reg.emplace<CollidersComponent>(e, cc);
            }
        }

        // Pass 2: restore HierarchyComponent parent/child relations.
        for (auto &entityJson : root["entities"])
        {
            int id = entityJson["id"].get<int>();
            if (!idToEntity.count(id))
                continue;
            auto e = idToEntity[id];

            if (entityJson.contains("hierarchy"))
            {
                auto &hj = entityJson["hierarchy"];
                if (hj.contains("parent") && !hj["parent"].is_null())
                {
                    int parentId = hj["parent"].get<int>();
                    if (idToEntity.count(parentId))
                    {
                        scene.setParent(e, idToEntity[parentId]);
                    }
                }
            }
        }

        std::cout << "[SceneSerializer] Scene loaded from: " << filepath
                  << " (" << idToEntity.size() << " entities)\n";
        return true;
    }

} // namespace SceneSerializer
