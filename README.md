# Danqing

A Vulkan deferred renderer with a custom 2D physics engine, an ECS-based scene editor and embedded Python scripting.

- Deferred G-buffer pipeline with HDR lighting, shadow maps (directional / spot / point), Hi-Z occlusion culling, SSAO, FXAA and bloom
- Custom 2D rigid-body physics: AABB tree broadphase, GJK/EPA and SAT narrowphase, sequential-impulse solver with warm starting, joints, CCD and island sleeping
- ImGui editor (docking) with viewport, hierarchy, inspector, render settings and script console
- EnTT-based ECS with JSON scene serialization
- Embedded Python 3 scripting via pybind11

## Features

**Renderer**

- Geometry pass → lighting pass with per-light volumes
- Directional, spot and point light shadows (cascaded-free, single/split shadow maps)
- Hi-Z depth downsample + frustum/occlusion culling for meshes
- SSAO with separable blur, FXAA, HDR bloom (threshold / downsample / upsample / composite)
- G-buffer debug views (position, normal, albedo, depth, Hi-Z mip 0)
- GPU and CPU profilers, built-in benchmark runner with CSV output

**Physics** (2D, custom)

- Dynamic AABB tree broadphase; GJK/EPA and SAT narrowphase with warm starting and manifold persistence
- Sequential-impulse velocity + position solver, friction and restitution, material properties
- Ball / fixed / hinge / slider joints, CCD (TOI), island sleeping, raycasts
- 35+ unit tests in `tests/physics`

**Editor & scripting**

- ImGui docking layout: Viewport (with ImGuizmo), Hierarchy, Inspector, Render Settings, Performance, Script Console
- Drag-and-drop `.obj` import; JSON scene load/save
- Python scripting over the ECS, components and GLM types

## Requirements

- CMake 3.20+
- A C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
- Vulkan SDK 1.2+ (with `glslc` on PATH or in `%VULKAN_SDK%/Bin`)
- Python 3.8+ with development headers (for the embedded interpreter; `python3-dev` on Linux)

All other dependencies (GLFW, GLM, VMA, tinyobjloader, stb, Dear ImGui, EnTT, nlohmann/json, ImGuizmo, pybind11) are fetched and built by CMake via FetchContent.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Shaders are compiled to SPIR-V during the build and copied next to the executable; run from the build output directory.

Physical tests build by default (`-DBUILD_PHYSICS_TESTS=OFF` disables). Run them with:

```sh
./build/tests/physics/physics_tests   # or build/tests/physics/Release/physics_tests.exe on Windows
```

## Run

```sh
./build/Danqing            # interactive
./build/Danqing --help
./build/Danqing --benchmark Stress1000 --frames 600 --auto-exit
```

Benchmark presets: `Empty`, `Stress100`, `Stress1000`, `LightStress`, `TextureStress`.

Example scenes live in `assets/` (`demo_mixed_shapes.json`, `demo_shadow.json`, `demo_bloom.json`, …) and can be loaded from the editor. Drag an `.obj` file onto the window to import it.

## Controls

| Input | Action |
|---|---|
| WASD + Mouse | Move / look (FPS camera) |
| Space / Shift | Up / Down |
| Scroll | Zoom (FOV) |
| Click viewport | Capture mouse (FPS mode) |
| ESC | Release mouse |
| Q | Quit |
| 0 / F5 | Final render |
| 1–4 / F1–F4 | G-buffer debug views (position, normal, albedo, depth) |
| F6 | Hi-Z mip 0 |

## Project layout

```
src/core        Vulkan context, swapchain, command buffers, VMA allocator
src/renderer    Deferred pipeline, shadow passes, Hi-Z, SSAO, bloom, profilers
src/physics     Broadphase, narrowphase, solver, joints, CCD
src/ecs         EnTT components, systems, scene serialization
src/scene       Camera, meshes, lights, vertex layout
src/editor      ImGui editor UI
src/scripting   pybind11 bindings, embedded Python interpreter
shaders         GLSL sources (compiled offline with glslc)
tests           Physics / renderer / python-embed test suites
benchmarks      Benchmark collection and reporting scripts
```

## License

[MIT](LICENSE)
