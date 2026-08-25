cmake_minimum_required(VERSION 3.20)
project(Danqing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Export compile_commands.json for clangd / IDE tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# --- GLFW: window / input ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
# Skip GLFW's docs, tests, examples and install targets to cut build time.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- GLM: header-only math ---
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
# Suppress the "< 3.10 deprecated" warning from glm 1.0.1's old
# cmake_minimum_required: CMAKE_POLICY_VERSION_MINIMUM (CMake 3.24+) makes the
# subproject run at that policy floor. Scoped locally and unset after so it
# does not leak into later dependencies.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(glm)
unset(CMAKE_POLICY_VERSION_MINIMUM)

# --- VMA: Vulkan memory allocator ---
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(vma)

# --- tinyobjloader: OBJ model loader ---
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobjloader)

# --- stb: header-only utilities (stb_image for textures) ---
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master                             # header-only, low churn — master is safe
)
FetchContent_MakeAvailable(stb)

# --- Dear ImGui (docking branch) ---
# Pin the docking tag; tracking the branch HEAD once hit a WIP assert
# (1.92.8 WIP: Debug AddPolyline assert at imgui_draw.cpp:829, InvalidMask).
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.7-docking
)
FetchContent_MakeAvailable(imgui)

# --- tinyfiledialogs: native file dialogs (single-file C) ---
FetchContent_Declare(
    tinyfiledialogs
    GIT_REPOSITORY https://git.code.sf.net/p/tinyfiledialogs/code
    GIT_TAG        master
)
FetchContent_MakeAvailable(tinyfiledialogs)

# --- EnTT: header-only ECS ---
FetchContent_Declare(
    entt
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG        v3.14.0
)
FetchContent_MakeAvailable(entt)

# --- nlohmann/json: scene serialization ---
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# --- ImGuizmo: viewport gizmos (built on Dear ImGui) ---
# ImGuizmo's own CMakeLists builds a static lib from all six components
# (ImCurveEdit, ImGradient, ImSequencer, ImVectorEditor, GraphEditor) but never
# passes imgui's include path, so "imgui.h" is not found. We only need
# ImGuizmo.cpp, so SOURCE_SUBDIR points at a subdir with no CMakeLists
# (docs/): MakeAvailable only populates, and ImGuizmo.cpp is added manually to
# the main target.
FetchContent_Declare(
    imguizmo
    GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
    GIT_TAG        master
    SOURCE_SUBDIR  docs           # no CMakeLists.txt here — skips upstream CMake
)
FetchContent_MakeAvailable(imguizmo)

# --- pybind11: Python bindings / embedded interpreter (script system) ---
# Embed mode: the engine hosts CPython through ScriptEngine's
# scoped_interpreter. Pin a stable tag — upstream-master drift once broke
# reproducibility (same class of bug as the ImGui docking HEAD).
#
# Requires a Python 3.x dev install (python.org ships headers on Windows;
# apt-get install python3-dev on Linux). pybind11's CMake runs its own
# find_package(Python3) and fails at configure time if none is found.
#
# Windows 11 pitfall: PATH's `%LocalAppData%\Microsoft\WindowsApps\python3.exe`
# is a Microsoft Store stub, not real Python. CMake finds it first, pybind11
# links a Python without stdlib, and scoped_interpreter throws
# `ImportError: initialization failed` (missing encodings etc.). Force the real
# python.org / Anaconda install:
#   * Python3_FIND_REGISTRY = LAST      — registry entries last (avoid the Store)
#   * Python3_FIND_STRATEGY = LOCATION  — pair the Library with the Interpreter
#   * explicit find_package(Python3 ... Development Interpreter) before
#     FetchContent, so pybind11's internal find_package uses the cache and is
#     not swayed by WindowsApps.
set(Python3_FIND_REGISTRY LAST)
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter Development.Embed)
message(STATUS "[pybind11] Using Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")
message(STATUS "[pybind11] Python3 stdlib: ${Python3_STDLIB}")

# Compute the PYTHONHOME path = parent of stdlib dir.
# CPython's PyConfig.home expects the install root (e.g. C:/.../Python39/),
# whose direct children are `Lib/` and `DLLs/`. Python3_STDLIB resolves to
# that `Lib/` directory, so its parent is exactly what we want.
get_filename_component(RENDERER_PYTHON_HOME_PATH "${Python3_STDLIB}" DIRECTORY)
message(STATUS "[pybind11] PYTHONHOME baked into exe: ${RENDERER_PYTHON_HOME_PATH}")

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Required — configure fails without the Vulkan SDK.
find_package(Vulkan REQUIRED)

# Offline shader compilation: GLSL -> SPIR-V.
find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found! Make sure Vulkan SDK is installed.")
endif()

set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

file(GLOB SHADER_SOURCES
    "${SHADER_SOURCE_DIR}/*.vert"
    "${SHADER_SOURCE_DIR}/*.frag"
    "${SHADER_SOURCE_DIR}/*.comp"
)

set(SHADER_BINARIES "")
foreach(SHADER_SRC ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
    set(SHADER_SPV "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC_EXECUTABLE} -O ${SHADER_SRC} -o ${SHADER_SPV}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling shader: ${SHADER_NAME}"
    )
    list(APPEND SHADER_BINARIES ${SHADER_SPV})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_BINARIES})

# CONFIGURE_DEPENDS rescans on build, avoiding the GLOB trap where new sources
# are missed and cause LNK2019 link errors.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.h"
)

# ImGui core + Vulkan/GLFW backends.
set(IMGUI_SOURCES
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES}
    "${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c"
    "${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp"
)
add_dependencies(${PROJECT_NAME} compile_shaders)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${Vulkan_INCLUDE_DIRS}"
    "${stb_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
    "${tinyfiledialogs_SOURCE_DIR}"
    "${entt_SOURCE_DIR}/src"
    "${imguizmo_SOURCE_DIR}/src"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    GPUOpen::VulkanMemoryAllocator
    tinyobjloader
    nlohmann_json::nlohmann_json
    pybind11::embed                  # embed mode also pulls in Python::Python
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLFW_INCLUDE_VULKAN              # GLFW must pull in vulkan.h itself
    GLM_FORCE_RADIANS                # GLM angles in radians
    GLM_FORCE_DEPTH_ZERO_TO_ONE      # Vulkan depth range is [0,1] (GL is [-1,1])
    SHADER_DIR="./shaders"           # runtime path, shaders/ next to the exe

    # Bake the CMake-discovered Python install into the exe. Embedded CPython
    # must find its own stdlib (encodings/os); the `WindowsApps\python3.exe`
    # Store stub on Windows 11 PATH has none, so Py_Initialize would throw
    # ImportError (initialization failed). ScriptEngine sets PyConfig home to
    # the install root (= parent of ${Python3_STDLIB}, e.g. .../Python39/).
    RENDERER_PYTHON_HOME="${RENDERER_PYTHON_HOME_PATH}"
    RENDERER_PYTHON_STDLIB="${Python3_STDLIB}"
)

# Copy compiled .spv next to the exe for the runtime SHADER_DIR.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_BINARY_DIR}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "Copying compiled shaders to output directory"
)

# Validation layers on in Debug (or when no build type is set), off in Release.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_VALIDATION_LAYERS=1)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_VALIDATION_LAYERS=0)
endif()

# =============================================================================
# Physics unit tests: standalone physics_tests target, independent of the main exe.
# Built by default; disable with -DBUILD_PHYSICS_TESTS=OFF.
# =============================================================================
option(BUILD_PHYSICS_TESTS "Build physics unit tests (tests/physics)" ON)
if(BUILD_PHYSICS_TESTS)
    add_subdirectory(tests/physics)
endif()

option(BUILD_RENDERER_TESTS "Build renderer unit tests (tests/renderer)" ON)
if(BUILD_RENDERER_TESTS)
    add_subdirectory(tests/renderer)
endif()

# Phase 13.1 — minimal embedded-Python sanity binary. Independent of the
# renderer; useful to triage "why does scoped_interpreter fail?" without
# Vulkan / ECS noise. Default OFF: the renderer-side path is now stable, but
# we keep the target so future Python integration regressions can be
# isolated from our pybind11 wiring with a single `cmake -DBUILD_PYTHON_EMBED_TEST=ON`.
option(BUILD_PYTHON_EMBED_TEST "Build the standalone embedded-Python sanity test" OFF)
if(BUILD_PYTHON_EMBED_TEST)
    add_subdirectory(tests/python_embed)
endif()
