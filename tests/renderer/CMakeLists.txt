#[[ ============================================================================
 tests/renderer/CMakeLists.txt — renderer_tests: pure-CPU GLSL-equivalent math
 validating `useLightVolume` pixel equivalence (glm + STL, no Vulkan dependency)
 ============================================================================ ]]

file(GLOB RENDERER_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_executable(renderer_tests ${RENDERER_TEST_SOURCES})

target_include_directories(renderer_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/physics"   # reuse test_framework.h (header-only)
)

target_link_libraries(renderer_tests PRIVATE glm::glm)

target_compile_definitions(renderer_tests PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if(MSVC)
    target_compile_options(renderer_tests PRIVATE /utf-8)
endif()

set_target_properties(renderer_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)
