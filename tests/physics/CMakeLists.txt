#[[ Physics unit tests, standalone `physics_tests` executable.
 Links STL + glm only; compiles the physics core cpp files directly until they
 move into the libphysics static library. ]]

file(GLOB PHYSICS_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_executable(physics_tests
    ${PHYSICS_TEST_SOURCES}
    "${CMAKE_SOURCE_DIR}/src/physics/PhysicsWorld.cpp"
    "${CMAKE_SOURCE_DIR}/src/physics/DynamicAABBTree.cpp"
    "${CMAKE_SOURCE_DIR}/src/physics/narrowphase_gjk.cpp"
    "${CMAKE_SOURCE_DIR}/src/physics/ContactConstraint.cpp"
    "${CMAKE_SOURCE_DIR}/src/physics/ccd_toi.cpp"
)

target_include_directories(physics_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src"              # enables #include "physics/PhysicsWorld.h"
    "${CMAKE_CURRENT_SOURCE_DIR}"          # local test_framework.h
)

target_link_libraries(physics_tests PRIVATE
    glm::glm
)

target_compile_definitions(physics_tests PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# Put the binary next to the main exe for manual runs.
set_target_properties(physics_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)
