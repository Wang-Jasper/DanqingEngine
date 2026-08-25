# Phase 13 minimal sanity check: prove that scoped_interpreter works in
# isolation. Configured by the parent CMakeLists when BUILD_PYTHON_EMBED_TEST
# is ON (it is by default) so we can `cmake --build build --target python_embed_test`.

add_executable(python_embed_test main.cpp)
target_link_libraries(python_embed_test PRIVATE pybind11::embed)
target_compile_definitions(python_embed_test PRIVATE
    RENDERER_PYTHON_HOME="${RENDERER_PYTHON_HOME_PATH}"
)
set_target_properties(python_embed_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_embed_test"
)
