find_package(GTest CONFIG REQUIRED)

add_executable(x7_tests
    test_vector_math.cpp
    test_palette.cpp
    test_mesh.cpp
    test_thread_pool.cpp
    test_ray_trace.cpp
    test_renderer.cpp
)
target_link_libraries(x7_tests PRIVATE _x7_renderer_core GTest::gtest_main)
target_compile_options(x7_tests PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wno-unused-parameter>
)

add_test(NAME x7_tests COMMAND x7_tests)

# Separate test binary that links against mock Embree to test error paths.
# Compiles RayTrace.cpp directly (not via _x7_renderer_core which links real Embree).
add_executable(x7_error_tests
    test_ray_trace_errors.cpp
    mock_embree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RayTrace.cpp
)
target_include_directories(x7_error_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Link against GTest and the Embree headers (but NOT the Embree library itself)
target_link_libraries(x7_error_tests PRIVATE GTest::gtest_main)
target_compile_options(x7_error_tests PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wno-unused-parameter>
)
# Need Embree headers for types/enums but we supply our own function definitions
find_package(embree 4 CONFIG REQUIRED)
target_include_directories(x7_error_tests PRIVATE $<TARGET_PROPERTY:embree,INTERFACE_INCLUDE_DIRECTORIES>)

add_test(NAME x7_error_tests COMMAND x7_error_tests)
