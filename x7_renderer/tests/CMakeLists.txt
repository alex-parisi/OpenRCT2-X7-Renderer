find_package(GTest CONFIG REQUIRED)

add_executable(x7_tests
    test_vector_math.cpp
    test_palette.cpp
    test_mesh.cpp
)
target_link_libraries(x7_tests PRIVATE _x7_renderer_core GTest::gtest_main)
target_compile_options(x7_tests PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wno-unused-parameter>
)

add_test(NAME x7_tests COMMAND x7_tests)
