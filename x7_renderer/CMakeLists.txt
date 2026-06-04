cmake_minimum_required(VERSION 3.25)
project(openrct2_vehicle_generator_x7_renderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Opt-in clang-tidy
option(ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy
        HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
    endif()
endif()

# Embree discovery
if(NOT DEFINED embree_DIR AND DEFINED ENV{EMBREE_ROOT})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{EMBREE_ROOT}")
endif()
if(APPLE AND NOT DEFINED embree_DIR)
    file(GLOB _brew_embree_cfg "/opt/homebrew/lib/cmake/embree-4*")
    if(_brew_embree_cfg)
        list(GET _brew_embree_cfg 0 embree_DIR)
    endif()
endif()

find_package(embree 4 CONFIG REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Core Library
add_library(_x7_renderer_core STATIC
    ${SRC}/Mesh.cpp
    ${SRC}/Palette.cpp
    ${SRC}/RayTrace.cpp
    ${SRC}/Renderer.cpp
)
set_target_properties(_x7_renderer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(_x7_renderer_core PUBLIC ${SRC})
target_compile_options(_x7_renderer_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wno-unused-parameter>
)
target_link_libraries(_x7_renderer_core PUBLIC embree Threads::Threads)

# Python Extension Module
option(BUILD_PYTHON_MODULE "Build the pybind11 Python extension module" ON)
if(BUILD_PYTHON_MODULE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_x7_renderer MODULE bindings.cpp)
    target_include_directories(_x7_renderer PRIVATE ${SRC})
    target_compile_options(_x7_renderer PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W3>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wno-unused-parameter>
    )
    target_link_libraries(_x7_renderer PRIVATE _x7_renderer_core)
    install(TARGETS _x7_renderer LIBRARY DESTINATION openrct2_x7_renderer)
endif()

# Unit Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
