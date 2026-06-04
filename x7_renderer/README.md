# X7 Renderer

The C++ / Embree core of `openrct2-x7-renderer`. It turns triangle meshes into
palette-indexed sprites via ray tracing, then exposes the pipeline to Python
through pybind11.

Heavily inspired by X7's [RCTGen](https://github.com/X123M3-256/RCTGen).

## How it works

Input is a set of triangle meshes with materials and optional textures. Output
is palette-indexed sprites in OpenRCT2's image format.

1. **Scene assembly** (`RayTrace.cpp`): meshes go into an Embree BVH. The scene
   stores a pointer per geometry so material and texture lookups work at hit
   time via the geometry ID.
2. **Camera** (`Renderer.cpp`): an orthographic dimetric projection matching
   OpenRCT2's view (2:1 pixel ratio, ~30° elevation). Each pixel fires 16
   primary rays in a 4×4 jittered grid for anti-aliasing.
3. **Shading**: diffuse plus specular at the hit, multiplied by ambient
   occlusion sampled with 32 rays over a stratified hemisphere (8×4 grid,
   uniform solid-angle distribution). Material flags pick out remap regions,
   metallic highlights, and transparency.
4. **Quantization** (`Palette.cpp`): nearest-color match against the RCT2
   internal palette, with Floyd-Steinberg dithering. Remap colors go into the
   palette ranges OpenRCT2 reserves for runtime recoloring.
5. **Output**: an `IndexedImage` crosses the pybind11 boundary; Python packs it
   into `images.dat`.

## Prerequisites

| Dependency | Version | Install (macOS) |
|---|---|---|
| [Embree](https://github.com/RenderKit/embree) | 4.x | `brew install embree` |
| C++ compiler | C++23 | Xcode / `brew install llvm` |
| CMake | 3.25+ | `brew install cmake` |
| Ninja | any | `brew install ninja` |
| GoogleTest | any | `brew install googletest` |

## Building and running the C++ tests

Configure the `dev` preset once, then build and run the test binary via CTest:

```bash
cd x7_renderer
cmake --preset dev          # configures build/ with BUILD_TESTING=ON
cmake --build --preset dev  # builds the x7_tests and x7_error_tests executables
ctest --preset dev          # runs the suite, prints failures
```

The test binary can also be invoked directly at `build/tests/x7_tests` —
useful for running a single GoogleTest filter, e.g.
`build/tests/x7_tests --gtest_filter=Palette.*`.

### Coverage

A separate `coverage` preset produces LLVM source-based coverage data:

```bash
cmake --preset coverage
cmake --build --preset coverage
ctest --preset coverage
```

## Source layout

```
x7_renderer/
├── .clang-format          # clang-format style
├── .clang-tidy            # clang-tidy config (naming, modernize, bugprone, …)
├── bindings.cpp           # pybind11 module (_x7_renderer)
├── CMakeLists.txt         # build system
├── CMakePresets.json      # dev / coverage presets
├── src/
│   ├── Color.hpp          # RGB colour struct
│   ├── Image.hpp          # IndexedImage type
│   ├── Mesh.cpp/.hpp      # triangle mesh & material types
│   ├── Palette.cpp/.hpp   # RCT2 palette + nearest-color quantization
│   ├── RayTrace.cpp/.hpp  # Embree scene, primary + AO ray casting
│   ├── Renderer.cpp/.hpp  # dimetric camera, shading orchestration
│   ├── ThreadPool.hpp     # persistent work-stealing thread pool
│   └── VectorMath.hpp     # vec3/mat3 helpers
└── tests/
    ├── CMakeLists.txt     # test build configuration
    ├── mock_embree.cpp/.h # mock Embree API for error-path tests
    ├── test_mesh.cpp
    ├── test_palette.cpp
    ├── test_ray_trace.cpp
    ├── test_ray_trace_errors.cpp
    ├── test_renderer.cpp
    ├── test_thread_pool.cpp
    └── test_vector_math.cpp
```

## License

GPL-3.0-or-later.
