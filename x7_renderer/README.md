### X7 Renderer

Heavily inspired by X7's [RCTGen](https://github.com/X123M3-256/RCTGen) project.

#### How it works

Input is a set of triangle meshes with materials and optional textures. Output
is palette-indexed sprites in OpenRCT2's image format.

1. Scene assembly (`RayTrace.cpp`): meshes go into an Embree BVH. Each
   triangle stores a pointer back to its mesh so material and texture lookups
   work at hit time.
2. Camera (`Renderer.cpp`): an orthographic projection matching OpenRCT2's
   isometric view (1:2 pixel ratio, ~30° pitch). Each pixel fires 16 primary
   rays in a 4x4 jittered grid for anti-aliasing.
3. Shading: diffuse plus specular at the hit, multiplied by ambient occlusion
   sampled with 32 rays over a cosine-weighted hemisphere (8x4). Material
   flags pick out remap regions, metallic highlights, and transparency.
4. Quantization (`Palette.cpp`): nearest-color match against the RCT2 internal
   palette, with Floyd-Steinberg dithering. Remap colors go into the palette
   ranges OpenRCT2 reserves for runtime recoloring.
5. Output: an `IndexedImage` crosses the pybind11 boundary; Python packs it
   into `images.dat`.

## Building and running the C++ tests

GoogleTest is required (`brew install googletest`). Configure the `dev` preset
once, then build and run the test binary via CTest:

```bash
cd x7_renderer
cmake --preset dev          # configures build/ with BUILD_TESTING=ON
cmake --build --preset dev  # builds the native_tests executable
ctest --preset dev          # runs the suite, prints failures
```

The test binary can also be invoked directly at `build/test/native_tests` —
useful for running a single GoogleTest filter, e.g.
`build/test/native_tests --gtest_filter=Palette.*`.
