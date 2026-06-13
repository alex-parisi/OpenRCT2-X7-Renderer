/// Renderer.hpp

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "Image.hpp"
#include "Mesh.hpp"
#include "Palette.hpp"
#include "RayTrace.hpp"
#include "ThreadPool.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    inline constexpr std::uint8_t kFragmentUnused = 255;
    inline constexpr std::uint8_t kRegionMask = 0x7;
    inline constexpr std::size_t kMaxRegions = 8;

    struct Rect {
        std::int32_t x_lower{};
        std::int32_t y_lower{};
        std::int32_t x_upper{};
        std::int32_t y_upper{};
    };

    struct Fragment {
        Vector3 color{};
        float depth{};
        float ghost_depth{};
        MaterialFlag flags{}; // full 16-bit flags; avoids silent truncation of IsFlatShaded (bit 8)
        std::uint8_t region{};
    };

    enum class LightType : std::uint16_t {
        Hemi = 0,
        Diffuse = 1,
        Specular = 2,
    };

    struct Light {
        LightType type{LightType::Hemi};
        std::uint16_t shadow{};
        Vector3 direction{};
        float intensity{};
    };

    struct Framebuffer {
        std::uint32_t width{};
        std::uint32_t height{};
        Vector2 offset{};
        std::vector<Fragment> fragments{};
    };

    enum class DitherMode : std::uint8_t {
        None = 0,
        // Error-diffusion: highest fidelity but temporally unstable. A tiny
        // per-frame colour change anywhere cascades through the serpentine scan,
        // so the dither pattern "swims" across an animation's frames.
        FloydSteinberg = 1,
        // Ordered dither anchored to the engine-screen coordinate. Each pixel's
        // output depends only on its own colour and its (frame-invariant) screen
        // position, so unchanged regions are byte-identical between frames.
        Bayer = 2,
        // Like Bayer, but driven by a 64x64 blue-noise threshold tile instead of
        // the 8x8 recursive matrix. Same frame-invariant screen anchoring, but
        // the mask has no low-frequency structure, so wherever the shading does
        // change between frames (rotation, animation) the residual dither motion
        // is far less perceptible than Bayer's regular cross-hatch.
        BlueNoise = 3,
    };

    struct Context {
        std::vector<Light> lights{};
        DitherMode dither{DitherMode::None};
        // Temporal-stability deadband, in 8-bit sRGB units. Before dithering, the
        // pre-quantisation colour is snapped onto a grid of this size so shading
        // changes smaller than the deadband quantise identically across frames
        // (suppressing sub-step "swimming"); the ordered dither masks the banding
        // this would otherwise introduce. 0 disables it.
        float stability{0.0f};
        bool finalized{};
        Matrix3 projection{};
        DeviceHandle rt_device;
        std::unique_ptr<Scene> rt_scene;
        Palette palette{};
        std::unique_ptr<ThreadPool> thread_pool;
    };

    inline constexpr std::array<Matrix3, 4> views{{
        {{1, 0, 0, 0, 1, 0, 0, 0, 1}},
        {{0, 0, 1, 0, 1, 0, -1, 0, 0}},
        {{-1, 0, 0, 0, 1, 0, 0, 0, -1}},
        {{0, 0, -1, 0, 1, 0, 1, 0, 0}},
    }};

    void ContextInit(
        Context& ctx, std::span<const Light> lights, DitherMode dither, float stability, Palette palette, float upt);

    void context_destroy(Context& ctx);

    void context_begin_render(Context& ctx);

    void context_add_model(Context& ctx, const Mesh& mesh, Transform xform, MeshFlag mask = MeshFlag::None);

    Image context_render_view(Context& ctx, Matrix3 view);

    Image context_render_silhouette(Context& ctx, Matrix3 view);

    void context_finalize_render(Context& ctx);

    void context_end_render(Context& ctx);
} // namespace RCTGen
