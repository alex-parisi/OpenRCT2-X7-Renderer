/// Renderer.hpp

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "Image.hpp"
#include "Mesh.hpp"
#include "Palette.hpp"
#include "RayTrace.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    inline constexpr int kRenderWidth = 255;
    inline constexpr int kRenderHeight = 256;
    inline constexpr int kRenderPixels = kRenderWidth * kRenderHeight;
    inline constexpr int kUnitsPerTile = 4096;
    inline constexpr int kUnitsPerPixel = 128;
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
        std::uint8_t flags{};
        std::uint8_t region{};
    };

    enum LightType : std::uint16_t {
        LIGHT_HEMI = 0,
        LIGHT_DIFFUSE = 1,
        LIGHT_SPECULAR = 2,
    };

    struct Light {
        std::uint16_t type{};
        std::uint16_t shadow{};
        Vector3 direction{};
        float intensity{};
    };

    struct Framebuffer {
        std::uint16_t width{};
        std::uint16_t height{};
        Vector2 offset{};
        std::vector<Fragment> fragments{};
    };

    struct Context {
        std::vector<Light> lights{};
        bool dither{};
        Matrix3 projection{};
        Device rt_device{};
        Scene rt_scene{};
        Palette palette{};
    };

    extern std::array<Matrix3, 4> views;

    void context_init(Context& ctx, std::span<const Light> lights, bool dither, Palette palette, float upt);

    void context_destroy(Context& ctx);

    void context_begin_render(Context& ctx);

    void context_add_model(Context& ctx, const Mesh& mesh, Transform xform, int mask = 0);

    Image context_render_view(Context& ctx, Matrix3 view);

    Image context_render_silhouette(Context& ctx, Matrix3 view);

    void context_finalize_render(Context& ctx);

    void context_end_render(Context& ctx);
} // namespace RCTGen
