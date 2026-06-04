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

    struct Context {
        std::vector<Light> lights{};
        bool dither{};
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

    void ContextInit(Context& ctx, std::span<const Light> lights, bool dither, Palette palette, float upt);

    void context_destroy(Context& ctx);

    void context_begin_render(Context& ctx);

    void context_add_model(Context& ctx, const Mesh& mesh, Transform xform, MeshFlag mask = MeshFlag::None);

    Image context_render_view(Context& ctx, Matrix3 view);

    Image context_render_silhouette(Context& ctx, Matrix3 view);

    void context_finalize_render(Context& ctx);

    void context_end_render(Context& ctx);
} // namespace RCTGen
