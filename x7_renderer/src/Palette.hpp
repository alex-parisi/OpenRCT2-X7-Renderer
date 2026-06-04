/// Palette.hpp

#pragma once

#include <array>
#include <cstdint>

#include "Color.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    inline constexpr std::size_t kNumRegions = 8;
    inline constexpr std::size_t kNumSubregions = 4;

    struct Region {
        std::uint8_t subregions;
        std::array<std::uint8_t, kNumSubregions> start_indices;
        std::array<std::uint8_t, kNumSubregions> end_indices;
        bool remap;
    };

    struct Palette {
        std::uint8_t transparent_index;
        std::array<Region, kNumRegions> regions;
        std::array<Color, 256> colors;
        std::array<Color, 16> remap_colors;
    };

    struct PaletteResult {
        std::uint8_t index{};
        Vector3 error{};
    };

    constexpr Color color(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept { return {r, g, b}; }

    Vector3 vector_from_color(Color c);

    Color color_from_vector(Vector3 vec);

    PaletteResult palette_get_nearest(const Palette& palette, std::uint8_t region, Vector3 color);

    Palette palette_rct2();
} // namespace RCTGen
