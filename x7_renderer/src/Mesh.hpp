/// Mesh.hpp

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "VectorMath.hpp"

namespace RCTGen {
    struct Texture {
        std::uint16_t width{};
        std::uint16_t height{};
        std::span<const Vector3> pixels{};
    };

    /* TODO: The historical bit layout and names are preserved exactly,
        but this should be modernized */
    inline constexpr std::uint16_t MATERIAL_HAS_TEXTURE = 1u << 0;
    inline constexpr std::uint16_t MATERIAL_IS_REMAPPABLE = 1u << 1;
    inline constexpr std::uint16_t MATERIAL_IS_MASK = 1u << 2;
    inline constexpr std::uint16_t MATERIAL_NO_AO = 1u << 3;
    inline constexpr std::uint16_t MATERIAL_BACKGROUND_AA = 1u << 4;
    inline constexpr std::uint16_t MATERIAL_BACKGROUND_AA_DARK = 1u << 5;
    inline constexpr std::uint16_t MATERIAL_IS_VISIBLE_MASK = 1u << 6;
    inline constexpr std::uint16_t MATERIAL_NO_BLEED = 1u << 7;
    inline constexpr std::uint16_t MATERIAL_IS_FLAT_SHADED = 1u << 8;

    struct Material {
        std::uint16_t flags{};
        std::uint8_t region{};
        float specular_exponent{};
        Vector3 specular_color{};
        Vector3 ambient_color{};
        Texture texture{};
        Vector3 color{};
    };

    struct Face {
        std::size_t material{};
        std::array<std::size_t, 3> indices{};
    };

    struct Mesh {
        std::span<const Vector3> vertices{};
        std::span<const Vector3> normals{};
        std::span<const Vector2> uvs{};
        std::span<const Face> faces{};
        std::span<const Material> materials{};
    };

    Vector3 texture_sample(const Texture& texture, Vector2 coord);
} // namespace RCTGen
