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

    enum class MaterialFlag : std::uint16_t {
        None             = 0,
        HasTexture       = 1u << 0,
        IsRemappable     = 1u << 1,
        IsMask           = 1u << 2,
        NoAO             = 1u << 3,
        BackgroundAA     = 1u << 4,
        BackgroundAADark = 1u << 5,
        IsVisibleMask    = 1u << 6,
        NoBleed          = 1u << 7,
        IsFlatShaded     = 1u << 8,
    };

    constexpr MaterialFlag operator|(MaterialFlag a, MaterialFlag b) noexcept {
        return static_cast<MaterialFlag>(static_cast<std::uint16_t>(a) | static_cast<std::uint16_t>(b));
    }
    constexpr MaterialFlag operator&(MaterialFlag a, MaterialFlag b) noexcept {
        return static_cast<MaterialFlag>(static_cast<std::uint16_t>(a) & static_cast<std::uint16_t>(b));
    }
    constexpr MaterialFlag& operator|=(MaterialFlag& a, MaterialFlag b) noexcept {
        return a = a | b;
    }
    constexpr MaterialFlag operator~(MaterialFlag a) noexcept {
        return static_cast<MaterialFlag>(~static_cast<std::uint16_t>(a));
    }
    constexpr bool has_flag(MaterialFlag flags, MaterialFlag test) noexcept {
        return (flags & test) != MaterialFlag::None;
    }

    struct Material {
        MaterialFlag flags{MaterialFlag::None};
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
