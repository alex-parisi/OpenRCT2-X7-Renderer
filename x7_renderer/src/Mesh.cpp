/// Mesh.cpp

#include "Mesh.hpp"
#include <algorithm>
#include <cmath>

namespace RCTGen {
    namespace {
        float wrap_coord(float coord) { return std::clamp(coord - std::floor(coord), 0.0f, 1.0f); }
    } // namespace

    Vector3 texture_sample(const Texture& texture, Vector2 coord) {
        auto tex_x = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(static_cast<float>(texture.width) * wrap_coord(coord.x)));
        auto tex_y = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(static_cast<float>(texture.height) * wrap_coord(coord.y)));
        if (tex_x == texture.width) tex_x = 0;
        if (tex_y == texture.height) tex_y = 0;
        return texture.pixels[tex_y * texture.width + tex_x];
    }
} // namespace RCTGen
