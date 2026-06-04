/// Mesh.cpp

#include "Mesh.hpp"
#include <algorithm>
#include <cmath>

namespace RCTGen {
    namespace {
        float WrapCoord(float coord) {
            return std::clamp(coord - std::floor(coord), 0.0f, 1.0f);
        }
    } // namespace

    // Nearest-neighbour texture lookup with UV wrapping.
    Vector3 texture_sample(const Texture& texture, Vector2 coord) {
        if (texture.width == 0 || texture.height == 0 || texture.pixels.empty()) return vector3(0.0f, 0.0f, 0.0f);
        auto tex_x = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(static_cast<float>(texture.width) * WrapCoord(coord.x)));
        auto tex_y = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(static_cast<float>(texture.height) * WrapCoord(coord.y)));
        if (tex_x >= texture.width) tex_x = 0;
        if (tex_y >= texture.height) tex_y = 0;
        return texture.pixels[tex_y * texture.width + tex_x];
    }
} // namespace RCTGen
