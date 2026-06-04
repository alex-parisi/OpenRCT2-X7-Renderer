/// Image.hpp

#pragma once

#include <cstdint>
#include <vector>

namespace RCTGen {
    struct Image {
        std::uint16_t width{};
        std::uint16_t height{};
        std::int16_t x_offset{};
        std::int16_t y_offset{};
        std::vector<std::uint8_t> pixels{};
    };
} // namespace RCTGen
