#include "Color.hpp"
#include "Palette.hpp"
#include "VectorMath.hpp"
#include <cstdint>
#include <gtest/gtest.h>

using namespace RCTGen;

static constexpr float kEps = 1e-4f;

// ---------------------------------------------------------------------------
// sRGB <-> linear round-trips via the public API
// ---------------------------------------------------------------------------

TEST(PaletteColorConversion, BlackRoundTrip) {
    Color const black{0, 0, 0};
    Vector3 const linear = vector_from_color(black);
    EXPECT_NEAR(linear.x, 0.0f, kEps);
    EXPECT_NEAR(linear.y, 0.0f, kEps);
    EXPECT_NEAR(linear.z, 0.0f, kEps);

    Color const back = color_from_vector(linear);
    EXPECT_EQ(back.r, 0);
    EXPECT_EQ(back.g, 0);
    EXPECT_EQ(back.b, 0);
}

TEST(PaletteColorConversion, WhiteRoundTrip) {
    Color const white{255, 255, 255};
    Vector3 const linear = vector_from_color(white);
    EXPECT_NEAR(linear.x, 1.0f, kEps);
    EXPECT_NEAR(linear.y, 1.0f, kEps);
    EXPECT_NEAR(linear.z, 1.0f, kEps);

    Color const back = color_from_vector(linear);
    EXPECT_EQ(back.r, 255);
    EXPECT_EQ(back.g, 255);
    EXPECT_EQ(back.b, 255);
}

TEST(PaletteColorConversion, MidGrayIsNonlinear) {
    // sRGB 128 should map to a linear value significantly below 0.5
    // due to the gamma curve (~0.216, not 0.5).
    Vector3 const linear = vector_from_color({128, 128, 128});
    EXPECT_GT(linear.x, 0.18f);
    EXPECT_LT(linear.x, 0.25f);
    EXPECT_FLOAT_EQ(linear.x, linear.y);
    EXPECT_FLOAT_EQ(linear.y, linear.z);
}

TEST(PaletteColorConversion, ColorRoundTripAllChannels) {
    // Each channel should survive a round-trip independently.
    for (std::uint8_t v : {0, 64, 128, 192, 255}) {
        Color const c{v, v, v};
        Color const back = color_from_vector(vector_from_color(c));
        EXPECT_EQ(back.r, v) << "round-trip failed for value " << static_cast<int>(v);
    }
}

// ---------------------------------------------------------------------------
// color_from_vector clamping
// ---------------------------------------------------------------------------

TEST(PaletteColorConversion, ClampNegative) {
    Color const c = color_from_vector(vector3(-1.0f, -10.0f, -0.5f));
    EXPECT_EQ(c.r, 0);
    EXPECT_EQ(c.g, 0);
    EXPECT_EQ(c.b, 0);
}

TEST(PaletteColorConversion, ClampAboveOne) {
    Color const c = color_from_vector(vector3(2.0f, 5.0f, 1.001f));
    EXPECT_EQ(c.r, 255);
    EXPECT_EQ(c.g, 255);
    EXPECT_EQ(c.b, 255);
}

// ---------------------------------------------------------------------------
// palette_get_nearest
// ---------------------------------------------------------------------------

namespace {
    class PaletteTest : public ::testing::Test {
    protected:
        Palette palette = palette_rct2();
    };
} // namespace

TEST_F(PaletteTest, ReturnsValidIndex) {
    // Region 0 covers indices 10–201, 214–226, 240–242. Any returned index
    // must be within the full palette range.
    auto result = palette_get_nearest(palette, 0, vector3(0.5f, 0.5f, 0.5f));
    EXPECT_GE(result.index, 0);
    EXPECT_LE(result.index, 254);
}

TEST_F(PaletteTest, BlackInputReturnsDarkColor) {
    // The nearest palette color to linear black should itself be very dark.
    auto result = palette_get_nearest(palette, 0, vector3(0.0f, 0.0f, 0.0f));
    Color const nearest = palette.colors[result.index];
    // In sRGB the returned color should be dark (all channels < 100).
    EXPECT_LT(nearest.r, 100);
    EXPECT_LT(nearest.g, 100);
    EXPECT_LT(nearest.b, 100);
}

TEST_F(PaletteTest, WhiteInputReturnsBrightColor) {
    auto result = palette_get_nearest(palette, 0, vector3(1.0f, 1.0f, 1.0f));
    Color const nearest = palette.colors[result.index];
    EXPECT_GT(nearest.r, 150);
    EXPECT_GT(nearest.g, 150);
    EXPECT_GT(nearest.b, 150);
}

TEST_F(PaletteTest, ErrorIsZeroForExactPaletteColor) {
    // If we look up a color that IS in the palette exactly, the error should
    // be essentially zero.
    Color const exact = palette.colors[11]; // sRGB (35, 51, 51)
    Vector3 const linear = vector_from_color(exact);
    auto result = palette_get_nearest(palette, 0, linear);
    EXPECT_NEAR(result.error.x, 0.0f, kEps);
    EXPECT_NEAR(result.error.y, 0.0f, kEps);
    EXPECT_NEAR(result.error.z, 0.0f, kEps);
}

TEST_F(PaletteTest, RemapRegionReturnsRemapIndex) {
    // Region 1 is the primary remap (player color). Its palette indices
    // are 243–254. The returned index must fall in that range.
    auto result = palette_get_nearest(palette, 1, vector3(0.5f, 0.5f, 0.5f));
    EXPECT_GE(result.index, 243);
    EXPECT_LE(result.index, 254);
}

TEST_F(PaletteTest, DifferentRegionsReturnDifferentIndices) {
    // The same input color looked up in different regions should generally
    // return different palette indices.
    Vector3 const mid_grey = vector3(0.5f, 0.5f, 0.5f);
    auto r0 = palette_get_nearest(palette, 0, mid_grey);
    auto r1 = palette_get_nearest(palette, 1, mid_grey);
    EXPECT_NE(r0.index, r1.index);
}
