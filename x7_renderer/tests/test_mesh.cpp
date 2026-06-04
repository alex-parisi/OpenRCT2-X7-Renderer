#include "Mesh.hpp"
#include "VectorMath.hpp"
#include <gtest/gtest.h>
#include <span>
#include <vector>

using namespace RCTGen;

namespace {
    constexpr float kEps = 1e-6f;

    // Build a Texture from a std::vector so span lifetime is tied to the vector.
    std::pair<std::vector<Vector3>, Texture>
    make_texture(std::uint16_t w, std::uint16_t h, std::initializer_list<Vector3> pixels) {
        std::vector<Vector3> buf(pixels);
        Texture const tex{w, h, std::span<const Vector3>(buf)};
        return {std::move(buf), tex};
    }
} // namespace

// ---------------------------------------------------------------------------
// Single-pixel texture
// ---------------------------------------------------------------------------

TEST(TextureSample, SinglePixelAlwaysReturnsItself) {
    auto [buf, tex] = make_texture(1, 1, {{1.0f, 0.5f, 0.25f}});
    for (float const u : {0.0f, 0.5f, 0.99f, 1.0f, 1.5f, 2.0f, -0.5f}) {
        for (float const v : {0.0f, 0.5f, 0.99f}) {
            Vector3 const s = texture_sample(tex, vector2(u, v));
            EXPECT_NEAR(s.x, 1.0f, kEps) << "u=" << u << " v=" << v;
            EXPECT_NEAR(s.y, 0.5f, kEps) << "u=" << u << " v=" << v;
            EXPECT_NEAR(s.z, 0.25f, kEps) << "u=" << u << " v=" << v;
        }
    }
}

// ---------------------------------------------------------------------------
// 2×2 texture: quadrant lookup
//
//   (0,0)→red   (0.5,0)→green
//   (0,0.5)→blue  (0.5,0.5)→yellow
// ---------------------------------------------------------------------------

TEST(TextureSample, QuadrantLookup) {
    // pixels stored row-major: [0,0], [1,0], [0,1], [1,1]
    auto [buf, tex] = make_texture(2, 2,
                                   {
                                       {1, 0, 0}, // (0,0) red
                                       {0, 1, 0}, // (1,0) green
                                       {0, 0, 1}, // (0,1) blue
                                       {1, 1, 0}, // (1,1) yellow
                                   });

    auto sample = [&](float u, float v) { return texture_sample(tex, vector2(u, v)); };

    Vector3 const red = sample(0.0f, 0.0f);
    Vector3 const green = sample(0.5f, 0.0f);
    Vector3 const blue = sample(0.0f, 0.5f);
    Vector3 const yellow = sample(0.5f, 0.5f);

    EXPECT_NEAR(red.x, 1.0f, kEps);
    EXPECT_NEAR(red.y, 0.0f, kEps);
    EXPECT_NEAR(red.z, 0.0f, kEps);
    EXPECT_NEAR(green.x, 0.0f, kEps);
    EXPECT_NEAR(green.y, 1.0f, kEps);
    EXPECT_NEAR(green.z, 0.0f, kEps);
    EXPECT_NEAR(blue.x, 0.0f, kEps);
    EXPECT_NEAR(blue.y, 0.0f, kEps);
    EXPECT_NEAR(blue.z, 1.0f, kEps);
    EXPECT_NEAR(yellow.x, 1.0f, kEps);
    EXPECT_NEAR(yellow.y, 1.0f, kEps);
    EXPECT_NEAR(yellow.z, 0.0f, kEps);
}

// ---------------------------------------------------------------------------
// UV wrapping
// ---------------------------------------------------------------------------

TEST(TextureSample, WrapAtExactlyOne) {
    // u=1.0 should wrap to pixel column 0 (same as u=0.0).
    auto [buf, tex] = make_texture(2, 2,
                                   {
                                       {1, 0, 0},
                                       {0, 1, 0},
                                       {0, 0, 1},
                                       {1, 1, 0},
                                   });
    auto red_at_0 = texture_sample(tex, vector2(0.0f, 0.0f));
    auto red_at_1 = texture_sample(tex, vector2(1.0f, 0.0f));
    EXPECT_NEAR(red_at_0.x, red_at_1.x, kEps);
    EXPECT_NEAR(red_at_0.y, red_at_1.y, kEps);
    EXPECT_NEAR(red_at_0.z, red_at_1.z, kEps);
}

TEST(TextureSample, WrapAboveOne) {
    // u=1.5 wraps to the same pixel as u=0.5.
    auto [buf, tex] = make_texture(2, 2,
                                   {
                                       {1, 0, 0},
                                       {0, 1, 0},
                                       {0, 0, 1},
                                       {1, 1, 0},
                                   });
    auto at_half = texture_sample(tex, vector2(0.5f, 0.0f));
    auto at_1half = texture_sample(tex, vector2(1.5f, 0.0f));
    EXPECT_NEAR(at_half.x, at_1half.x, kEps);
    EXPECT_NEAR(at_half.y, at_1half.y, kEps);
    EXPECT_NEAR(at_half.z, at_1half.z, kEps);
}

TEST(TextureSample, WrapNegativeUV) {
    // u=-0.5 → frac = -0.5 - floor(-0.5) = -0.5 + 1.0 = 0.5, same as u=0.5.
    auto [buf, tex] = make_texture(2, 2,
                                   {
                                       {1, 0, 0},
                                       {0, 1, 0},
                                       {0, 0, 1},
                                       {1, 1, 0},
                                   });
    auto at_half = texture_sample(tex, vector2(0.5f, 0.0f));
    auto at_neg = texture_sample(tex, vector2(-0.5f, 0.0f));
    EXPECT_NEAR(at_half.x, at_neg.x, kEps);
    EXPECT_NEAR(at_half.y, at_neg.y, kEps);
    EXPECT_NEAR(at_half.z, at_neg.z, kEps);
}

// ---------------------------------------------------------------------------
// 4×1 texture: verify per-column sampling across a wider range
// ---------------------------------------------------------------------------

TEST(TextureSample, FourColumnTexture) {
    auto [buf, tex] = make_texture(4, 1,
                                   {
                                       {1, 0, 0}, // col 0: red     u ∈ [0.00, 0.25)
                                       {0, 1, 0}, // col 1: green   u ∈ [0.25, 0.50)
                                       {0, 0, 1}, // col 2: blue    u ∈ [0.50, 0.75)
                                       {1, 1, 0}, // col 3: yellow  u ∈ [0.75, 1.00)
                                   });

    auto s0 = texture_sample(tex, vector2(0.1f, 0.0f));
    EXPECT_NEAR(s0.x, 1.0f, kEps);
    EXPECT_NEAR(s0.y, 0.0f, kEps); // red

    auto s1 = texture_sample(tex, vector2(0.3f, 0.0f));
    EXPECT_NEAR(s1.x, 0.0f, kEps);
    EXPECT_NEAR(s1.y, 1.0f, kEps); // green

    auto s2 = texture_sample(tex, vector2(0.6f, 0.0f));
    EXPECT_NEAR(s2.z, 1.0f, kEps);
    EXPECT_NEAR(s2.x, 0.0f, kEps); // blue

    auto s3 = texture_sample(tex, vector2(0.8f, 0.0f));
    EXPECT_NEAR(s3.x, 1.0f, kEps);
    EXPECT_NEAR(s3.y, 1.0f, kEps); // yellow
}
