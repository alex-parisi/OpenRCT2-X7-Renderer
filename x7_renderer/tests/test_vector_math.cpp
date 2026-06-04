#include "VectorMath.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace RCTGen;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {
    constexpr float kEps = 1e-5f;

    void ExpectVec3Near(Vector3 actual, Vector3 expected, float eps = kEps) {
        EXPECT_NEAR(actual.x, expected.x, eps);
        EXPECT_NEAR(actual.y, expected.y, eps);
        EXPECT_NEAR(actual.z, expected.z, eps);
    }

    void ExpectMat3Near(Matrix3 actual, Matrix3 expected, float eps = kEps) {
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                EXPECT_NEAR(actual(r, c), expected(r, c), eps) << "at (" << r << "," << c << ")";
    }
} // namespace

// ---------------------------------------------------------------------------
// Vector2
// ---------------------------------------------------------------------------

TEST(Vector2, Arithmetic) {
    auto a = vector2(1.0f, 2.0f);
    auto b = vector2(3.0f, 4.0f);

    auto sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 4.0f);
    EXPECT_FLOAT_EQ(sum.y, 6.0f);

    auto diff = b - a;
    EXPECT_FLOAT_EQ(diff.x, 2.0f);
    EXPECT_FLOAT_EQ(diff.y, 2.0f);

    auto scaled = a * 3.0f;
    EXPECT_FLOAT_EQ(scaled.x, 3.0f);
    EXPECT_FLOAT_EQ(scaled.y, 6.0f);
}

TEST(Vector2, Dot) {
    EXPECT_FLOAT_EQ(vector2(1, 0).dot(vector2(0, 1)), 0.0f);  // perpendicular
    EXPECT_FLOAT_EQ(vector2(3, 4).dot(vector2(3, 4)), 25.0f); // self = squared norm
    EXPECT_FLOAT_EQ(vector2(1, 2).dot(vector2(3, 4)), 11.0f);
}

TEST(Vector2, Norm) {
    EXPECT_NEAR(vector2(3.0f, 4.0f).norm(), 5.0f, kEps);
    EXPECT_NEAR(vector2(1.0f, 0.0f).norm(), 1.0f, kEps);
    EXPECT_NEAR(vector2(0.0f, 0.0f).norm(), 0.0f, kEps);
}

// ---------------------------------------------------------------------------
// Vector3
// ---------------------------------------------------------------------------

TEST(Vector3, Arithmetic) {
    auto a = vector3(1, 2, 3);
    auto b = vector3(4, 5, 6);

    auto sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 5.0f);
    EXPECT_FLOAT_EQ(sum.y, 7.0f);
    EXPECT_FLOAT_EQ(sum.z, 9.0f);

    auto diff = b - a;
    EXPECT_FLOAT_EQ(diff.x, 3.0f);
    EXPECT_FLOAT_EQ(diff.y, 3.0f);
    EXPECT_FLOAT_EQ(diff.z, 3.0f);

    auto scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x, 2.0f);
    EXPECT_FLOAT_EQ(scaled.y, 4.0f);
    EXPECT_FLOAT_EQ(scaled.z, 6.0f);
}

TEST(Vector3, Dot) {
    // Basis vectors are mutually perpendicular.
    EXPECT_FLOAT_EQ(vector3(1, 0, 0).dot(vector3(0, 1, 0)), 0.0f);
    EXPECT_FLOAT_EQ(vector3(1, 0, 0).dot(vector3(0, 0, 1)), 0.0f);
    EXPECT_FLOAT_EQ(vector3(0, 1, 0).dot(vector3(0, 0, 1)), 0.0f);
    // Self-dot equals squared norm.
    EXPECT_FLOAT_EQ(vector3(1, 2, 2).dot(vector3(1, 2, 2)), 9.0f);
}

TEST(Vector3, Cross) {
    // Standard basis cross products.
    ExpectVec3Near(vector3(1, 0, 0).cross(vector3(0, 1, 0)), vector3(0, 0, 1));
    ExpectVec3Near(vector3(0, 1, 0).cross(vector3(0, 0, 1)), vector3(1, 0, 0));
    ExpectVec3Near(vector3(0, 0, 1).cross(vector3(1, 0, 0)), vector3(0, 1, 0));

    // Anticommutativity: a × b = -(b × a).
    auto a = vector3(1, 2, 3);
    auto b = vector3(4, 5, 6);
    auto ab = a.cross(b);
    auto ba = b.cross(a);
    ExpectVec3Near(ab, ba * -1.0f);

    // Parallel vectors produce zero cross product.
    ExpectVec3Near(vector3(1, 2, 3).cross(vector3(2, 4, 6)), vector3(0, 0, 0));
}

TEST(Vector3, Norm) {
    EXPECT_NEAR(vector3(1, 2, 2).norm(), 3.0f, kEps);
    EXPECT_NEAR(vector3(1, 0, 0).norm(), 1.0f, kEps);
}

TEST(Vector3, Normalized) {
    auto v = vector3(3, 4, 0).normalized();
    EXPECT_NEAR(v.norm(), 1.0f, kEps);
    EXPECT_NEAR(v.x, 0.6f, kEps);
    EXPECT_NEAR(v.y, 0.8f, kEps);
    EXPECT_NEAR(v.z, 0.0f, kEps);
}

TEST(Vector3, Splat) {
    auto v = Vector3::splat(3.14f);
    EXPECT_FLOAT_EQ(v.x, 3.14f);
    EXPECT_FLOAT_EQ(v.y, 3.14f);
    EXPECT_FLOAT_EQ(v.z, 3.14f);
}

// ---------------------------------------------------------------------------
// Matrix3
// ---------------------------------------------------------------------------

TEST(Matrix3, IdentityMultiplication) {
    Matrix3 const I = matrix_identity();
    auto v = vector3(1.0f, 2.0f, 3.0f);
    ExpectVec3Near(matrix_vector(I, v), v);

    Matrix3 const m = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9);
    ExpectMat3Near(matrix_mult(I, m), m);
    ExpectMat3Near(matrix_mult(m, I), m);
}

TEST(Matrix3, Determinant) {
    EXPECT_NEAR(matrix_identity().determinant(), 1.0f, kEps);

    // det([[1,2,3],[0,1,4],[5,6,0]]) = -24 + 40 - 15 = 1
    Matrix3 const m = matrix(1, 2, 3, 0, 1, 4, 5, 6, 0);
    EXPECT_NEAR(m.determinant(), 1.0f, kEps);

    // Scaling matrix: det = s^3
    float const s = 2.0f;
    Matrix3 const scale = matrix(s, 0, 0, 0, s, 0, 0, 0, s);
    EXPECT_NEAR(scale.determinant(), s * s * s, kEps);
}

TEST(Matrix3, InverseIsIdentity) {
    // A * inv(A) should equal identity.
    Matrix3 const m = matrix(1, 2, 3, 0, 1, 4, 5, 6, 0);
    ExpectMat3Near(matrix_mult(m, matrix_inverse(m)), matrix_identity());
    ExpectMat3Near(matrix_mult(matrix_inverse(m), m), matrix_identity());
}

TEST(Matrix3, InverseOfIdentity) { ExpectMat3Near(matrix_inverse(matrix_identity()), matrix_identity()); }

TEST(Matrix3, Transpose) {
    Matrix3 m = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9);
    Matrix3 mt = matrix_transpose(m);

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c) EXPECT_FLOAT_EQ(mt(r, c), m(c, r));

    // Double-transpose recovers original.
    ExpectMat3Near(matrix_transpose(mt), m);
}

TEST(Matrix3, Multiply) {
    // Verify a known product.
    Matrix3 const a = matrix(1, 0, 2, 0, 3, 0, 4, 0, 5);
    Matrix3 const b = matrix(1, 2, 3, 0, 1, 0, 4, 0, 1);
    // Row 0: (1,0,2)*(col0=(1,0,4), col1=(2,1,0), col2=(3,0,1))
    //        → (1+0+8, 2+0+0, 3+0+2) = (9,2,5)
    // Row 1: (0,3,0)*(same cols)
    //        → (0,3,0) = (0,3,0)
    // Row 2: (4,0,5)*(same cols)
    //        → (4+0+20, 8+0+0, 12+0+5) = (24,8,17)
    Matrix3 const expected = matrix(9, 2, 5, 0, 3, 0, 24, 8, 17);
    ExpectMat3Near(matrix_mult(a, b), expected);
}

TEST(Matrix3, VectorMultiply) {
    Matrix3 const m = matrix(1, 2, 0, 0, 3, 4, 5, 0, 6);
    auto v = vector3(1, 2, 3);
    // Row 0: 1*1 + 2*2 + 0*3 = 5
    // Row 1: 0*1 + 3*2 + 4*3 = 18
    // Row 2: 5*1 + 0*2 + 6*3 = 23
    ExpectVec3Near(matrix_vector(m, v), vector3(5, 18, 23));
}

// ---------------------------------------------------------------------------
// Rotation matrices
// ---------------------------------------------------------------------------

TEST(Rotation, RotateX) {
    // rotate_x(0) = identity
    ExpectMat3Near(rotate_x(0.0f), matrix_identity());

    // rotate_x(pi/2): Y -> Z
    float const pi2 = std::numbers::pi_v<float> / 2.0f;
    Matrix3 const r = rotate_x(pi2);
    ExpectVec3Near(matrix_vector(r, vector3(1, 0, 0)), vector3(1, 0, 0));  // X unchanged
    ExpectVec3Near(matrix_vector(r, vector3(0, 1, 0)), vector3(0, 0, 1));  // Y -> Z
    ExpectVec3Near(matrix_vector(r, vector3(0, 0, 1)), vector3(0, -1, 0)); // Z -> -Y
}

TEST(Rotation, RotateY) {
    ExpectMat3Near(rotate_y(0.0f), matrix_identity());

    float const pi2 = std::numbers::pi_v<float> / 2.0f;
    Matrix3 const r = rotate_y(pi2);
    ExpectVec3Near(matrix_vector(r, vector3(0, 1, 0)), vector3(0, 1, 0));  // Y unchanged
    ExpectVec3Near(matrix_vector(r, vector3(0, 0, 1)), vector3(1, 0, 0));  // Z -> X
    ExpectVec3Near(matrix_vector(r, vector3(1, 0, 0)), vector3(0, 0, -1)); // X -> -Z
}

TEST(Rotation, RotateZ) {
    ExpectMat3Near(rotate_z(0.0f), matrix_identity());

    float const pi2 = std::numbers::pi_v<float> / 2.0f;
    Matrix3 const r = rotate_z(pi2);
    ExpectVec3Near(matrix_vector(r, vector3(0, 0, 1)), vector3(0, 0, 1));  // Z unchanged
    ExpectVec3Near(matrix_vector(r, vector3(1, 0, 0)), vector3(0, 1, 0));  // X -> Y
    ExpectVec3Near(matrix_vector(r, vector3(0, 1, 0)), vector3(-1, 0, 0)); // Y -> -X
}

TEST(Rotation, OrthogonalAndRightHanded) {
    // For any rotation R: R^T = R^-1, and det(R) = 1.
    float const angle = 1.234f;
    for (auto R : {rotate_x(angle), rotate_y(angle), rotate_z(angle)}) {
        // R * R^T should be identity.
        ExpectMat3Near(matrix_mult(R, matrix_transpose(R)), matrix_identity());
        // det = 1.
        EXPECT_NEAR(R.determinant(), 1.0f, kEps);
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

TEST(Transform, IdentityTransform) {
    Transform const t = transform(matrix_identity(), vector3(0, 0, 0));
    auto v = vector3(1, 2, 3);
    ExpectVec3Near(transform_vector(t, v), v);
}

TEST(Transform, Translation) {
    Transform const t = transform(matrix_identity(), vector3(1, 2, 3));
    ExpectVec3Near(transform_vector(t, vector3(0, 0, 0)), vector3(1, 2, 3));
    ExpectVec3Near(transform_vector(t, vector3(1, 0, 0)), vector3(2, 2, 3));
}

TEST(Transform, RotationAndTranslation) {
    float const pi2 = std::numbers::pi_v<float> / 2.0f;
    Transform const t = transform(rotate_z(pi2), vector3(5, 0, 0));
    // rotate_z(pi/2) maps (1,0,0) -> (0,1,0), then translate by (5,0,0)
    ExpectVec3Near(transform_vector(t, vector3(1, 0, 0)), vector3(5, 1, 0));
}

TEST(Transform, Compose) {
    // Two translations: T1=(1,0,0), T2=(0,2,0), compose = (1,2,0).
    Transform const t1 = transform(matrix_identity(), vector3(1, 0, 0));
    Transform const t2 = transform(matrix_identity(), vector3(0, 2, 0));
    Transform const t12 = transform_compose(t1, t2);
    ExpectVec3Near(transform_vector(t12, vector3(0, 0, 0)), vector3(1, 2, 0));
}
