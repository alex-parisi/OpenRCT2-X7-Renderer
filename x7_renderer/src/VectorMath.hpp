/// VectorMath.hpp

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace RCTGen {
    struct Vector2 {
        float x;
        float y;

        constexpr Vector2 operator+(Vector2 rhs) const noexcept { return {x + rhs.x, y + rhs.y}; }
        constexpr Vector2 operator-(Vector2 rhs) const noexcept { return {x - rhs.x, y - rhs.y}; }
        constexpr Vector2 operator*(float s) const noexcept { return {x * s, y * s}; }
        constexpr float dot(Vector2 rhs) const noexcept { return x * rhs.x + y * rhs.y; }
        [[nodiscard]] float norm() const noexcept { return std::sqrt(x * x + y * y); }
    };

    struct Vector3 {
        float x;
        float y;
        float z;

        static constexpr Vector3 splat(float v) noexcept { return {v, v, v}; }

        constexpr Vector3 operator+(Vector3 rhs) const noexcept { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
        constexpr Vector3 operator-(Vector3 rhs) const noexcept { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
        constexpr Vector3 operator*(float s) const noexcept { return {x * s, y * s, z * s}; }
        constexpr float dot(Vector3 rhs) const noexcept { return x * rhs.x + y * rhs.y + z * rhs.z; }

        constexpr Vector3 cross(Vector3 rhs) const noexcept {
            return {y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x};
        }

        [[nodiscard]] float norm() const noexcept { return std::sqrt(dot(*this)); }

        [[nodiscard]] Vector3 normalized() const noexcept {
            const float n = norm();
            return *this * (1.0f / n);
        }
    };

    struct Matrix3 {
        std::array<float, 9> entries;

        constexpr float& operator()(std::size_t row, std::size_t col) noexcept { return entries[3 * row + col]; }
        constexpr float operator()(std::size_t row, std::size_t col) const noexcept { return entries[3 * row + col]; }

        [[nodiscard]] constexpr float determinant() const noexcept {
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
                   - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
                   + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        }
    };

    struct Transform {
        Matrix3 matrix;
        Vector3 translation;
    };

    // Constructor-style helpers (constexpr where possible). Kept as
    // lowercase free functions because every math expression in the
    // renderer / track-render / rct2-ride-gen calls them by these names;
    // renaming would textually edit hundreds of math-critical sites that
    // must stay byte-equivalent to the goldens.
    constexpr Vector2 vector2(float x, float y) noexcept { return {x, y}; }
    constexpr Vector3 vector3(float x, float y, float z) noexcept { return {x, y, z}; }
    constexpr Vector3 vector3_from_scalar(float a) noexcept { return Vector3::splat(a); }

    constexpr Vector2 vector2_add(Vector2 a, Vector2 b) noexcept { return a + b; }
    constexpr Vector2 vector2_sub(Vector2 a, Vector2 b) noexcept { return a - b; }
    constexpr Vector2 vector2_mult(Vector2 a, float b) noexcept { return a * b; }
    constexpr float vector2_dot(Vector2 a, Vector2 b) noexcept { return a.dot(b); }
    inline float vector2_norm(Vector2 a) noexcept { return a.norm(); }

    constexpr Vector3 vector3_add(Vector3 a, Vector3 b) noexcept { return a + b; }
    constexpr Vector3 vector3_sub(Vector3 a, Vector3 b) noexcept { return a - b; }
    constexpr Vector3 vector3_mult(Vector3 a, float b) noexcept { return a * b; }
    constexpr float vector3_dot(Vector3 a, Vector3 b) noexcept { return a.dot(b); }
    constexpr Vector3 vector3_cross(Vector3 a, Vector3 b) noexcept { return a.cross(b); }
    inline float vector3_norm(Vector3 a) noexcept { return a.norm(); }
    inline Vector3 vector3_normalize(Vector3 a) noexcept { return a.normalized(); }

    constexpr Matrix3 matrix(float a, float b, float c, float d, float e, float f, float g, float h, float i) noexcept {
        return Matrix3{{a, b, c, d, e, f, g, h, i}};
    }

    constexpr Matrix3 matrix_identity() noexcept { return matrix(1, 0, 0, 0, 1, 0, 0, 0, 1); }
    constexpr float matrix_determinant(Matrix3 m) noexcept { return m.determinant(); }

    constexpr Matrix3 matrix_inverse(Matrix3 m) noexcept {
        const float d = m.determinant();
        return matrix((m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d, (m(0, 2) * m(2, 1) - m(2, 2) * m(0, 1)) / d,
                      (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2)) / d, (m(1, 2) * m(2, 0) - m(2, 2) * m(1, 0)) / d,
                      (m(0, 0) * m(2, 2) - m(2, 0) * m(0, 2)) / d, (m(0, 2) * m(1, 0) - m(1, 2) * m(0, 0)) / d,
                      (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) / d, (m(0, 1) * m(2, 0) - m(2, 1) * m(0, 0)) / d,
                      (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) / d);
    }

    constexpr Matrix3 matrix_transpose(Matrix3 m) noexcept {
        return matrix(m(0, 0), m(1, 0), m(2, 0), m(0, 1), m(1, 1), m(2, 1), m(0, 2), m(1, 2), m(2, 2));
    }

    constexpr Matrix3 matrix_mult(Matrix3 a, Matrix3 b) noexcept {
        Matrix3 result{{}};
        for (std::size_t col = 0; col < 3; ++col)
            for (std::size_t row = 0; row < 3; ++row)
                result(row, col) = a(row, 0) * b(0, col) + a(row, 1) * b(1, col) + a(row, 2) * b(2, col);
        return result;
    }

    constexpr Vector3 matrix_vector(Matrix3 m, Vector3 v) noexcept {
        return vector3(m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z, m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z,
                       m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z);
    }

    constexpr Transform transform(Matrix3 m, Vector3 v) noexcept { return {m, v}; }

    constexpr Transform transform_compose(Transform a, Transform b) noexcept {
        return {matrix_mult(a.matrix, b.matrix), matrix_vector(a.matrix, b.translation) + a.translation};
    }

    constexpr Vector3 transform_vector(Transform t, Vector3 v) noexcept {
        return matrix_vector(t.matrix, v) + t.translation;
    }

    inline Matrix3 rotate_x(float theta) noexcept {
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        return matrix(1, 0, 0, 0, c, -s, 0, s, c);
    }

    inline Matrix3 rotate_y(float theta) noexcept {
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        return matrix(c, 0, s, 0, 1, 0, -s, 0, c);
    }

    inline Matrix3 rotate_z(float theta) noexcept {
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        return matrix(c, -s, 0, s, c, 0, 0, 0, 1);
    }
} // namespace RCTGen