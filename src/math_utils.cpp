#include "math_utils.h"

#include <cmath>

namespace math {
Mat4 IdentityMatrix() {
    return {
        1.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 1.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 1.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 1.0f,  //
    };
}

Mat4 RotationYMatrix(float degrees) {
    constexpr float kPi = 3.1415926535f;
    float r = degrees * kPi / 180.0f;
    float c = std::cos(r);
    float s = std::sin(r);
    return {
        c,  0.0f, s, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        -s, 0.0f, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
}

Mat4 TranslationMatrix(float x, float y, float z) {
    Mat4 matrix = IdentityMatrix();
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    return matrix;
}

Mat4 PerspectiveMatrix(float fovDegrees, float aspect, float zNear, float zFar) {
    constexpr float kPi = 3.1415926535f;
    float radians = fovDegrees * kPi / 180.0f;
    float tanHalf = std::tan(radians / 2.0f);

    Mat4 matrix{};
    matrix[0] = 1.0f / (aspect * tanHalf);
    matrix[5] = -1.0f / tanHalf;  // Flip Y for Vulkan clip space
    matrix[10] = zFar / (zNear - zFar);
    matrix[11] = -1.0f;
    matrix[14] = (zNear * zFar) / (zNear - zFar);
    return matrix;
}

Mat4 MultiplyMatrix(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            result[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    return result;
}
}  // namespace math

