#pragma once

#include <array>

namespace math {
using Mat4 = std::array<float, 16>;

Mat4 IdentityMatrix();
Mat4 RotationYMatrix(float degrees);
Mat4 TranslationMatrix(float x, float y, float z);
Mat4 PerspectiveMatrix(float fovDegrees, float aspect, float zNear, float zFar);
Mat4 MultiplyMatrix(const Mat4& a, const Mat4& b);
}  // namespace math

