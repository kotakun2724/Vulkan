cmake_minimum_required(VERSION 3.20)
project(vulkan_triangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(vulkan_terrain
    src/main.cpp
    src/math_utils.cpp
    src/vulkan_helpers.cpp
)

target_compile_definitions(vulkan_terrain PRIVATE GLFW_INCLUDE_VULKAN)
target_include_directories(vulkan_terrain PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_terrain PRIVATE glfw Vulkan::Vulkan)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BUILD_DIR})

add_custom_command(
    OUTPUT ${SHADER_BUILD_DIR}/triangle.vert.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SHADER_BUILD_DIR}/triangle.vert.spv ${SHADER_DIR}/triangle.vert
    DEPENDS ${SHADER_DIR}/triangle.vert
)

add_custom_command(
    OUTPUT ${SHADER_BUILD_DIR}/triangle.frag.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SHADER_BUILD_DIR}/triangle.frag.spv ${SHADER_DIR}/triangle.frag
    DEPENDS ${SHADER_DIR}/triangle.frag
)

add_custom_target(shaders ALL
    DEPENDS
        ${SHADER_BUILD_DIR}/triangle.vert.spv
        ${SHADER_BUILD_DIR}/triangle.frag.spv
)

add_dependencies(vulkan_terrain shaders)

target_compile_definitions(vulkan_terrain PRIVATE
    SHADER_VERT_PATH="${SHADER_BUILD_DIR}/triangle.vert.spv"
    SHADER_FRAG_PATH="${SHADER_BUILD_DIR}/triangle.frag.spv"
)

# ---------------------------------------------------------------------------
#  Ray tracing LT demo (software ray tracer running on macOS/MoltenVK)
# ---------------------------------------------------------------------------
add_executable(vulkan_raytracer
    src/raytracer_main.cpp
    src/math_utils.cpp
    src/vulkan_helpers.cpp
)

target_compile_definitions(vulkan_raytracer PRIVATE GLFW_INCLUDE_VULKAN)
target_include_directories(vulkan_raytracer PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_raytracer PRIVATE glfw Vulkan::Vulkan)

add_custom_command(
    OUTPUT ${SHADER_BUILD_DIR}/raytracer.vert.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SHADER_BUILD_DIR}/raytracer.vert.spv ${SHADER_DIR}/raytracer.vert
    DEPENDS ${SHADER_DIR}/raytracer.vert
)

add_custom_command(
    OUTPUT ${SHADER_BUILD_DIR}/raytracer.frag.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SHADER_BUILD_DIR}/raytracer.frag.spv ${SHADER_DIR}/raytracer.frag
    DEPENDS ${SHADER_DIR}/raytracer.frag
)

add_custom_target(raytracer_shaders ALL
    DEPENDS
        ${SHADER_BUILD_DIR}/raytracer.vert.spv
        ${SHADER_BUILD_DIR}/raytracer.frag.spv
)

add_dependencies(vulkan_raytracer raytracer_shaders)

target_compile_definitions(vulkan_raytracer PRIVATE
    SHADER_RT_VERT_PATH="${SHADER_BUILD_DIR}/raytracer.vert.spv"
    SHADER_RT_FRAG_PATH="${SHADER_BUILD_DIR}/raytracer.frag.spv"
)

