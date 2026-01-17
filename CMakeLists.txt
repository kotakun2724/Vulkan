cmake_minimum_required(VERSION 3.20)
project(vulkan_triangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(vulkan_triangle
    src/main.cpp
)

target_compile_definitions(vulkan_triangle PRIVATE GLFW_INCLUDE_VULKAN)
target_include_directories(vulkan_triangle PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_triangle PRIVATE glfw Vulkan::Vulkan)

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

add_dependencies(vulkan_triangle shaders)

target_compile_definitions(vulkan_triangle PRIVATE
    SHADER_VERT_PATH="${SHADER_BUILD_DIR}/triangle.vert.spv"
    SHADER_FRAG_PATH="${SHADER_BUILD_DIR}/triangle.frag.spv"
)

