#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkhelpers {
VkCommandBuffer BeginSingleTimeCommands(VkDevice device,
                                        VkCommandPool commandPool);
void EndSingleTimeCommands(VkDevice device, VkQueue queue,
                           VkCommandPool commandPool,
                           VkCommandBuffer commandBuffer);
void CopyBuffer(VkDevice device, VkQueue queue, VkCommandPool commandPool,
                VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

void CreateBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                  VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties, VkBuffer& buffer,
                  VkDeviceMemory& bufferMemory);

void CreateImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width,
                 uint32_t height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                 VkImage& image, VkDeviceMemory& imageMemory);

VkImageView CreateImageView(VkDevice device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspectFlags);

uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties);
VkFormat FindSupportedFormat(VkPhysicalDevice physicalDevice,
                             const std::vector<VkFormat>& candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features);
}  // namespace vkhelpers

