// ===========================================================================
//  Vulkan によるレイトレーシング - LT デモ (macOS / MoltenVK 対応)
//
//  方針:
//    MoltenVK は VK_KHR_ray_tracing_pipeline / VK_KHR_acceleration_structure を
//    サポートしないため、ここではフラグメントシェーダ内で手書きのレイ計算を行う
//    「ソフトウェアレイトレーサ」を作る。
//    GPU 側 (raytracer.frag) で各 rayGen / intersection / closestHit / miss
//    に相当するブロックを分けて書いており、本ファイルはそれを動かすための
//    最小 Vulkan ホストコード (ウィンドウ + スワップチェーン + パイプライン +
//    1 つの UBO + 三角形 1 枚の描画) のみを担う。
// ===========================================================================

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vulkan_helpers.h"

namespace {

constexpr uint32_t kWidth = 960;
constexpr uint32_t kHeight = 600;
constexpr int kMaxFramesInFlight = 2;
constexpr float kCameraOrbitSpeed = 0.35f;  // radians per second

const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

// Must match the std140 layout of `RTUbo` in raytracer.frag.
struct RTUbo {
    float time;
    float aspect;
    float cameraYaw;
    float _pad;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool IsComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

class RayTracerApp {
   public:
    void Run() {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
    }

   private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages_;
    VkFormat swapChainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent_{};
    std::vector<VkImageView> swapChainImageViews_;
    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool framebufferResized_ = false;

    std::chrono::steady_clock::time_point startTime_;

    static void FramebufferResizeCallback(GLFWwindow* window, int, int) {
        auto* app = reinterpret_cast<RayTracerApp*>(
            glfwGetWindowUserPointer(window));
        app->framebufferResized_ = true;
    }

    void InitWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to init GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(kWidth, kHeight,
                                   "Vulkan Ray Tracing (LT demo)",
                                   nullptr, nullptr);
        if (!window_) {
            throw std::runtime_error("Failed to create window");
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, FramebufferResizeCallback);
    }

    void InitVulkan() {
        CreateInstance();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
        startTime_ = std::chrono::steady_clock::now();
    }

    void MainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            DrawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void Cleanup() {
        CleanupSwapChain();

        for (size_t i = 0; i < uniformBuffers_.size(); ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
        }

        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);

        for (size_t i = 0; i < imageAvailableSemaphores_.size(); ++i) {
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        for (size_t i = 0; i < renderFinishedSemaphores_.size(); ++i) {
            vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
        }

        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);

        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    // -----------------------------------------------------------------
    //  Instance / surface / device
    // -----------------------------------------------------------------
    bool CheckValidationLayerSupport() {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> available(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, available.data());
        for (const char* name : kValidationLayers) {
            bool found = false;
            for (const auto& props : available) {
                if (std::strcmp(props.layerName, name) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    std::vector<const char*> GetRequiredInstanceExtensions() {
        uint32_t glfwExtCount = 0;
        const char** glfwExt =
            glfwGetRequiredInstanceExtensions(&glfwExtCount);
        std::vector<const char*> extensions(glfwExt, glfwExt + glfwExtCount);
        // macOS/MoltenVK: must enumerate non-conformant portability ICDs.
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        return extensions;
    }

    void CreateInstance() {
        if (kEnableValidationLayers && !CheckValidationLayerSupport()) {
            std::cerr << "Validation layers requested but not available; "
                         "continuing without them.\n";
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Ray Tracing LT";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        auto extensions = GetRequiredInstanceExtensions();

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (kEnableValidationLayers && CheckValidationLayerSupport()) {
            createInfo.enabledLayerCount =
                static_cast<uint32_t>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to create Vulkan instance (VkResult=" +
                std::to_string(static_cast<int>(result)) + ")");
        }
    }

    void CreateSurface() {
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count,
                                                 families.data());
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_,
                                                 &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (indices.IsComplete()) break;
        }
        return indices;
    }

    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_,
                                                  &details.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount,
                                             nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device, surface_, &formatCount, details.formats.data());
        }

        uint32_t presentCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_,
                                                  &presentCount, nullptr);
        if (presentCount != 0) {
            details.presentModes.resize(presentCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface_, &presentCount, details.presentModes.data());
        }
        return details;
    }

    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                             available.data());
        std::set<std::string> required = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (const auto& ext : available) {
            required.erase(ext.extensionName);
        }
        return required.empty();
    }

    bool IsDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = FindQueueFamilies(device);
        bool extOk = CheckDeviceExtensionSupport(device);
        bool swapOk = false;
        if (extOk) {
            auto details = QuerySwapChainSupport(device);
            swapOk = !details.formats.empty() && !details.presentModes.empty();
        }
        return indices.IsComplete() && extOk && swapOk;
    }

    void PickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) {
            throw std::runtime_error("No Vulkan-capable GPU found");
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());
        for (const auto& dev : devices) {
            if (IsDeviceSuitable(dev)) {
                physicalDevice_ = dev;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error("No suitable GPU found");
        }
    }

    void CreateLogicalDevice() {
        QueueFamilyIndices indices = FindQueueFamilies(physicalDevice_);
        std::set<uint32_t> uniqueFamilies = {indices.graphicsFamily.value(),
                                             indices.presentFamily.value()};

        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        float priority = 1.0f;
        for (uint32_t fam : uniqueFamilies) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = fam;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queueInfos.push_back(qi);
        }

        VkPhysicalDeviceFeatures features{};

        std::vector<const char*> requestedExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        // Required to advertise on MoltenVK if the extension is present.
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &count,
                                             nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &count,
                                             available.data());
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName,
                            VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0) {
                requestedExtensions.push_back(
                    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
                break;
            }
        }
#endif

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.pEnabledFeatures = &features;
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(requestedExtensions.size());
        createInfo.ppEnabledExtensionNames = requestedExtensions.data();

        if (kEnableValidationLayers && CheckValidationLayerSupport()) {
            createInfo.enabledLayerCount =
                static_cast<uint32_t>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }
        vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0,
                         &graphicsQueue_);
        vkGetDeviceQueue(device_, indices.presentFamily.value(), 0,
                         &presentQueue_);
    }

    // -----------------------------------------------------------------
    //  Swapchain & friends
    // -----------------------------------------------------------------
    VkSurfaceFormatKHR ChooseSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& formats) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return f;
            }
        }
        return formats.front();
    }

    VkPresentModeKHR ChoosePresentMode(
        const std::vector<VkPresentModeKHR>& modes) {
        for (const auto& m : modes) {
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& caps) {
        if (caps.currentExtent.width != UINT32_MAX) {
            return caps.currentExtent;
        }
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        VkExtent2D extent = {static_cast<uint32_t>(w),
                             static_cast<uint32_t>(h)};
        extent.width = std::clamp(extent.width, caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
        return extent;
    }

    void CreateSwapChain() {
        auto support = QuerySwapChainSupport(physicalDevice_);
        VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = ChoosePresentMode(support.presentModes);
        VkExtent2D extent = ChooseExtent(support.capabilities);

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 &&
            imageCount > support.capabilities.maxImageCount) {
            imageCount = support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface_;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = FindQueueFamilies(physicalDevice_);
        uint32_t queueIndices[] = {indices.graphicsFamily.value(),
                                   indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
        swapChainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount,
                                swapChainImages_.data());
        swapChainImageFormat_ = surfaceFormat.format;
        swapChainExtent_ = extent;
    }

    void CreateImageViews() {
        swapChainImageViews_.resize(swapChainImages_.size());
        for (size_t i = 0; i < swapChainImages_.size(); ++i) {
            swapChainImageViews_[i] = vkhelpers::CreateImageView(
                device_, swapChainImages_[i], swapChainImageFormat_,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void CreateRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAttachment;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device_, &rpInfo, nullptr, &renderPass_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }
    }

    void CreateDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(device_, &info, nullptr,
                                        &descriptorSetLayout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }
    }

    VkShaderModule CreateShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &info, nullptr, &module) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module");
        }
        return module;
    }

    void CreateGraphicsPipeline() {
        auto vertCode = ReadFile(SHADER_RT_VERT_PATH);
        auto fragCode = ReadFile(SHADER_RT_FRAG_PATH);
        VkShaderModule vertModule = CreateShaderModule(vertCode);
        VkShaderModule fragModule = CreateShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertModule;
        vertStage.pName = "main";

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragModule;
        fragStage.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {vertStage,
                                                                 fragStage};

        // No vertex buffers; the vertex shader synthesises a fullscreen
        // triangle from gl_VertexIndex.
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState blendAttach{};
        blendAttach.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttach.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &blendAttach;

        std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dyn.pDynamicStates = dynStates.data();

        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &descriptorSetLayout_;
        if (vkCreatePipelineLayout(device_, &plInfo, nullptr,
                                   &pipelineLayout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &ia;
        pipelineInfo.pViewportState = &vp;
        pipelineInfo.pRasterizationState = &rs;
        pipelineInfo.pMultisampleState = &ms;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dyn;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &graphicsPipeline_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        vkDestroyShaderModule(device_, vertModule, nullptr);
        vkDestroyShaderModule(device_, fragModule, nullptr);
    }

    void CreateFramebuffers() {
        swapChainFramebuffers_.resize(swapChainImageViews_.size());
        for (size_t i = 0; i < swapChainImageViews_.size(); ++i) {
            VkImageView attachments[] = {swapChainImageViews_[i]};
            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = renderPass_;
            info.attachmentCount = 1;
            info.pAttachments = attachments;
            info.width = swapChainExtent_.width;
            info.height = swapChainExtent_.height;
            info.layers = 1;
            if (vkCreateFramebuffer(device_, &info, nullptr,
                                    &swapChainFramebuffers_[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
    }

    void CreateCommandPool() {
        QueueFamilyIndices indices = FindQueueFamilies(physicalDevice_);
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = indices.graphicsFamily.value();
        if (vkCreateCommandPool(device_, &info, nullptr, &commandPool_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
    }

    void CreateUniformBuffers() {
        VkDeviceSize size = sizeof(RTUbo);
        uniformBuffers_.resize(kMaxFramesInFlight);
        uniformBuffersMemory_.resize(kMaxFramesInFlight);
        uniformBuffersMapped_.resize(kMaxFramesInFlight);
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            vkhelpers::CreateBuffer(
                device_, physicalDevice_, size,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers_[i], uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, size, 0,
                        &uniformBuffersMapped_[i]);
        }
    }

    void CreateDescriptorPool() {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(kMaxFramesInFlight);

        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.poolSizeCount = 1;
        info.pPoolSizes = &poolSize;
        info.maxSets = static_cast<uint32_t>(kMaxFramesInFlight);
        if (vkCreateDescriptorPool(device_, &info, nullptr, &descriptorPool_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }
    }

    void CreateDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight,
                                                   descriptorSetLayout_);
        VkDescriptorSetAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = descriptorPool_;
        info.descriptorSetCount = static_cast<uint32_t>(kMaxFramesInFlight);
        info.pSetLayouts = layouts.data();

        descriptorSets_.resize(kMaxFramesInFlight);
        if (vkAllocateDescriptorSets(device_, &info, descriptorSets_.data()) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor sets");
        }
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers_[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(RTUbo);

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSets_[i];
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufferInfo;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
    }

    void CreateCommandBuffers() {
        commandBuffers_.resize(kMaxFramesInFlight);
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commandPool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        if (vkAllocateCommandBuffers(device_, &info, commandBuffers_.data()) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers");
        }
    }

    void CreateSyncObjects() {
        // imageAvailable + inFlight fence: indexed per frame-in-flight.
        imageAvailableSemaphores_.resize(kMaxFramesInFlight);
        inFlightFences_.resize(kMaxFramesInFlight);
        // renderFinished: indexed per swapchain image to avoid reuse-while-
        // pending warnings on present.
        renderFinishedSemaphores_.resize(swapChainImages_.size());

        VkSemaphoreCreateInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fInfo{};
        fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            if (vkCreateSemaphore(device_, &sInfo, nullptr,
                                  &imageAvailableSemaphores_[i]) !=
                    VK_SUCCESS ||
                vkCreateFence(device_, &fInfo, nullptr,
                              &inFlightFences_[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create sync objects");
            }
        }
        for (size_t i = 0; i < renderFinishedSemaphores_.size(); ++i) {
            if (vkCreateSemaphore(device_, &sInfo, nullptr,
                                  &renderFinishedSemaphores_[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to create sync objects");
            }
        }
    }

    // -----------------------------------------------------------------
    //  Per-frame: record + submit + present
    // -----------------------------------------------------------------
    void RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer");
        }

        VkClearValue clear{};
        clear.color = {{0.02f, 0.02f, 0.03f, 1.0f}};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = swapChainFramebuffers_[imageIndex];
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = swapChainExtent_;
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphicsPipeline_);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent_.width);
        viewport.height = static_cast<float>(swapChainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent_;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout_, 0, 1,
                                &descriptorSets_[currentFrame_], 0, nullptr);

        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
    }

    void UpdateUniformBuffer(uint32_t frameIndex) {
        auto now = std::chrono::steady_clock::now();
        float t = std::chrono::duration<float, std::chrono::seconds::period>(
                      now - startTime_)
                      .count();

        RTUbo ubo{};
        ubo.time = t;
        ubo.aspect = swapChainExtent_.height > 0
                         ? static_cast<float>(swapChainExtent_.width) /
                               static_cast<float>(swapChainExtent_.height)
                         : 1.0f;
        ubo.cameraYaw = t * kCameraOrbitSpeed;
        ubo._pad = 0.0f;
        std::memcpy(uniformBuffersMapped_[frameIndex], &ubo, sizeof(ubo));
    }

    void DrawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                        UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult acquire = vkAcquireNextImageKHR(
            device_, swapChain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE,
            &imageIndex);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
            RecreateSwapChain();
            return;
        } else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }

        UpdateUniformBuffer(currentFrame_);

        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        RecordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSems[] = {imageAvailableSemaphores_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = waitSems;
        submit.pWaitDstStageMask = waitStages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers_[currentFrame_];
        // Signal a per-image semaphore so present can safely wait on it
        // without colliding with reuse on subsequent frames.
        VkSemaphore signalSems[] = {renderFinishedSemaphores_[imageIndex]};
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = signalSems;

        if (vkQueueSubmit(graphicsQueue_, 1, &submit,
                          inFlightFences_[currentFrame_]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer");
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = signalSems;
        VkSwapchainKHR swapChains[] = {swapChain_};
        present.swapchainCount = 1;
        present.pSwapchains = swapChains;
        present.pImageIndices = &imageIndex;

        VkResult presentRes = vkQueuePresentKHR(presentQueue_, &present);
        if (presentRes == VK_ERROR_OUT_OF_DATE_KHR ||
            presentRes == VK_SUBOPTIMAL_KHR || framebufferResized_) {
            framebufferResized_ = false;
            RecreateSwapChain();
        } else if (presentRes != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image");
        }

        currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
    }

    // -----------------------------------------------------------------
    //  Resize handling
    // -----------------------------------------------------------------
    void CleanupSwapChain() {
        for (auto fb : swapChainFramebuffers_) {
            vkDestroyFramebuffer(device_, fb, nullptr);
        }
        swapChainFramebuffers_.clear();
        for (auto view : swapChainImageViews_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        swapChainImageViews_.clear();
        if (swapChain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapChain_, nullptr);
            swapChain_ = VK_NULL_HANDLE;
        }
    }

    void RecreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);

        CleanupSwapChain();
        CreateSwapChain();
        CreateImageViews();
        CreateFramebuffers();

        // The swapchain image count may have changed: rebuild the
        // per-image renderFinished semaphores to match.
        for (size_t i = 0; i < renderFinishedSemaphores_.size(); ++i) {
            vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
        }
        renderFinishedSemaphores_.assign(swapChainImages_.size(),
                                         VK_NULL_HANDLE);
        VkSemaphoreCreateInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (size_t i = 0; i < renderFinishedSemaphores_.size(); ++i) {
            if (vkCreateSemaphore(device_, &sInfo, nullptr,
                                  &renderFinishedSemaphores_[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error(
                    "Failed to recreate render-finished semaphores");
            }
        }
    }
};

}  // namespace

int main() {
    try {
        RayTracerApp app;
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
