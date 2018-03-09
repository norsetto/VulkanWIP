#include "config.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <cstring>
#include <limits>
#include <array>
#include <chrono>
#include <string>
#include <iomanip>
#include <sstream>
#include <algorithm>

const int WIDTH = 800;
const int HEIGHT = 600;
std::string WINDOW_TITLE = "Vulkan window";

VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physical_device = VK_NULL_HANDLE;
VkPhysicalDeviceMemoryProperties memProperties;
std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
int graphicsFamily = -1;
VkDevice device = VK_NULL_HANDLE;
VkSurfaceKHR surface;
VkFormat swapchain_image_format = VK_FORMAT_UNDEFINED;
VkColorSpaceKHR imageColorSpace;
VkSwapchainKHR swapChain = VK_NULL_HANDLE;
VkExtent2D swapChainExtent = { WIDTH, HEIGHT };
std::vector<VkImageView> swapChainImageView;
std::vector<VkImage> swapChainImages;
VkRenderPass renderPass;
VkCommandPool commandPool;
std::vector<VkCommandBuffer> commandBuffers;
std::vector<VkFramebuffer> swapChainFramebuffers;
std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout;
VkBuffer vertexBuffer, indexBuffer, uniformBuffer, stagingUniformBuffer;
VkDeviceMemory vertexBufferMemory, indexBufferMemory, uniformBufferMemory, stagingUniformBufferMemory;
VkQueue graphicsQueue;
VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
VkDescriptorSet descriptorSet;

#ifdef VK_DEBUG
VkDebugReportCallbackEXT callback;
#endif

//Hardcoded vertices data
struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

const std::vector<Vertex> vertices = {
  { { -0.5f,  0.5f, -0.5f },{ 0.0f, 0.0f, 0.0f } },
  { {  0.5f,  0.5f, -0.5f },{ 0.0f, 0.0f, 1.0f } },
  { {  0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f, 0.0f } },
  { { -0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f, 1.0f } },
  { { -0.5f,  0.5f,  0.5f },{ 1.0f, 0.0f, 1.0f } },
  { {  0.5f,  0.5f,  0.5f },{ 1.0f, 0.0f, 0.0f } },
  { {  0.5f, -0.5f,  0.5f },{ 1.0f, 1.0f, 1.0f } },
  { { -0.5f, -0.5f,  0.5f },{ 1.0f, 1.0f, 0.0f } }
};

/*
   4+_____5+
    /      /|
   /      / |
 0+____1+/ 6+
  |     |  /
  |     | /
  +_____+/
  3     2
*/

const std::vector<uint16_t> indices = {
  0, 1, 2, 0, 2, 3, //front face
  3, 6, 7, 3, 2, 6, //bottom face
  0, 7, 4, 7, 0, 3, //left face
  5, 1, 0, 0, 4, 5, //top face
  6, 2, 5, 2, 1, 5, //right face
  7, 6, 4, 4, 6, 5  //back face
};

struct UniformBufferObject {
  glm::mat4 mvp;
};

std::vector<UniformBufferObject> uniforms;

#include <fstream>

static std::vector<char> readFile(const std::string& filename) {

  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

void createShaderModule(const std::vector<char>& code, VkShaderModule &shaderModule) {

  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = (uint32_t*)code.data();

  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
}

#ifdef VK_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL reportCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType,
						     uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {

  std::cerr << layerPrefix << " : " << msg << std::endl;

  return VK_FALSE;
}
#endif

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

static void copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkBufferCopy copyRegion = {};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create vertex buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate vertex buffer memory!");
  }
  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

static void set_instance() {
  uint32_t extensionCount = 0;

  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  std::cout << extensionCount << " extensions supported" << std::endl;

  std::vector<VkExtensionProperties> extensions(extensionCount);

  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

  std::cout << "available extensions:" << std::endl;

  for (const auto& extension : extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }

  VkApplicationInfo appInfo = {};

  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan test";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo InstanceCreateInfo = {};

  InstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  InstanceCreateInfo.pApplicationInfo = &appInfo;

  unsigned int glfwExtensionCount = 0;
  const char** glfwExtensions;

  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> enabledInstanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef VK_DEBUG
  enabledInstanceExtensions.push_back("VK_EXT_debug_report");
#endif

  InstanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
  InstanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();

#ifdef VK_DEBUG
  std::vector<const char *> enabledInstanceLayers;
  std::vector<const char*> unsupportedInstanceLayers;
  enabledInstanceLayers.push_back("VK_LAYER_LUNARG_standard_validation");
  enabledInstanceLayers.push_back("VK_LAYER_LUNARG_api_dump");
  
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::cout << layerCount << " layers supported" << std::endl;
  
  std::vector<VkLayerProperties> layers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

  std::cout << "available layers:" << std::endl;

  for (const auto& layer : layers) {
    std::cout << "\t" << layer.layerName << std::endl;
  }

  for (uint32_t i = 0; i < enabledInstanceLayers.size(); i++) {
    VkBool32 isSupported = false;
    for (uint32_t j = 0; j < layerCount; j++) {
      if (!strcmp(enabledInstanceLayers[i], layers[j].layerName)) {
	isSupported = true;
	break;
      }
    }
    if (!isSupported) {
      std::cout << "No layer support found for " << enabledInstanceLayers[i] << std::endl;
      unsupportedInstanceLayers.push_back(enabledInstanceLayers[i]);
    }
  }
  for (auto unsupportedLayer : unsupportedInstanceLayers) {
    auto it = std::find(enabledInstanceLayers.begin(), enabledInstanceLayers.end(), unsupportedLayer);
    if (it != enabledInstanceLayers.end()) enabledInstanceLayers.erase(it);
  }
  InstanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledInstanceLayers.size());
  InstanceCreateInfo.ppEnabledLayerNames = enabledInstanceLayers.data();
#else
  InstanceCreateInfo.enabledLayerCount = 0;
#endif

  std::cout << "required extensions:" << std::endl;

  for (const auto& required_extension : enabledInstanceExtensions) {
    std::cout << "\t" << required_extension << std::endl;
  }

  VkResult res;
  res = vkCreateInstance(&InstanceCreateInfo, nullptr, &instance);

  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    throw std::runtime_error("cannot find a compatible Vulkan ICD!");
  }
  else if (res) {
    throw std::runtime_error("failed to create instance!");
  }
}

static void set_physical_device() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  std::cout << "physical devices:" << std::endl;

  //We select the (first detected) discrete GPU with the greatest local heap memory
  const char * deviceType[5] = { "OTHER", "INTEGRATED GPU", "DISCRETE GPU", "VIRTUAL GPU", "CPU" };
  int selectedDevice = -1;
  int deviceId = 0;
  VkDeviceSize deviceMemory = 0;
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties device_property;
    vkGetPhysicalDeviceProperties(device, &device_property);
    std::cout << "Device : " << deviceId << std::endl;
    std::cout << "\tName           : " << device_property.deviceName << std::endl;
    std::cout << "\tDriver version : " << device_property.driverVersion << std::endl;
    std::cout << "\tType           : " << deviceType[device_property.deviceType] << std::endl;

    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);
    VkDeviceSize memoryHeap = 0;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
      if ((memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && (memProperties.memoryHeaps[i].size > memoryHeap)) {
	memoryHeap = memProperties.memoryHeaps[i].size;
      }
    }
    std::cout << "\tMemory         : " << memoryHeap << " bytes" << std::endl;
    if ((device_property.deviceType == 2) && (memoryHeap > deviceMemory)) {
      selectedDevice = deviceId;
      deviceMemory = memoryHeap;
    }
    deviceId++;
  }

  if (selectedDevice < 0) {
    throw std::runtime_error("no suitable GPU found!");
  }

  physical_device = devices[selectedDevice];
  std::cout << "Selected device " << selectedDevice << std::endl;

  //Do I really want to cache this!?
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);
}

static void check_graphics_queue() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, nullptr);

  if (queueFamilyCount > 0) {
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, queueFamilies.data());

    if (queueFamilyCount > 0) {
      int i = 0;
      for (const auto& queueFamily : queueFamilies) {
	if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
	  graphicsFamily = i;
	  break;
	}
	i++;
      }
    }
  }

  if (graphicsFamily < 0) {
    throw std::runtime_error("selected device doesn't support graphics!");
  }

  //We also check that it supports all required extensions
  uint32_t extensionCount = 0;

  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensionCount, availableExtensions.data());

  for (const auto& requiredExtension : deviceExtensions) {
    bool extensionFound = false;

    for (const auto& extensionProperties : availableExtensions) {
      if (strcmp(requiredExtension, extensionProperties.extensionName) == 0) {
	extensionFound = true;
	break;
      }
    }

    if (!extensionFound) {
      std::string errorMessage = std::string("selected device doesn't support ") + requiredExtension;
      throw std::runtime_error(errorMessage);
    }

  }
}

static void set_logical_device() {
  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = graphicsFamily;
  queueCreateInfo.queueCount = 1;
  float QueuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &QueuePriority;

  VkPhysicalDeviceFeatures deviceFeatures = {};

  VkDeviceCreateInfo DeviceCreateInfo = {};
  DeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DeviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  DeviceCreateInfo.queueCreateInfoCount = 1;
  DeviceCreateInfo.pEnabledFeatures = &deviceFeatures;
  DeviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
  DeviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
  DeviceCreateInfo.enabledLayerCount = 0;

  if (vkCreateDevice(physical_device, &DeviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
}

static void create_swapchain() {
  VkSwapchainKHR oldSwapchain = swapChain;

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);

  uint32_t presentModeCount;
  std::vector<VkPresentModeKHR> presentModes;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentModeCount, presentModes.data());
  }

  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto& availablePresentMode : presentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      presentMode = availablePresentMode;
      break;
    }
  }

  VkSwapchainCreateInfoKHR SwapCreateInfo = {};
  SwapCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  SwapCreateInfo.surface = surface;
  SwapCreateInfo.minImageCount = capabilities.minImageCount + 1;
  SwapCreateInfo.imageFormat = swapchain_image_format;
  SwapCreateInfo.imageColorSpace = imageColorSpace;
  SwapCreateInfo.imageExtent = swapChainExtent;
  SwapCreateInfo.imageArrayLayers = 1;
  SwapCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  SwapCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  SwapCreateInfo.queueFamilyIndexCount = 0;
  SwapCreateInfo.pQueueFamilyIndices = nullptr;
  SwapCreateInfo.preTransform = capabilities.currentTransform;
  SwapCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  SwapCreateInfo.presentMode = presentMode;
  SwapCreateInfo.clipped = VK_TRUE;
  SwapCreateInfo.oldSwapchain = swapChain;

  if (vkCreateSwapchainKHR(device, &SwapCreateInfo, nullptr, &swapChain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  if (oldSwapchain != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (const auto& imageView : swapChainImageView) {
      vkDestroyImageView(device, imageView, nullptr);
    }
    for (const auto& framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
  }

  //Retrieve the swapchain image handles
  uint32_t imageCount;

  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
  swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
}

static void create_image_views() {
  swapChainImageView.resize(swapChainImages.size());

  for (uint32_t i = 0; i < swapChainImages.size(); i++) {
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = swapChainImages[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = swapchain_image_format;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageView[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image views!");
    }
  }
}

static void create_render_pass() {
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swapchain_image_format;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

static void create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding uboLayoutBinding = {};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  uboLayoutBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uboLayoutBinding;

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

static void create_graphics_pipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStages) {

  VkVertexInputBindingDescription bindingDescription = {};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);

  //Set state for the fixed functionality pipeline stages
  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapChainExtent.width;
  viewport.height = (float)swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = { 0, 0 };
  scissor.extent = swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;
  /*
    VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
  */
  VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout };

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = setLayouts;

  VkPushConstantRange pushConstantRange = {};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = 4;

  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
  pipelineInfo.pStages = shaderStages.data();
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = nullptr;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.basePipelineIndex = -1;

  if (graphicsPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
  }

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
}

static void create_framebuffers() {
  swapChainFramebuffers.resize(swapChainImageView.size());

  for (size_t i = 0; i < swapChainImageView.size(); i++) {
    VkImageView attachments[] = {
      swapChainImageView[i]
    };

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swapChainExtent.width;
    framebufferInfo.height = swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

template<typename T, typename A>
static void create_data_buffer(std::vector<T, A> const& bufferData, VkBuffer &buffer, VkDeviceMemory &bufferMemory, VkBufferUsageFlagBits usage) {
  VkDeviceSize bufferSize = sizeof(bufferData[0]) * bufferData.size();

  //Create the staging buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  //Fill the staging buffer
  void* data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, bufferData.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  //Create and fill the buffer
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

  copy_buffer(stagingBuffer, buffer, bufferSize);

  vkFreeMemory(device, stagingBufferMemory, nullptr);
  vkDestroyBuffer(device, stagingBuffer, nullptr);
}

template<typename T, typename A>
static void create_data_double_buffer(std::vector<T, A> const& bufferData, VkBuffer &buffer1, VkBuffer &buffer2, VkDeviceMemory &bufferMemory1, VkDeviceMemory &bufferMemory2, VkBufferUsageFlagBits usage) {
  VkDeviceSize bufferSize = sizeof(bufferData[0]) * bufferData.size();

  //Create the staging buffer
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer1, bufferMemory1);

  //Fill the staging buffer
  void* data;
  vkMapMemory(device, bufferMemory1, 0, bufferSize, 0, &data);
  memcpy(data, bufferData.data(), (size_t)bufferSize);
  vkUnmapMemory(device, bufferMemory1);

  //Create and fill the buffer
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer2, bufferMemory2);

  copy_buffer(buffer1, buffer2, bufferSize);
}

static void create_descriptor_pool() {
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

static void create_descriptor_set() {
  VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = layouts;

  if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor set!");
  }

  VkDescriptorBufferInfo bufferInfo = {};
  bufferInfo.buffer = uniformBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = sizeof(UniformBufferObject);

  VkWriteDescriptorSet descriptorWrite = {};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = descriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

static void create_command_pool() {
  commandBuffers.resize(swapChainFramebuffers.size());

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = graphicsFamily;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

static void update_command_buffer(float time) {
  VkRenderPassBeginInfo renderPassBeginInfo = {};
  renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBeginInfo.renderPass = renderPass;
  renderPassBeginInfo.renderArea.offset = { 0, 0 };
  renderPassBeginInfo.renderArea.extent = swapChainExtent;
  VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
  renderPassBeginInfo.clearValueCount = 1;
  renderPassBeginInfo.pClearValues = &clearColor;

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  for (size_t i = 0; i < commandBuffers.size(); i++) {

    //Set target frame buffer
    renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];

    //Start the render pass
    vkBeginCommandBuffer(commandBuffers[i], &beginInfo);

    //Render pass
    vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdPushConstants(commandBuffers[i], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &time);
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(commandBuffers[i]);

    //End render pass
    if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }
}

static void create_command_buffers() {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  update_command_buffer(0.0f);
}

static void onWindowResize(GLFWwindow* window, int width, int height) {
  if (width == 0 || height == 0) return;

  swapChainExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
  vkDeviceWaitIdle(device);

  create_swapchain();
  create_image_views();
  create_graphics_pipeline(shaderStages);
  create_framebuffers();
}

int main(int argc, char ** argv) {

  std::cout << argv[0] << " Version " VERSION " build " BUILD_TYPE << std::endl;

  try {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_TITLE.c_str(), nullptr, nullptr);
    glfwSetWindowSizeCallback(window, onWindowResize);

    //Set instance
    set_instance();

#ifdef VK_DEBUG
    PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>
      (vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
    PFN_vkDebugReportMessageEXT vkDebugReportMessageEXT =
      reinterpret_cast<PFN_vkDebugReportMessageEXT>
      (vkGetInstanceProcAddr(instance, "vkDebugReportMessageEXT"));
    PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>
      (vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));

    VkDebugReportCallbackCreateInfoEXT CallbackCreateInfo = {};
    CallbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    CallbackCreateInfo.pNext = nullptr;
    CallbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;;
    CallbackCreateInfo.pfnCallback = &reportCallback;
    CallbackCreateInfo.pUserData = nullptr;

    if (vkCreateDebugReportCallbackEXT(instance, &CallbackCreateInfo, nullptr, &callback) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug callback!");
    }
#endif

    //Set physical device
    set_physical_device();

    //Lets check that the device supports a graphics queue
    check_graphics_queue();

    //Create the logical device
    set_logical_device();

    //Retrieve queue handle
    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

    //Set surface and check that queue supports presentation
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, graphicsFamily, surface, &presentSupport);

    if (!presentSupport) {
      throw std::runtime_error("selected queue doesn't support presentation!");
    }

    //Check that the required format is supported
    uint32_t formatCount;
    std::vector<VkSurfaceFormatKHR> formats;

    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
      formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, formats.data());
    }

    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
      swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
      imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    else {
      for (const auto& availableFormat : formats) {
	if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
	  swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	  imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	  break;
	}
      }
    }

    if (swapchain_image_format == VK_FORMAT_UNDEFINED) {
      swapchain_image_format = formats[0].format;
      imageColorSpace = formats[0].colorSpace;
    }

    //Create swapchain
    create_swapchain();

    //Create the image views
    create_image_views();

    //Read shaders
    auto vertShaderCode = readFile(SHADERS_LOCATION "basic/shader.vert.spv");
    auto fragShaderCode = readFile(SHADERS_LOCATION "basic/shader.frag.spv");

    //Create shader modules
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
    createShaderModule(vertShaderCode, vertShaderModule);
    createShaderModule(fragShaderCode, fragShaderModule);

    //Create shader stage
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    shaderStages.push_back(vertShaderStageInfo);
    shaderStages.push_back(fragShaderStageInfo);

    //Set the render passes
    create_render_pass();

    //Set the descriptor set layout
    create_descriptor_set_layout();

    //Set the graphics pipeline
    create_graphics_pipeline(shaderStages);

    //Setup framebuffer
    create_framebuffers();

    //Create the command pool
    create_command_pool();

    //Create the vertex buffer
    create_data_buffer(vertices, vertexBuffer, vertexBufferMemory, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    //Create the index buffer
    create_data_buffer(indices, indexBuffer, indexBufferMemory, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    //Create the uniform buffer
    uniforms.resize(1);
    create_data_double_buffer(uniforms, stagingUniformBuffer, uniformBuffer, stagingUniformBufferMemory, uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    //Create the descriptor pool and select a set
    create_descriptor_pool();
    create_descriptor_set();

    //Setup the command buffers
    create_command_buffers();

    //Setup semaphores
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS
	||
	vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS)
      {
	throw std::runtime_error("failed to create semaphores!");
      }

    uint32_t frame = 0;
    float total_time = 0.0f;
#define TIME_INTERVAL 1.0f

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      static auto startTime = std::chrono::high_resolution_clock::now();
      auto startFrameTime = std::chrono::high_resolution_clock::now();

      float time = std::chrono::duration_cast<std::chrono::microseconds>(startFrameTime - startTime).count() / 1e6f;

      //Update uniform buffer
      UniformBufferObject ubo = {};
      float x_rot_axis = abs(sin(time * glm::radians(60.0f)));
      float y_rot_axis = abs(cos(time * glm::radians(40.0f)));
      float z_rot_axis = 1.0f - x_rot_axis * x_rot_axis - y_rot_axis * y_rot_axis;
      if (z_rot_axis < 0.0f) {
	z_rot_axis = -sqrt(-z_rot_axis);
      }
      else {
	z_rot_axis = sqrt(z_rot_axis);
      }
      glm::mat4 model = glm::rotate(glm::mat4(), time * glm::radians(90.0f), glm::vec3(x_rot_axis, y_rot_axis, z_rot_axis));
      glm::mat4 view = glm::lookAt(glm::vec3(2.0f, 2.0f, -2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
      glm::mat4 proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
      ubo.mvp = proj * view * model;

      void* data;
      vkMapMemory(device, stagingUniformBufferMemory, 0, sizeof(ubo), 0, &data);
      memcpy(data, &ubo, sizeof(ubo));
      VkMappedMemoryRange memory_range = {};

      memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;

      memory_range.memory = stagingUniformBufferMemory;

      memory_range.size = VK_WHOLE_SIZE;

      vkFlushMappedMemoryRanges(device, 1, &memory_range);
      vkUnmapMemory(device, stagingUniformBufferMemory);

      copy_buffer(stagingUniformBuffer, uniformBuffer, sizeof(ubo));

      //Update command buffer
      update_command_buffer(time);

      //Draw frame
      uint32_t imageIndex;
      vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

      VkSubmitInfo submitInfo = {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

      VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
      VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores = waitSemaphores;
      submitInfo.pWaitDstStageMask = waitStages;

      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

      VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores = signalSemaphores;

      if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
	throw std::runtime_error("failed to submit draw command buffer!");
      }

      VkPresentInfoKHR presentInfo = {};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores = signalSemaphores;

      VkSwapchainKHR swapChains[] = { swapChain };
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = swapChains;
      presentInfo.pImageIndices = &imageIndex;

      vkQueuePresentKHR(graphicsQueue, &presentInfo);

      auto endFrameTime = std::chrono::high_resolution_clock::now();
      float frameTime = std::chrono::duration_cast<std::chrono::microseconds>(endFrameTime - startFrameTime).count() / 1e6f;
      total_time += frameTime;
      frame++;
      if (total_time >= TIME_INTERVAL) {

	std::stringstream stream;
	stream  << WINDOW_TITLE << " - " << std::fixed << std::setprecision(0) << (float)frame / total_time << " fps";
	glfwSetWindowTitle(window, stream.str().c_str());
	frame = 0;
	total_time = 0.0f;
      }
    }

    vkDeviceWaitIdle(device);

    //Cleanup
    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    vkFreeMemory(device, stagingUniformBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkDestroyBuffer(device, stagingUniformBuffer, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (const auto& imageView : swapChainImageView) {
      vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    for (const auto& framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
#ifdef VK_DEBUG
    vkDestroyDebugReportCallbackEXT(instance, callback, nullptr);
#endif
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
