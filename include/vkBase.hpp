#pragma once

#include "vkTools.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>

class VkBase
{
public:

  enum VideoBuffer: uint32_t {
    SINGLE_BUFFER = 1,
    DOUBLE_BUFFER = 2,
    TRIPLE_BUFFER = 3
  };

  struct Texture {
    VkImageView view;
    VkSampler sampler;
    VkImage image;
    VkDeviceMemory imageMemory;
    uint32_t binding;
    
    Texture(uint32_t binding = 0) : binding(binding) {
        view = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        imageMemory = VK_NULL_HANDLE;
    };
  };

  struct Buffer {
    VkBuffer buffer;
    VkDeviceSize size;
    uint32_t binding;
        
    Buffer(VkBuffer buffer = VK_NULL_HANDLE, VkDeviceSize size = 0, uint32_t binding = 0) :
        buffer(buffer), size(size), binding(binding) {};
  };

  VkBase() {};
  virtual ~VkBase();

  //Instance and device methods
#ifdef VK_DEBUG
  void createInstance(std::vector<const char *> &requiredLayers, const std::vector<const char *> &requiredExtensions = {});
#else
  void createInstance(const std::vector<const char *> &requiredExtensions = {});
#endif  
  void selectPhysicalDevice(int deviceId = -1);
  void checkDeviceExtensions(const std::vector<const char *> &extensions);
  void checkDeviceExtensions(void);
  void selectGraphicsQueue(void);
  void selectComputeQueue(void);
  void setLogicalDevice(VkPhysicalDeviceFeatures deviceFeatures = {});
  VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);


  //Image presentation methods
  void createSurface(VkSurfaceKHR surface);
  void createSwapchain(VkExtent2D swapChainExtent, VideoBuffer videoBuffer, VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR);

  //Renderpass and framebuffer methods
  void createRenderPass(std::vector<VkAttachmentDescription> const & attachments_descriptions, std::vector<VkSubpassDescription> const & subpass_descriptions, std::vector<VkSubpassDependency> const & subpass_dependencies = {});
  void createRenderPass(std::vector<VkAttachmentDescription> const & attachments_descriptions,	      std::vector<VkSubpassDescription> const & subpass_descriptions, std::vector<VkSubpassDependency> const & subpass_dependencies, VkRenderPass renderPass);

  void createFramebuffer(std::vector<VkImageView> const & attachments, VkExtent2D size, uint32_t layers, VkFramebuffer & frame_buffer);
  virtual void createFrameBuffers(void);

  //Graphics and Compute pipelines
  void createShaderModule(const std::vector<char>& code, VkShaderModule &shaderModule);
  void createShaderStage(const std::string& filename, const VkShaderStageFlagBits stage, VkPipelineShaderStageCreateInfo &shaderStageInfo);

  //Resource and memory methods
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer &buffer);
  void allocateAndBindMemoryObjectToBuffer(VkBuffer buffer, VkMemoryPropertyFlagBits memory_properties, VkDeviceMemory & memory_object);
  void createAllocateAndBindBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
  void createSampler(VkFilter mag_filter, VkFilter min_filter, VkSamplerMipmapMode mipmap_mode, VkSamplerAddressMode u_address_mode, VkSamplerAddressMode v_address_mode, VkSamplerAddressMode w_address_mode, float lod_bias, bool anisotropy_enable, float max_anisotropy, bool compare_enable, VkCompareOp compare_operator, float min_lod, float max_lod, VkBorderColor border_color, bool unnormalized_coords, VkSampler & sampler);
  void createSampledImage(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkImageUsageFlags usage, bool cubemap, VkImageViewType view_type, VkImageAspectFlags aspect, bool linear_filtering, VkImage & sampled_image, VkDeviceMemory & memory_object, VkImageView & sampled_image_view);

  void createCombinedImageSampler(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkImageUsageFlags usage, bool cubemap, VkImageViewType view_type, VkImageAspectFlags aspect, VkFilter mag_filter, VkFilter min_filter, VkSamplerMipmapMode mipmap_mode, VkSamplerAddressMode u_address_mode, VkSamplerAddressMode v_address_mode, VkSamplerAddressMode w_address_mode, float lod_bias, bool anisotropy_enable, float max_anisotropy, bool compare_enable, VkCompareOp compare_operator, float min_lod, float max_lod, VkBorderColor border_color, bool unnormalized_coords, VkSampler & sampler, VkImage & sampled_image, VkDeviceMemory & memory_object, VkImageView & sampled_image_view);

  void createImage(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkSampleCountFlagBits samples, VkImageUsageFlags usage_scenarios, bool cubemap, VkImage & image);
  void allocateAndBindMemoryObjectToImage(VkImage image, VkMemoryPropertyFlagBits memory_properties, VkDeviceMemory & memory_object);
  void createImageView(VkImage image, VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect, VkImageView & image_view);
  void create2DImageAndView(VkFormat format, VkExtent2D size, uint32_t num_mipmaps, uint32_t num_layers, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage & image, VkDeviceMemory & memory_object, VkImageView & image_view);
  void loadTexture(Texture& texture, std::string filename);
  void copyDataToBuffer(const VkDeviceMemory memory, void * data, const size_t data_size);
  void copyBufferToBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
  void copyBufferToBuffer(VkCommandPool command_pool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
  void copyBufferToImage(VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height);
  template<typename T, typename A>
  void createDataBuffer(std::vector<T, A> const& bufferData, VkBuffer &buffer, VkDeviceMemory &bufferMemory, VkBufferUsageFlagBits usage);
  template<typename T, typename A>
  void createDataDoubleBuffer(std::vector<T, A> const& bufferData, VkBuffer &buffer1, VkBuffer &buffer2, VkDeviceMemory &bufferMemory1, VkDeviceMemory &bufferMemory2, VkBufferUsageFlagBits usage);

  //Descriptor set methods
  void createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings);
  void createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings, VkDescriptorSetLayout & descriptor_set_layout);
  void createDescriptorPool(bool free_individual_sets, uint32_t max_sets_count, std::vector<VkDescriptorPoolSize> const & descriptor_types);
  void createDescriptorPool(bool free_individual_sets, uint32_t max_sets_count, std::vector<VkDescriptorPoolSize> const & descriptor_types, VkDescriptorPool & descriptor_pool);
  void allocateDescriptorSets(void);
  void allocateDescriptorSets(VkDescriptorSetLayout const & descriptor_set_layout, VkDescriptorSet & descriptor_set);
  void allocateDescriptorSets(std::vector<VkDescriptorSetLayout> const & descriptor_set_layouts, std::vector<VkDescriptorSet> & descriptor_sets);
  void allocateDescriptorSets(VkDescriptorPool descriptor_pool, std::vector<VkDescriptorSetLayout> const & descriptor_set_layouts, std::vector<VkDescriptorSet> & descriptor_sets);

  //Command buffer methods
  void createCommandPool(void);
  void createCommandPool(VkCommandPool & command_pool, VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  void allocateCommandBuffers(void);
  void allocateCommandBuffers(std::vector<VkCommandBuffer> & commandBuffers);
  void allocateCommandBuffers(VkCommandPool command_pool, uint32_t count, std::vector<VkCommandBuffer> & command_buffers, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  //Drawing methods
  void setupDrawSemaphores(void);
  void draw(void);

  //getters
  VkInstance instance(void)
  {
    return m_instance;
  };
  VkDevice device(void)
  {
    return m_device;
  };
  VkFormat swapchainImageFormat(void)
  {
    return m_swapchain_image_format;
  };
  
private:
  void add_extension(std::vector<const char *> &extensions, const char* extension)
  {
    if (std::none_of(extensions.begin(), extensions.end(), [&extension](const char * str) {
	  return std::strcmp(str, extension) == 0;
        } )) {
      extensions.push_back(extension);
    }
  };
  void add_extensions(std::vector<const char *> &currentExtensions, const std::vector<const char *> &requiredExtensions)
  {
    for (const auto& extension : requiredExtensions) {
      add_extension(currentExtensions, extension);
    }
  };
#ifdef VK_DEBUG
  void checkInstanceLayers(std::vector<const char *> &requiredLayers);
  void setup_debug_callback(void);
  static VKAPI_ATTR VkBool32 VKAPI_CALL reportCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char * layerPrefix, const char * msg, void * userData);
  VkDebugReportCallbackEXT m_callback = VK_NULL_HANDLE;
  PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT;
#endif

protected:
  VkInstance m_instance = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkFormat m_swapchain_image_format = VK_FORMAT_UNDEFINED;
  VkExtent2D m_swapChainExtent = {};
  VkSwapchainKHR m_swapChain = VK_NULL_HANDLE;
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
  VkRenderPass m_renderPass = VK_NULL_HANDLE;
  VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  VkCommandPool m_commandPool = VK_NULL_HANDLE;

  VkPhysicalDeviceMemoryProperties m_memProperties;
  VkColorSpaceKHR m_imageColorSpace;
  VkQueue m_graphicsQueue;
  VkQueue m_computeQueue;
  VkDescriptorSet m_descriptorSet;
  VkSemaphore m_imageAvailableSemaphore;
  VkSemaphore m_renderFinishedSemaphore;

  int m_graphicsFamily = -1;
  int m_computeFamily = -1;

  std::vector<const char*> m_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };
  std::vector<VkImageView> m_swapChainImageView = {};
  std::vector<VkImage> m_swapChainImages = {};
  std::vector<VkFramebuffer> m_swapChainFramebuffers = {};
  std::vector<VkCommandBuffer> m_commandBuffers {};
};

VkBase::~VkBase()
{
  vkDeviceWaitIdle(m_device);

  vkDestroySemaphore(m_device, m_renderFinishedSemaphore, nullptr);
  vkDestroySemaphore(m_device, m_imageAvailableSemaphore, nullptr);
  vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());

  if (m_instance != VK_NULL_HANDLE) {
    if (m_device != VK_NULL_HANDLE) {
      if (m_commandPool != VK_NULL_HANDLE) {
	vkDestroyCommandPool(m_device, m_commandPool, nullptr);
	m_commandPool = VK_NULL_HANDLE;
      }
      if (m_descriptorPool != VK_NULL_HANDLE) {
	vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
	m_descriptorPool = VK_NULL_HANDLE;
      }
      if (m_descriptorSetLayout != VK_NULL_HANDLE) {
	vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
	m_descriptorSetLayout = VK_NULL_HANDLE;
      }
      vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
      if (m_renderPass != VK_NULL_HANDLE) {
	vkDestroyRenderPass(m_device, m_renderPass, nullptr);
	m_renderPass = VK_NULL_HANDLE;
      }
      if (m_pipelineLayout != VK_NULL_HANDLE) {
	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
	m_pipelineLayout = VK_NULL_HANDLE;
      }
      for (const auto& imageView : m_swapChainImageView) {
	vkDestroyImageView(m_device, imageView, nullptr);
      }
      for (const auto& frameBuffer : m_swapChainFramebuffers) {
	vkDestroyFramebuffer(m_device, frameBuffer, nullptr);
      }
      if (m_swapChain != VK_NULL_HANDLE) {
	vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
	m_swapChain = VK_NULL_HANDLE;
      }
      if (m_surface != VK_NULL_HANDLE) {
	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	m_surface = VK_NULL_HANDLE;
      }
      vkDestroyDevice(m_device, nullptr);
      m_device = VK_NULL_HANDLE;
    }
#ifdef VK_DEBUG
    if (m_callback != VK_NULL_HANDLE) {
      vkDestroyDebugReportCallbackEXT(m_instance, m_callback, nullptr);
    }
#endif
    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

#ifdef VK_DEBUG
void VkBase::checkInstanceLayers(std::vector<const char *> &requiredInstanceLayers)
  {
    std::vector<const char*> unsupportedInstanceLayers;
  
    uint32_t instanceLayerCount = 0;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
    std::cout << instanceLayerCount << " layers supported" << std::endl;
  
    std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

    std::cout << "available layers:" << std::endl;

    for (const auto& instanceLayer : instanceLayers) {
      std::cout << "\t" << instanceLayer.layerName << std::endl;
    }

    for (uint32_t i = 0; i < requiredInstanceLayers.size(); i++) {
      VkBool32 isSupported = false;
      for (uint32_t j = 0; j < instanceLayerCount; j++) {
	if (!strcmp(requiredInstanceLayers[i], instanceLayers[j].layerName)) {
	  isSupported = true;
	  break;
	}
      }
      if (!isSupported) {
	std::cout << "No layer support found for " << requiredInstanceLayers[i] << std::endl;
	unsupportedInstanceLayers.push_back(requiredInstanceLayers[i]);
      }
    }
    for (auto unsupportedInstanceLayer : unsupportedInstanceLayers) {
      auto it = std::find(requiredInstanceLayers.begin(), requiredInstanceLayers.end(), unsupportedInstanceLayer);
      if (it != requiredInstanceLayers.end()) requiredInstanceLayers.erase(it);
    }
  }
#endif  

#ifdef VK_DEBUG
void VkBase::createInstance(std::vector<const char *> &requiredLayers, const std::vector<const char *> &requiredExtensions)
#else
void VkBase::createInstance(const std::vector<const char *> &requiredExtensions)
#endif  
{
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
  std::cout << extensionCount << " instance extensions supported" << std::endl;

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

  if (requiredExtensions.size() > 0)
    add_extensions(enabledInstanceExtensions, requiredExtensions);
  InstanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
  InstanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();

  std::cout << "required extensions:" << std::endl;
  for (const auto& requiredExtension : enabledInstanceExtensions) {
    std::cout << "\t" << requiredExtension << std::endl;
  }
#ifdef VK_DEBUG
  checkInstanceLayers(requiredLayers);
  InstanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
  InstanceCreateInfo.ppEnabledLayerNames = requiredLayers.data();
  std::cout << "required layers:" << std::endl;
  for (const auto& requiredLayer : requiredLayers) {
    std::cout << "\t" << requiredLayer << std::endl;
  }
#else
  InstanceCreateInfo.enabledLayerCount = 0;
#endif

  VkResult res;
  res = vkCreateInstance(&InstanceCreateInfo, nullptr, &m_instance);

  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    throw std::runtime_error("cannot find a compatible Vulkan ICD!");
  } else if (res) {
    throw std::runtime_error("failed to create instance!");
  }

#ifdef VK_DEBUG
  setup_debug_callback();
#endif
}

#ifdef VK_DEBUG
VKAPI_ATTR VkBool32 VKAPI_CALL VkBase::reportCallback(
						      VkDebugReportFlagsEXT flags,
						      VkDebugReportObjectTypeEXT objType,
						      uint64_t obj,
						      size_t location,
						      int32_t code,
						      const char* layerPrefix,
						      const char* msg,
						      void* userData)
{
  std::cerr << layerPrefix << "(";
  if(flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT)
    std::cerr << "INFO ";
  if(flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
    std::cerr << "WARN ";
  if(flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    std::cerr << "PERF ";
  if(flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    std::cerr << "ERR ";
  if(flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
    std::cerr << "DBG ";

  std::cerr << "): object: 0x" << std::hex << obj << std::dec;
  std::cerr << " type: " << objType;
  std::cerr << " location: " << location;
  std::cerr << " msgCode: " << code;
  std::cerr << ": " << msg << std::endl;

  return VK_FALSE;
}

void VkBase::setup_debug_callback(void)
{
  PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
    reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>
    (vkGetInstanceProcAddr(m_instance, "vkCreateDebugReportCallbackEXT"));
  vkDestroyDebugReportCallbackEXT =
    reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>
    (vkGetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT"));

  VkDebugReportCallbackCreateInfoEXT CallbackCreateInfo = {};
  CallbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
  CallbackCreateInfo.pNext = nullptr;
  CallbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
  CallbackCreateInfo.pfnCallback = &reportCallback;
  CallbackCreateInfo.pUserData = nullptr;

  if (vkCreateDebugReportCallbackEXT(m_instance, &CallbackCreateInfo, nullptr, &m_callback) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug callback!");
  }
}
#endif

void VkBase::selectPhysicalDevice(int selectedDevice)
{
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

  if (selectedDevice < 0) {
    std::cout << "physical devices:" << std::endl;

    //We select the (first detected) discrete GPU with the greatest local heap memory
    const char * deviceType[5] = { "OTHER", "INTEGRATED GPU", "DISCRETE GPU", "VIRTUAL GPU", "CPU" };
    int deviceId = 0;
    VkDeviceSize deviceMemory = 0;
    for (const auto& device : devices) {
      VkPhysicalDeviceProperties device_property;
      vkGetPhysicalDeviceProperties(device, &device_property);
      std::cout << "Device : " << deviceId << std::endl;
      std::cout << "\tName           : " << device_property.deviceName << std::endl;
      std::cout << "\tDriver version : " << device_property.driverVersion << std::endl;
      std::cout << "\tType           : " << deviceType[device_property.deviceType] << std::endl;

      vkGetPhysicalDeviceMemoryProperties(device, &m_memProperties);
      VkDeviceSize memoryHeap = 0;
      for (uint32_t i = 0; i < m_memProperties.memoryHeapCount; i++) {
	if ((m_memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && (m_memProperties.memoryHeaps[i].size > memoryHeap)) {
	  memoryHeap = m_memProperties.memoryHeaps[i].size;
	}
      }
      std::cout << "\tMemory         : " << memoryHeap << " bytes" << std::endl;
      if ((device_property.deviceType == 2) && (memoryHeap > deviceMemory)) {
	selectedDevice = deviceId;
	deviceMemory = memoryHeap;
      }
      deviceId++;
    }
  }
  if ((selectedDevice < 0) || (selectedDevice + 1 > static_cast<int>(deviceCount))) {
    throw std::runtime_error("no suitable GPU found!");
  }

  m_physical_device = devices[selectedDevice];
  std::cout << "Selected device " << selectedDevice << std::endl;

  //We cache this for later use in createBuffer
  vkGetPhysicalDeviceMemoryProperties(m_physical_device, &m_memProperties);
}

void VkBase::checkDeviceExtensions(const std::vector<const char *> &extensions)
{
  if (extensions.size() > 0) add_extensions(m_deviceExtensions, extensions);
  checkDeviceExtensions();
}

void VkBase::checkDeviceExtensions()
{
  uint32_t extensionCount = 0;

  vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &extensionCount, nullptr);

  std::cout << extensionCount << " device extensions supported" << std::endl;

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &extensionCount, availableExtensions.data());

  std::cout << "available extensions:" << std::endl;

  for (const auto& extension : availableExtensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }

  std::cout << "required extensions:" << std::endl;

  for (const auto& requiredExtension : m_deviceExtensions) {
    std::cout << "\t" << requiredExtension << std::endl;
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

void VkBase::selectGraphicsQueue()
{
  uint32_t queueFamilyCount = 0;
  uint32_t familySize = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, nullptr);

  if (queueFamilyCount > 0) {
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, queueFamilies.data());

    if (queueFamilyCount > 0) {
      int i = 0;
      for (const auto& queueFamily : queueFamilies) {
	if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
	  m_graphicsFamily = i;
	  familySize = queueFamily.queueCount;
	  break;
	}
	i++;
      }
    }
  }

  if (m_graphicsFamily < 0) {
    throw std::runtime_error("selected device doesn't support graphics!");
  }
  std::cout << "Selected graphics queue family " << m_graphicsFamily << std::endl;
  std::cout << "Count of queues in this family " << familySize << std::endl;
}

void VkBase::selectComputeQueue()
{
  uint32_t queueFamilyCount = 0;
  uint32_t familySize = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, nullptr);

  if (queueFamilyCount > 0) {
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, queueFamilies.data());

    if (queueFamilyCount > 0) {
      int i = 0;
      for (const auto& queueFamily : queueFamilies) {
	if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT && i != m_graphicsFamily) {
	  m_computeFamily = i;
	  familySize = queueFamily.queueCount;
	  break;
	}
	i++;
      }
      i = 0;
      if (m_computeFamily < 0) {
	for (const auto& queueFamily : queueFamilies) {
	  if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
	    m_computeFamily = i;
	    familySize = queueFamily.queueCount;
	    break;
	  }
	  i++;
	}
      }
    }
  }

  if (m_computeFamily < 0) {
    throw std::runtime_error("selected device doesn't support compute!");
  }
  std::cout << "Selected compute queue family " << m_computeFamily << std::endl;
  std::cout << "Count of queues in this family " << familySize << std::endl;
}

void VkBase::setLogicalDevice(VkPhysicalDeviceFeatures deviceFeatures)
{
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfo;
  VkDeviceQueueCreateInfo queueCreateInfoTmp = {};
  queueCreateInfoTmp.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfoTmp.queueFamilyIndex = m_graphicsFamily;
  queueCreateInfoTmp.queueCount = 1;
  float QueuePriority = 1.0f;
  queueCreateInfoTmp.pQueuePriorities = &QueuePriority;
  queueCreateInfo.push_back(queueCreateInfoTmp);
  uint32_t queueCount = 1;

  if (m_computeFamily != m_graphicsFamily) {
    queueCount = 2;
    queueCreateInfoTmp.queueFamilyIndex = m_computeFamily;
    queueCreateInfo.push_back(queueCreateInfoTmp);
  }

  VkDeviceCreateInfo DeviceCreateInfo = {};
  DeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DeviceCreateInfo.pQueueCreateInfos = queueCreateInfo.data();
  DeviceCreateInfo.queueCreateInfoCount = queueCount;
  DeviceCreateInfo.pEnabledFeatures = &deviceFeatures;
  DeviceCreateInfo.enabledExtensionCount = (uint32_t)m_deviceExtensions.size();
  DeviceCreateInfo.ppEnabledExtensionNames = m_deviceExtensions.data();
  DeviceCreateInfo.enabledLayerCount = 0;

  if (vkCreateDevice(m_physical_device, &DeviceCreateInfo, nullptr, &m_device) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  //Retrieve queues handle
  vkGetDeviceQueue(m_device, m_graphicsFamily, 0, &m_graphicsQueue);
  vkGetDeviceQueue(m_device,  m_computeFamily, 0, &m_computeQueue);
}

VkFormat VkBase::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(m_physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    }
    else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  throw std::runtime_error("failed to find supported format!");
}

void VkBase::createSurface(VkSurfaceKHR surface)
{
  if (surface == VK_NULL_HANDLE) throw std::runtime_error("given surface is not valid!");

  m_surface = surface;

  VkBool32 presentSupport = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(m_physical_device, m_graphicsFamily, m_surface, &presentSupport);

  if (!presentSupport) {
    throw std::runtime_error("selected queue doesn't support presentation!");
  }

  //Check that the required format is supported
  uint32_t formatCount;
  std::vector<VkSurfaceFormatKHR> formats;

  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &formatCount, nullptr);

  if (formatCount != 0) {
    formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &formatCount, formats.data());
  }

  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    m_swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
    m_imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  } else {
    for (const auto& availableFormat : formats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
	m_swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	m_imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	break;
      }
    }
  }

  if (m_swapchain_image_format == VK_FORMAT_UNDEFINED) {
    m_swapchain_image_format = formats[0].format;
    m_imageColorSpace = formats[0].colorSpace;
  }
}

void VkBase::createSwapchain(VkExtent2D swapChainExtent, VideoBuffer videoBuffer, VkPresentModeKHR mode)
{
  //Cache the swapchain handle in case we have to clear it (ie. because of window resizing)
  VkSwapchainKHR oldSwapchain = m_swapChain;

  //Query the current hardware capabilities
  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &capabilities);

  //Check that the required presentation mode is available, if not fallback to VK_PRESENT_MODE_FIFO_KHR
  uint32_t presentModeCount = 0;
  std::vector<VkPresentModeKHR> presentModes;
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &presentModeCount, presentModes.data());
  }

  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto& availablePresentMode : presentModes) {
    if (availablePresentMode == mode) {
      presentMode = availablePresentMode;
      break;
    }
  }

  //Check that the size is within the hardware/OS limits
  if (capabilities.currentExtent.width  == 0xFFFFFFFF &&
      capabilities.currentExtent.height == 0xFFFFFFFF ) {

    if (swapChainExtent.width < capabilities.minImageExtent.width) {
      swapChainExtent.width = capabilities.minImageExtent.width;
    }
    else if (swapChainExtent.width > capabilities.maxImageExtent.width) {
      swapChainExtent.width = capabilities.maxImageExtent.width;
    }
    if (swapChainExtent.height < capabilities.minImageExtent.height) {
      swapChainExtent.height = capabilities.minImageExtent.height;
    }
    else if (swapChainExtent.height > capabilities.maxImageExtent.height) {
      swapChainExtent.height = capabilities.maxImageExtent.height;
    }
  }
  else {
    swapChainExtent = capabilities.currentExtent;
  }

  VkSwapchainCreateInfoKHR SwapCreateInfo = {};
  SwapCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  SwapCreateInfo.surface = m_surface;

  //Check that it is possible to create the requested number of video buffers, or fallback to the max possible
  if (capabilities.maxImageCount > 0 && videoBuffer > capabilities.maxImageCount) {
    SwapCreateInfo.minImageCount = capabilities.maxImageCount;
  } else {
    SwapCreateInfo.minImageCount = videoBuffer;
  }
  SwapCreateInfo.imageFormat = m_swapchain_image_format;
  SwapCreateInfo.imageColorSpace = m_imageColorSpace;
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
  SwapCreateInfo.oldSwapchain = m_swapChain;

  if (vkCreateSwapchainKHR(m_device, &SwapCreateInfo, nullptr, &m_swapChain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  m_swapChainExtent = swapChainExtent;
  if (oldSwapchain != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    for (const auto& imageView : m_swapChainImageView) {
      vkDestroyImageView(m_device, imageView, nullptr);
    }
    for (const auto& framebuffer : m_swapChainFramebuffers) {
      vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }
    vkDestroySwapchainKHR(m_device, oldSwapchain, nullptr);
  }

  //Retrieve the swapchain image handles
  uint32_t imageCount = 0;

  if (vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr) != VK_SUCCESS || imageCount == 0)
    throw std::runtime_error("couldn't get the number of swapchain images!");

  m_swapChainImages.resize(imageCount);
  if (vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, m_swapChainImages.data()) != VK_SUCCESS || imageCount == 0)
    throw std::runtime_error("couldn't enumerate swapchain images!");

  m_swapChainImageView.resize(m_swapChainImages.size());

  for (uint32_t i = 0; i < m_swapChainImages.size(); i++) {
    createImageView(m_swapChainImages[i], VK_IMAGE_VIEW_TYPE_2D, m_swapchain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, m_swapChainImageView[i]);
  }
}

void VkBase::createRenderPass(std::vector<VkAttachmentDescription> const & attachments_descriptions,
			      std::vector<VkSubpassDescription> const & subpass_descriptions,
			      std::vector<VkSubpassDependency> const & subpass_dependencies)
{
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments_descriptions.size());
  renderPassInfo.pAttachments = attachments_descriptions.data();
  renderPassInfo.subpassCount = static_cast<uint32_t>(subpass_descriptions.size());
  renderPassInfo.pSubpasses = subpass_descriptions.data();
  renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass_dependencies.size());
  renderPassInfo.pDependencies = subpass_dependencies.data();

  if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void VkBase::createRenderPass(std::vector<VkAttachmentDescription> const & attachments_descriptions,
			      std::vector<VkSubpassDescription> const & subpass_descriptions,
			      std::vector<VkSubpassDependency> const & subpass_dependencies, VkRenderPass renderPass)
{
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments_descriptions.size());
  renderPassInfo.pAttachments = attachments_descriptions.data();
  renderPassInfo.subpassCount = static_cast<uint32_t>(subpass_descriptions.size());
  renderPassInfo.pSubpasses = subpass_descriptions.data();
  renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass_dependencies.size());
  renderPassInfo.pDependencies = subpass_dependencies.data();

  if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void VkBase::createSampler(VkFilter mag_filter, VkFilter min_filter, VkSamplerMipmapMode mipmap_mode, VkSamplerAddressMode u_address_mode, VkSamplerAddressMode v_address_mode, VkSamplerAddressMode w_address_mode, float lod_bias, bool anisotropy_enable, float max_anisotropy, bool compare_enable, VkCompareOp compare_operator, float min_lod, float max_lod, VkBorderColor border_color, bool unnormalized_coords, VkSampler & sampler)

{

  VkSamplerCreateInfo sampler_create_info = {};
  sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_create_info.magFilter = mag_filter;
  sampler_create_info.minFilter = min_filter;
  sampler_create_info.mipmapMode = mipmap_mode;
  sampler_create_info.addressModeU = u_address_mode;
  sampler_create_info.addressModeV = v_address_mode;
  sampler_create_info.addressModeW = w_address_mode;
  sampler_create_info.mipLodBias = lod_bias;
  sampler_create_info.anisotropyEnable = anisotropy_enable;
  sampler_create_info.maxAnisotropy = max_anisotropy;
  sampler_create_info.compareEnable = compare_enable;
  sampler_create_info.compareOp = compare_operator;
  sampler_create_info.minLod = min_lod;
  sampler_create_info.maxLod = max_lod;
  sampler_create_info.borderColor = border_color;
  sampler_create_info.unnormalizedCoordinates = unnormalized_coords;
  
  if (vkCreateSampler(m_device, &sampler_create_info, nullptr, &sampler) != VK_SUCCESS)
    throw std::runtime_error("could not create sampler!");
}

void VkBase::createSampledImage(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkImageUsageFlags usage, bool cubemap, VkImageViewType view_type, VkImageAspectFlags aspect, bool linear_filtering, VkImage & sampled_image, VkDeviceMemory & memory_object, VkImageView & sampled_image_view)
{

  VkFormatProperties format_properties;
  vkGetPhysicalDeviceFormatProperties(m_physical_device, format, &format_properties);

  if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
    throw std::runtime_error("provided format is not supported for a sampled image!");
  }

  if (linear_filtering &&
      !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    throw std::runtime_error("provided format is not supported for a linear image filtering!");
  }

  createImage(type, format, size, num_mipmaps, num_layers, VK_SAMPLE_COUNT_1_BIT, usage | VK_IMAGE_USAGE_SAMPLED_BIT, cubemap, sampled_image);
  allocateAndBindMemoryObjectToImage(sampled_image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object);
  createImageView(sampled_image, view_type, format, aspect, sampled_image_view);
}

void VkBase::createCombinedImageSampler(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkImageUsageFlags usage, bool cubemap, VkImageViewType view_type, VkImageAspectFlags aspect, VkFilter mag_filter, VkFilter min_filter, VkSamplerMipmapMode mipmap_mode, VkSamplerAddressMode u_address_mode, VkSamplerAddressMode v_address_mode, VkSamplerAddressMode w_address_mode, float lod_bias, bool anisotropy_enable, float max_anisotropy, bool compare_enable, VkCompareOp compare_operator, float min_lod, float max_lod, VkBorderColor border_color, bool unnormalized_coords, VkSampler & sampler, VkImage & sampled_image, VkDeviceMemory & memory_object, VkImageView & sampled_image_view)
{
  createSampler(mag_filter, min_filter, mipmap_mode, u_address_mode, v_address_mode, w_address_mode, lod_bias, anisotropy_enable, max_anisotropy, compare_enable, compare_operator, min_lod, max_lod, border_color, unnormalized_coords, sampler);

  bool linear_filtering = (mag_filter == VK_FILTER_LINEAR) || (min_filter == VK_FILTER_LINEAR) || (mipmap_mode == VK_SAMPLER_MIPMAP_MODE_LINEAR);

  createSampledImage(type, format, size, num_mipmaps, num_layers, usage, cubemap, view_type, aspect, linear_filtering, sampled_image, memory_object, sampled_image_view);
}

void VkBase::createImage(VkImageType type, VkFormat format, VkExtent3D size, uint32_t num_mipmaps, uint32_t num_layers, VkSampleCountFlagBits samples, VkImageUsageFlags usage_scenarios, bool cubemap, VkImage & image)
{
  VkImageCreateInfo image_create_info = {};
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.flags = cubemap ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
  image_create_info.imageType = type;
  image_create_info.format = format;
  image_create_info.extent = size;
  image_create_info.mipLevels = num_mipmaps;
  image_create_info.arrayLayers =	cubemap ? 6 * num_layers : num_layers;
  image_create_info.samples =	samples;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = usage_scenarios;
  image_create_info.sharingMode =	VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (vkCreateImage(m_device, &image_create_info, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("could not create an image!");
  }
}

void VkBase::allocateAndBindMemoryObjectToImage(VkImage image, VkMemoryPropertyFlagBits memory_properties, VkDeviceMemory & memory_object)
{
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(m_device, image, &memory_requirements);

  uint32_t type = VkTools::findMemoryType(memory_requirements.memoryTypeBits, m_memProperties, memory_properties);

  VkMemoryAllocateInfo image_memory_allocate_info = {};
  image_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  image_memory_allocate_info.allocationSize = memory_requirements.size;
  image_memory_allocate_info.memoryTypeIndex = type;

  memory_object = VK_NULL_HANDLE;
  if (vkAllocateMemory(m_device, &image_memory_allocate_info, nullptr, &memory_object) != VK_SUCCESS) {
    throw std::runtime_error("could not allocate memory for an image!");
  }

  if (vkBindImageMemory(m_device, image, memory_object, 0) != VK_SUCCESS) {
    throw std::runtime_error("could not bind memory object to an image!");
  }
}

void VkBase::allocateAndBindMemoryObjectToBuffer(VkBuffer buffer, VkMemoryPropertyFlagBits memory_properties, VkDeviceMemory & memory_object)
{
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memory_requirements);

  uint32_t type = VkTools::findMemoryType(memory_requirements.memoryTypeBits, m_memProperties, memory_properties);

  VkMemoryAllocateInfo buffer_memory_allocate_info = {};
  buffer_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  buffer_memory_allocate_info.allocationSize = memory_requirements.size;
  buffer_memory_allocate_info.memoryTypeIndex = type;

  memory_object = VK_NULL_HANDLE;
  if (vkAllocateMemory(m_device, &buffer_memory_allocate_info, nullptr, &memory_object) != VK_SUCCESS) {
    throw std::runtime_error("could not allocate memory for a buffer!");
  }

  if (vkBindBufferMemory(m_device, buffer, memory_object, 0) != VK_SUCCESS) {
    throw std::runtime_error("could not bind memory object to a buffer!");
  }
}

void VkBase::createImageView(VkImage image, VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect, VkImageView & image_view)
{
  VkImageViewCreateInfo image_view_create_info = {};
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.image = image;
  image_view_create_info.viewType = view_type;
  image_view_create_info.format = format;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask = aspect;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

  if (vkCreateImageView(m_device, &image_view_create_info, nullptr, &image_view) != VK_SUCCESS) {
    throw std::runtime_error("could not create an image view!");
  }
}

void VkBase::create2DImageAndView(VkFormat format, VkExtent2D size, uint32_t num_mipmaps, uint32_t num_layers, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage & image, VkDeviceMemory & memory_object, VkImageView & image_view)
{
  createImage(VK_IMAGE_TYPE_2D, format, { size.width, size.height, 1 }, num_mipmaps, num_layers, samples, usage, false, image);
  allocateAndBindMemoryObjectToImage(image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object);
  createImageView(image, VK_IMAGE_VIEW_TYPE_2D, format, aspect, image_view);
}

void VkBase::loadTexture(Texture &texture, std::string filename)
{
    //Load texture data
    std::vector<unsigned char> image_data;
    int image_width, image_height, image_num_components, image_data_size;
    
    VkTools::loadTextureDataFromFile(filename.c_str(), 4, image_data, &image_width, &image_height, &image_num_components, &image_data_size);
    std::cout << "loaded " << image_width << "X" << image_height << " (";
    if (image_data_size >= 1048576)
        std::cout << image_data_size / 1048576 << "MB) from ";
    else
        std::cout << image_data_size / 1024 << "KB) from ";
    std::cout << filename << std::endl;

    //Create texture 
    VkSampler _sampler;
    VkImage _image;
    VkDeviceMemory _imageMemory;
    VkImageView _imageView;
    
    createCombinedImageSampler(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, { (uint32_t)image_width, (uint32_t)image_height, 1 }, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.0f, false, 1.0f, false, VK_COMPARE_OP_ALWAYS, 0.0f, 1.0f, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, false, _sampler, _image, _imageMemory, _imageView);
    
    //Create texture data staging buffers
    VkBuffer stagingBufferImage;
    VkDeviceMemory stagingBufferMemoryImage;
    
    createAllocateAndBindBuffer(image_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferImage, stagingBufferMemoryImage);
    
    //Copy data to staging buffer
    copyDataToBuffer(stagingBufferMemoryImage, image_data.data(), static_cast<size_t>(image_data_size));
    
    //Copy staging buffer to texture
    copyBufferToImage(stagingBufferImage, _image, (uint32_t)image_width, (uint32_t)image_height);
    
    vkDestroyBuffer(m_device, stagingBufferImage, nullptr);
    vkFreeMemory(m_device, stagingBufferMemoryImage, nullptr);
    
    //Fill in return data
    texture.view = _imageView;
    texture.sampler = _sampler;
    texture.image = _image;
    texture.imageMemory = _imageMemory;
}

void VkBase::createFramebuffer(std::vector<VkImageView> const & attachments, VkExtent2D size, uint32_t layers, VkFramebuffer & frame_buffer)
{
  VkFramebufferCreateInfo framebuffer_create_info = {};
  framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_create_info.renderPass = m_renderPass;
  framebuffer_create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  framebuffer_create_info.pAttachments = attachments.data();
  framebuffer_create_info.width = size.width;
  framebuffer_create_info.height = size.height;
  framebuffer_create_info.layers = layers;

  if (vkCreateFramebuffer(m_device, &framebuffer_create_info, nullptr, & frame_buffer) != VK_SUCCESS) {
    throw std::runtime_error("could not create a framebuffer!");
  }
}

void VkBase::createFrameBuffers()
{
  m_swapChainFramebuffers.resize(m_swapChainImageView.size());

  for (size_t i = 0; i < m_swapChainImageView.size(); i++) {
    createFramebuffer({ m_swapChainImageView[i] }, m_swapChainExtent, 1, m_swapChainFramebuffers[i]);
  }
}

void VkBase::createShaderModule(const std::vector<char>& code, VkShaderModule &shaderModule)
{
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = (uint32_t*)code.data();

  if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
}

void VkBase::createShaderStage(const std::string& filename, const VkShaderStageFlagBits stage, VkPipelineShaderStageCreateInfo &shaderStageInfo)
{
  //Read shader
  auto shaderCode = VkTools::readFile(filename);

  //Create shader module
  VkShaderModule shaderModule;
  createShaderModule(shaderCode, shaderModule);

  //Create shader stage
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = stage;
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = "main";
}

void VkBase::createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding> & layoutBindings, VkDescriptorSetLayout & descriptorSetLayout)
{
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
  layoutInfo.pBindings = layoutBindings.data();

  if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void VkBase::createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding> & layoutBindings)
{
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
  layoutInfo.pBindings = layoutBindings.data();

  if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void VkBase::createCommandPool()
{
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = m_graphicsFamily;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

void VkBase::createCommandPool(VkCommandPool & command_pool, VkCommandPoolCreateFlags flags)
{
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = m_graphicsFamily;
  poolInfo.flags =flags;

  if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

void VkBase::copyDataToBuffer(const VkDeviceMemory memory, void* data, const size_t data_size)
{
  void* local_data;

  vkMapMemory(m_device, memory, 0, data_size, 0, &local_data);
  std::memcpy(local_data, data, data_size);
  VkMappedMemoryRange memory_range = {};
  memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memory_range.memory = memory;
  memory_range.size = VK_WHOLE_SIZE;

  vkFlushMappedMemoryRanges(m_device, 1, &memory_range);
  vkUnmapMemory(m_device, memory);
}

void VkBase::copyBufferToBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

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

  vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphicsQueue);

  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}



void VkBase::copyBufferToBuffer(VkCommandPool command_pool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

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

  vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphicsQueue);

  vkFreeCommandBuffers(m_device, command_pool, 1, &commandBuffer);
}

void VkBase::copyBufferToImage(VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height)
{
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkImageMemoryBarrier image_memory_barriers = {
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    nullptr,
    0,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    dstImage,
    { VK_IMAGE_ASPECT_COLOR_BIT,
      0,
      VK_REMAINING_MIP_LEVELS,
      0,
      VK_REMAINING_ARRAY_LAYERS
    }
  };

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_memory_barriers);

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;

  region.imageOffset = { 0, 0, 0 };
  region.imageExtent = { width, height, 1};

  vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  image_memory_barriers.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  image_memory_barriers.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  image_memory_barriers.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  image_memory_barriers.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_memory_barriers);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphicsQueue);

  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);

}

void VkBase::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer &buffer)
{
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }
}

void VkBase::createAllocateAndBindBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
{
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = VkTools::findMemoryType(memRequirements.memoryTypeBits, m_memProperties, properties);

  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }
  if (vkBindBufferMemory(m_device, buffer, bufferMemory, 0) != VK_SUCCESS) {
    throw std::runtime_error("could not bind memory object to a buffer!");
  }
}

void VkBase::createDescriptorPool(bool free_individual_sets, uint32_t max_sets_count, std::vector<VkDescriptorPoolSize> const & descriptor_types, VkDescriptorPool & descriptor_pool)
{
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = free_individual_sets ? VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0u;
  poolInfo.poolSizeCount = static_cast<uint32_t>(descriptor_types.size());
  poolInfo.pPoolSizes = descriptor_types.data();
  poolInfo.maxSets = max_sets_count;

  if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &descriptor_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void VkBase::createDescriptorPool(bool free_individual_sets, uint32_t max_sets_count, std::vector<VkDescriptorPoolSize> const & descriptor_types)
{
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = free_individual_sets ? VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0u;
  poolInfo.poolSizeCount = static_cast<uint32_t>(descriptor_types.size());
  poolInfo.pPoolSizes = descriptor_types.data();
  poolInfo.maxSets = max_sets_count;

  if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void VkBase::allocateDescriptorSets()
{
  VkDescriptorSetLayout layouts[] = { m_descriptorSetLayout };
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = layouts;
  

  if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {

    throw std::runtime_error("failed to allocate descriptor set!");

  }
}

void VkBase::allocateDescriptorSets(VkDescriptorSetLayout const & descriptor_set_layout, VkDescriptorSet & descriptor_set)
{
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptor_set_layout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &descriptor_set) != VK_SUCCESS)
      throw std::runtime_error("could not allocate descriptor sets!");
}

void VkBase::allocateDescriptorSets(std::vector<VkDescriptorSetLayout> const & descriptor_set_layouts, std::vector<VkDescriptorSet> & descriptor_sets)
{
  if (descriptor_set_layouts.size() > 0) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    allocInfo.pSetLayouts = descriptor_set_layouts.data();

    descriptor_sets.resize(descriptor_set_layouts.size());

    if (vkAllocateDescriptorSets(m_device, &allocInfo, descriptor_sets.data()) != VK_SUCCESS)
      throw std::runtime_error("could not allocate descriptor sets!");
  }
  else {
    throw std::runtime_error("incorrect descriptor sets!");
  }
}

void VkBase::allocateDescriptorSets(VkDescriptorPool descriptor_pool, std::vector<VkDescriptorSetLayout> const & descriptor_set_layouts, std::vector<VkDescriptorSet> & descriptor_sets)
{
  if (descriptor_set_layouts.size() > 0) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptor_pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    allocInfo.pSetLayouts = descriptor_set_layouts.data();

    descriptor_sets.resize(descriptor_set_layouts.size());

    if (vkAllocateDescriptorSets(m_device, &allocInfo, descriptor_sets.data()) != VK_SUCCESS)
      throw std::runtime_error("could not allocate descriptor sets!");
  }
  else {
    throw std::runtime_error("incorrect descriptor sets!");
  }
}

template<typename T, typename A>
void VkBase::createDataBuffer(std::vector<T, A> const& bufferData, VkBuffer &buffer, VkDeviceMemory &bufferMemory, VkBufferUsageFlagBits usage)
{
  VkDeviceSize bufferSize = sizeof(bufferData[0]) * bufferData.size();

  //Create the staging buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createAllocateAndBindBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  //Fill the staging buffer
  void* data;
  vkMapMemory(m_device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, bufferData.data(), (size_t)bufferSize);
  vkUnmapMemory(m_device, stagingBufferMemory);

  //Create and fill the buffer
  createAllocateAndBindBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

  copyBufferToBuffer(stagingBuffer, buffer, bufferSize);

  vkFreeMemory(m_device, stagingBufferMemory, nullptr);
  vkDestroyBuffer(m_device, stagingBuffer, nullptr);
}

template<typename T, typename A>
void VkBase::createDataDoubleBuffer(std::vector<T, A> const& bufferData, VkBuffer &buffer1, VkBuffer &buffer2, VkDeviceMemory &bufferMemory1, VkDeviceMemory &bufferMemory2, VkBufferUsageFlagBits usage)
{
  VkDeviceSize bufferSize = sizeof(bufferData[0]) * bufferData.size();

  //Create the staging buffer
  createAllocateAndBindBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer1, bufferMemory1);

  //Fill the staging buffer
  void* data;
  vkMapMemory(m_device, bufferMemory1, 0, bufferSize, 0, &data);
  memcpy(data, bufferData.data(), (size_t)bufferSize);
  vkUnmapMemory(m_device, bufferMemory1);

  //Create and fill the buffer
  createAllocateAndBindBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer2, bufferMemory2);

  copyBufferToBuffer(buffer1, buffer2, bufferSize);
}

void VkBase::allocateCommandBuffers()
{
  m_commandBuffers.resize(m_swapChainFramebuffers.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = m_commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

  if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void VkBase::allocateCommandBuffers(std::vector<VkCommandBuffer> & commandBuffers)
{
  commandBuffers.resize(m_swapChainFramebuffers.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = m_commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  if (vkAllocateCommandBuffers(m_device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void VkBase::allocateCommandBuffers(VkCommandPool command_pool, uint32_t count, std::vector<VkCommandBuffer> & command_buffers, VkCommandBufferLevel level)
{
  command_buffers.resize(count);

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = level;
  allocInfo.commandBufferCount = count;

  if (vkAllocateCommandBuffers(m_device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void VkBase::setupDrawSemaphores()
{
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore) != VK_SUCCESS
      ||
      vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("failed to create semaphores!");
  }
}

void VkBase::draw()
{
  uint32_t imageIndex;
  vkAcquireNextImageKHR(m_device, m_swapChain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphore };
  VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

  VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphore };
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = { m_swapChain };
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
}
