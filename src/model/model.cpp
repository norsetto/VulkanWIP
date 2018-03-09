#include "config.h"
#include "vkBase.hpp"
#include "vkTools.hpp"
#include "vkTest.hpp"
#include "camera.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#ifdef VK_DEBUG
#include <glm/gtx/string_cast.hpp>
#endif
#include <limits>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

VkTest *vkTest;
VkDevice device = VK_NULL_HANDLE;

const VkBase::VideoBuffer videoBuffer = VkBase::TRIPLE_BUFFER;
int width = 800;
int height = 600;
std::string WINDOW_TITLE = "Vulkan Test";
const VkPresentModeKHR mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {};
float frameTime;
Camera *camera;
float max_view;

VkTools::Mesh model = {};

static void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (action) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    case 'W':
      camera->move_forward(frameTime);
      break;
    case 'S':
      camera->move_backward(frameTime);
      break;
    case 'D':
      camera->move_right(frameTime);
      break;
    case 'A':
      camera->move_left(frameTime);
      break;
    case 'Q':
      camera->move_up(frameTime);
      break;
    case 'Z':
      camera->move_down(frameTime);
      break;
    }
  }
}

static void onMouseButton(GLFWwindow* window, int button, int action, int mods) {
  mouse.pressed = !mouse.pressed;
  mouse.button = button;
  glfwGetCursorPos(window, &mouse.xpos, &mouse.ypos);
}

static void onMouseScroll(GLFWwindow* window, double xoffset, double yoffset) {
  if (yoffset > 0) {
    camera->move_forward(frameTime);
  }
  else {
    camera->move_backward(frameTime);
  }
}

static void onWindowResize(GLFWwindow* window, int w, int h)
{
  if (w == 0 || h == 0) return;


  vkDeviceWaitIdle(device);

  vkTest->createSwapchain( {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}, videoBuffer, mode);
  vkTest->createGraphicsPipeline(shaderStages);
  vkTest->createFrameBuffers();
  vkTest->recordCommandBuffers(model);
  width = w;
  height = h;
  camera->set_proj_matrix(width / (float)height, 0.01f, max_view);
}

int main(int argc, char ** argv)
{
  std::cout << argv[0] << " Version " VERSION " build " BUILD_TYPE << std::endl;
  try {
    //Load assets
    VkTools::load3DModelFromObjFile(MODELS_LOCATION "kila.obj", model, true);

    if (model.stride != 14) {
      throw std::runtime_error("model is not valid!");
    }

    std::vector<unsigned char> image_data;
    int image_width, image_height, image_num_components, image_data_size;

    VkTools::loadTextureDataFromFile(DATA_LOCATION "KilaMain_Diff.png", 0, image_data, &image_width, &image_height, &image_num_components, &image_data_size);
    std::cout << "loaded " << image_width << "X" << image_height << " (" << image_data_size / 1048576 << "MB)" << std::endl;
		
    std::vector<unsigned char> normal_data;
    int normal_width, normal_height, normal_num_components, normal_data_size;

    VkTools::loadTextureDataFromFile(DATA_LOCATION "KilaMain_Norm.png", 0, normal_data, &normal_width, &normal_height, &normal_num_components, &normal_data_size);
    std::cout << "loaded " << normal_width << "X" << normal_height << " (" << normal_data_size / 1048576 << "MB)" << std::endl;
		
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(width, height, WINDOW_TITLE.c_str(), nullptr, nullptr);

    //Create instance, select physical device, check queues and device extensions
    vkTest = new VkTest;
  
#ifdef VK_DEBUG
    std::vector<const char *> requiredInstanceLayers;
    requiredInstanceLayers.push_back("VK_LAYER_LUNARG_standard_validation");
    requiredInstanceLayers.push_back("VK_LAYER_LUNARG_api_dump");
    vkTest->createInstance(requiredInstanceLayers);
#else
    vkTest->createInstance();
#endif
    
    vkTest->selectPhysicalDevice();
    depthFormat = vkTest->findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM },

					      VK_IMAGE_TILING_OPTIMAL,

					      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT );

#ifdef VK_DEBUG

    std::cout << "Selected format " << depthFormat << std::endl;

#endif

    vkTest->selectGraphicsQueue();
    vkTest->selectComputeQueue();
    //        vkTest->checkDeviceExtensions({ "VK_KHR_push_descriptor" });
    vkTest->checkDeviceExtensions();

    /*Set logical device with tessellation enabled
      VkPhysicalDeviceFeatures features = {};
      features.tessellationShader = VK_TRUE;
      vkTest->setLogicalDevice(features);
    */
    //Set Logical device
    vkTest->setLogicalDevice();

    //Set resize callback
    glfwSetWindowSizeCallback(window, onWindowResize);

    //Set keyboard callback
    glfwSetKeyCallback(window, onKey);

    //Set Mouse button callback
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetScrollCallback(window, onMouseScroll);

    //Set camera
    glm::vec3 model_centre = glm::vec3(0.5f * (model.min_x + model.max_x),

				       0.5f * (model.min_y + model.max_y),

				       0.5f * (model.min_z + model.max_z));
    glm::vec3 model_size = glm::vec3(model.max_x - model.min_x,
				     model.max_y - model.min_y,
				     model.max_z - model.min_z);

    max_view = 2.0f * glm::length(model_size);
    if (max_view < 100.0f) max_view = 100.0f;

    camera = new Camera(model_centre - 1.5f * glm::vec3(0.0f, 0.0f, glm::length(model_size)));
    camera->lookAt(model_centre);
    camera->set_proj_matrix(width / (float)height, 0.01f, max_view);
    camera->set_speed(10000.0f);

    //Set light
    float azimuth = glm::radians(180.0f), elevation = glm::radians(-45.0f);
    float cos_azimuth = cos(azimuth);
    float sin_azimuth = sin(azimuth);
    float cos_elevation = cos(elevation);
    float sin_elevation = sin(elevation);
    glm::vec4 light = glm::vec4(cos_elevation * sin_azimuth, sin_elevation, cos_elevation * cos_azimuth, 1.0f);

    //Create surface
    VkSurfaceKHR surface;
    VkInstance instance = vkTest->instance();
    device = vkTest->device();
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    vkTest->createSurface(surface);

    //Create swapchain
    vkTest->createSwapchain( {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}, videoBuffer, mode);

    //Create render pass with one color and one depth attachment
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = vkTest->swapchainImageFormat();
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

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkSubpassDependency> subpass_dependencies = {
      {
	VK_SUBPASS_EXTERNAL,
	0,
	VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	VK_ACCESS_MEMORY_READ_BIT,
	VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	VK_DEPENDENCY_BY_REGION_BIT
      },
      {
	0,
	VK_SUBPASS_EXTERNAL,
	VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	VK_ACCESS_MEMORY_READ_BIT,
	VK_DEPENDENCY_BY_REGION_BIT
      }
    };

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    vkTest->createRenderPass({ colorAttachment, depthAttachment }, { subpass }, subpass_dependencies);

    //Setup the descriptor set layout

    //One uniform buffer accessible by the vertex shader
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    //two samplers accessible by the fragment shader
    VkDescriptorSetLayoutBinding samplerLayoutBinding1 = {};
    samplerLayoutBinding1.binding = 1;
    samplerLayoutBinding1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding1.descriptorCount = 1;
    samplerLayoutBinding1.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding2 = {};
    samplerLayoutBinding2.binding = 2;
    samplerLayoutBinding2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding2.descriptorCount = 1;
    samplerLayoutBinding2.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    vkTest->createDescriptorSetLayout({ uboLayoutBinding, samplerLayoutBinding1, samplerLayoutBinding2 });

    //Load shaders
    shaderStages.resize(2);
    vkTest->createShaderStage(SHADERS_LOCATION "model/shader.vert.spv", VK_SHADER_STAGE_VERTEX_BIT, shaderStages[0]);
    vkTest->createShaderStage(SHADERS_LOCATION "model/shader.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT, shaderStages[1]);

    //Set the graphics pipeline
    vkTest->createGraphicsPipeline(shaderStages);

    //Setup framebuffer
    vkTest->createFrameBuffers();

    //Create the command pool
    vkTest->createCommandPool();

    //Create the vertex buffer
    vkTest->createDataBuffer(model.Data, vertexBuffer, vertexBufferMemory, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    //Create the diffuse texture 
    VkSampler SamplerDiffuse;
    VkImage ImageDiffuse;
    VkDeviceMemory ImageMemoryDiffuse;
    VkImageView ImageViewDiffuse;
    vkTest->createCombinedImageSampler(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, { (uint32_t)image_width, (uint32_t)image_height, 1 },

				       1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR,

				       VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT,

				       VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.0f, false, 1.0f, false, VK_COMPARE_OP_ALWAYS, 0.0f, 1.0f, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,

				       false, SamplerDiffuse, ImageDiffuse, ImageMemoryDiffuse, ImageViewDiffuse);

    //Create texture data staging buffers
    VkBuffer stagingBufferImage;

    VkDeviceMemory stagingBufferMemoryImage;

    vkTest->createAllocateAndBindBuffer(image_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferImage, stagingBufferMemoryImage);


    //Copy data to staging buffer
    vkTest->copyDataToBuffer(stagingBufferMemoryImage, image_data.data(), static_cast<size_t>(image_data_size));
    

    //Copy staging buffer to texture
    vkTest->copyBufferToImage(stagingBufferImage, ImageDiffuse, image_width, image_height);
    image_data.clear();
		
    //Create the normal texture
    VkSampler SamplerNormal;
    VkImage ImageNormal;
    VkDeviceMemory ImageMemoryNormal;
    VkImageView ImageViewNormal;
    vkTest->createCombinedImageSampler(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, { (uint32_t)normal_width, (uint32_t)normal_height, 1 },

				       1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR,

				       VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT,

				       VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.0f, false, 1.0f, false, VK_COMPARE_OP_ALWAYS, 0.0f, 1.0f, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,

				       false, SamplerNormal, ImageNormal, ImageMemoryNormal, ImageViewNormal);
		
    //Create texture data staging buffers
    VkBuffer stagingBufferNormal;

    VkDeviceMemory stagingBufferMemoryNormal;

    vkTest->createAllocateAndBindBuffer(normal_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferNormal, stagingBufferMemoryNormal);

		
    //Copy data to staging buffer
    vkTest->copyDataToBuffer(stagingBufferMemoryNormal, normal_data.data(), static_cast<size_t>(normal_data_size));
		
    //Copy staging buffer to texture
    vkTest->copyBufferToImage(stagingBufferNormal, ImageNormal, normal_width, normal_height);
    normal_data.clear();
		
    //Create the uniform buffer
    uniforms.resize(1);
    vkTest->createDataDoubleBuffer(uniforms, stagingUniformBuffer, uniformBuffer, stagingUniformBufferMemory, uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    //Create the descriptor pool
    vkTest->createDescriptorPool(false, 1, { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 } });

    //Allocate and update the descriptor set
    vkTest->createDescriptorSets({ { uniformBuffer, sizeof(UniformBufferObject), 0 } }, { { ImageViewDiffuse, SamplerDiffuse, 1 }, { ImageViewNormal, SamplerNormal, 2 } });

    //Setup the command buffers
    vkTest->allocateCommandBuffers();

    //Record commands
    vkTest->recordCommandBuffers(model);

    //Setup semaphores
    vkTest->setupDrawSemaphores();

    uint32_t frame = 0;
    float total_time = 0.0f;
#define TIME_INTERVAL 1.0f

    //Initial position and orientation of model
    glm::mat4 model = glm::mat4(1.0f);
    glm::vec3 translate = glm::vec3(0.0f);

    model = glm::translate(model, model_centre);
    model = glm::rotate(model, glm::radians(180.0f), camera->get_right());
    model = glm::translate(model, -model_centre);

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      //static auto startTime = std::chrono::high_resolution_clock::now();
      auto startFrameTime = std::chrono::high_resolution_clock::now();

      //float time = std::chrono::duration_cast<std::chrono::microseconds>(startFrameTime - startTime).count() / 1e6f;

      //Check inputs
      if (mouse.pressed) {
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	switch (mouse.button) {
	case GLFW_MOUSE_BUTTON_LEFT:
	  translate = (float(xpos - mouse.xpos) * camera->get_right() - float(ypos - mouse.ypos) * camera->get_up()) * 0.1f;
	  model = glm::translate(model, translate);
	  model_centre += translate;
	  break;
	case GLFW_MOUSE_BUTTON_RIGHT:
	  model = glm::translate(model, model_centre);
	  model = glm::rotate(model, float(xpos - mouse.xpos) * 0.01f, camera->get_up());
	  model = glm::rotate(model, float(ypos - mouse.ypos) * 0.01f, camera->get_right());
	  model = glm::translate(model, -model_centre);
	  break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
	  azimuth += float(xpos - mouse.xpos) * 0.005f;
	  elevation += float(ypos - mouse.ypos) * 0.005f;
	  cos_azimuth = cos(azimuth);
	  sin_azimuth = sin(azimuth);
	  cos_elevation = cos(elevation);
	  sin_elevation = sin(elevation);
	  light = glm::vec4(cos_elevation * sin_azimuth, sin_elevation, cos_elevation * cos_azimuth, 1.0f);
	  break;
	}
	mouse.xpos = xpos;
	mouse.ypos = ypos;
      }

      //Update uniform buffer
      UniformBufferObject ubo = {};
      ubo.mv  = camera->get_view_matrix() * model;
      ubo.mvp = camera->get_proj_matrix() * ubo.mv;
      ubo.light = light;

      vkTest->copyDataToBuffer(stagingUniformBufferMemory, &ubo, sizeof(ubo));
      vkTest->copyBufferToBuffer(stagingUniformBuffer, uniformBuffer, sizeof(ubo));

      //Draw frame
      vkTest->draw();

      auto endFrameTime = std::chrono::high_resolution_clock::now();
      frameTime = std::chrono::duration_cast<std::chrono::microseconds>(endFrameTime - startFrameTime).count() / 1e6f;
      total_time += frameTime;
      frame++;
      if (total_time >= TIME_INTERVAL) {
	std::stringstream stream;
	stream << WINDOW_TITLE << " - " << std::fixed << std::setprecision(0) << (float)frame / total_time << " fps";
	glfwSetWindowTitle(window, stream.str().c_str());
	frame = 0;
	total_time = 0.0f;
      }
    }

    //Cleanup
    vkDeviceWaitIdle(device);
    for (auto shaderStage: shaderStages) {
      vkDestroyShaderModule(device, shaderStage.module, nullptr);
    }
    vkDestroyBuffer(device, stagingBufferImage, nullptr);
    vkDestroyBuffer(device, stagingBufferNormal, nullptr);
    vkFreeMemory(device, stagingBufferMemoryImage, nullptr);
    vkFreeMemory(device, stagingBufferMemoryNormal, nullptr);
    vkDestroySampler(device, SamplerDiffuse, nullptr);
    vkDestroySampler(device, SamplerNormal, nullptr);
    vkDestroyImage(device, ImageDiffuse, nullptr);
    vkDestroyImage(device, ImageNormal, nullptr);
    vkFreeMemory(device, ImageMemoryDiffuse, nullptr);
    vkFreeMemory(device, ImageMemoryNormal, nullptr);
    vkDestroyImageView(device, ImageViewDiffuse, nullptr);
    vkDestroyImageView(device, ImageViewNormal, nullptr);
    delete vkTest;

    glfwDestroyWindow(window);
    glfwTerminate();

  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
