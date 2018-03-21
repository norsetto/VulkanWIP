#include "config.h"
#include "vkBase.hpp"
#include "vkTools.hpp"
#include "asModel.hpp"
#include "camera.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#ifdef VK_DEBUG
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/euler_angles.hpp>
#endif
#include <limits>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

class VkTest: public VkBase
{
public:
    
    using VkBase::createFrameBuffers;
    void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStages, VkSampleCountFlagBits samples);
    void createRenderPass(VkSampleCountFlagBits samples);
    virtual void createFrameBuffers(VkSampleCountFlagBits samples);
    void createDescriptorSetLayouts(void);
    void allocateDescriptorSets(Model *model);
    void updateDescriptorSets(const VkBase::Buffer &uniformBuffer);
    void updateDescriptorSets(Model *model);
    void recordCommandBuffers(Model *model);
    VkSampleCountFlagBits getMaxSampleCount(void);
    float getMaxAnisotropy(void) { return m_devProperties.limits.maxSamplerAnisotropy; };
    //Setters
    void setDepthFormat(VkFormat depthFormat) { this->depthFormat = depthFormat; };
    
    ~VkTest(void)
    {
        depth.destroy(m_device);
        depthMSAA.destroy(m_device);
        colorMSAA.destroy(m_device);
        for (auto descriptorSetLayout : descriptorSetLayouts) {
            if (descriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_device, descriptorSetLayout, nullptr);
                descriptorSetLayout = VK_NULL_HANDLE;
            }
            descriptorSetLayouts.clear();
        }
    };
    
private:
    Image depth, depthMSAA, colorMSAA;
    VkFormat depthFormat;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkDescriptorSet> meshDescriptorSets;
};

VkTest *vkTest;
Model *model;

VkDevice device = VK_NULL_HANDLE;

const VkBase::VideoBuffer videoBuffer = VkBase::TRIPLE_BUFFER;
int width = 800;
int height = 600;
std::string WINDOW_TITLE = "Vulkan Test";
const std::string pipelineCacheFilename = "asModel_pipeline_cache.bin";
const VkPresentModeKHR mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {};
float frameTime;
Camera *camera;
float max_view;
VkSampleCountFlagBits sampleCountMSAA = VK_SAMPLE_COUNT_8_BIT;

struct UniformBufferObject {
  glm::mat4 mv;
  glm::mat4 mvp;
  glm::vec4 l;
};

std::vector<UniformBufferObject> uniforms;
VkBuffer uniformBuffer, stagingUniformBuffer;
VkDeviceMemory uniformBufferMemory, stagingUniformBufferMemory;

//Mouse struct for mouse handling
struct MOUSE {
  bool pressed;
  int button;
  double xpos;
  double ypos;
} mouse;

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
      camera->move_down(frameTime);
      break;
    case 'Z':
      camera->move_up(frameTime);
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
  vkTest->createSwapchain({static_cast<uint32_t>(w), static_cast<uint32_t>(h)}, videoBuffer, mode);
  vkTest->createGraphicsPipeline(shaderStages, sampleCountMSAA);
  vkTest->createFrameBuffers(sampleCountMSAA);
  vkTest->recordCommandBuffers(model);
  
  width = w;
  height = h;
  camera->set_proj_matrix(width / (float)height, 0.01f, max_view);
}

int main(int argc, char ** argv)
{
  std::cout << argv[0] << " Version " VERSION " build " BUILD_TYPE << std::endl;
  try {
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
    VkFormat depthFormat = vkTest->findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT );
    vkTest->setDepthFormat(depthFormat);
#ifdef VK_DEBUG
    std::cout << "Selected format " << depthFormat << std::endl;
#endif

    vkTest->selectGraphicsQueue();
    vkTest->selectComputeQueue();
    vkTest->checkDeviceExtensions();

    //Set Logical device
    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    vkTest->setLogicalDevice(deviceFeatures);

    //Create the command pool
    vkTest->createCommandPool();

    //Set resize callback
    glfwSetWindowSizeCallback(window, onWindowResize);

    //Set keyboard callback
    glfwSetKeyCallback(window, onKey);

    //Set Mouse button callback
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetScrollCallback(window, onMouseScroll);

    //Create surface
    VkSurfaceKHR surface;
    VkInstance instance = vkTest->instance();
    device = vkTest->device();
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    vkTest->createSurface(surface);

    //Load assets
    model = new Model(vkTest);
    if (argc == 1)
        model->load(MODELS_LOCATION "kila.dae", true, true, true, vkTest->getMaxAnisotropy(), true);
    else
        model->load(argv[1], true, true, true, vkTest->getMaxAnisotropy(), true);

    //Create swapchain
    vkTest->createSwapchain({static_cast<uint32_t>(width), static_cast<uint32_t>(height)}, videoBuffer, mode);

    //Create render pass with two color and two depth attachments
    sampleCountMSAA = vkTest->getMaxSampleCount();
    vkTest->createRenderPass(sampleCountMSAA);
    
    //Load shaders
    shaderStages.resize(2);
    vkTest->createShaderStage(SHADERS_LOCATION "asModel/model.vert", VK_SHADER_STAGE_VERTEX_BIT, shaderStages[0]);
    vkTest->createShaderStage(SHADERS_LOCATION "asModel/model.frag", VK_SHADER_STAGE_FRAGMENT_BIT, shaderStages[1]);

    //Create the uniform buffer
    uniforms.resize(1);
    vkTest->createDataDoubleBuffer(uniforms, stagingUniformBuffer, uniformBuffer, stagingUniformBufferMemory, uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    //Create the descriptor pool
    vkTest->createDescriptorPool(false, 1 + model->getNumMeshes(), { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + model->getNumMeshes()}, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * model->getNumMeshes() } });

    //Create Descriptor Set layouts
    vkTest->createDescriptorSetLayouts();
    
    //Allocate and update the descriptor set for the uniform buffer
    vkTest->allocateDescriptorSets(model);
    vkTest->updateDescriptorSets(VkBase::Buffer(uniformBuffer, sizeof(UniformBufferObject), 0));
    vkTest->updateDescriptorSets(model);
    
    //Set the graphics pipeline
    vkTest->loadPipelineCacheFromDisk(pipelineCacheFilename, true);
    vkTest->createGraphicsPipeline(shaderStages, sampleCountMSAA);

    //Setup framebuffer
    vkTest->createFrameBuffers(sampleCountMSAA);

    //Set camera
    glm::vec3 model_centre = 0.5f * (model->min() + model->max());
    glm::vec3 model_size = (model->max() - model->min());

    max_view = 2.0f * glm::length(model_size);
    if (max_view < 1000.0f) max_view = 1000.0f;

    camera = new Camera(model_centre - 1.5f * glm::vec3(0.0f, 0.0f, glm::length(model_size)));
    camera->lookAt(model_centre);
    camera->set_proj_matrix(width / (float)height, 0.01f, max_view);
    camera->set_speed(10000.0f);

    //Set light
    float lightRadius = 1000.0f;
    glm::vec4 light = lightRadius * glm::vec4(0.0f, 0.7071f, -0.7071f, 1.0f);
    glm::mat4 lightMatrix = glm::mat4(1.0f);

    //Setup the command buffers
    vkTest->allocateCommandBuffers(2);

    //Record commands
    vkTest->recordCommandBuffers(model);

    //Setup semaphores
    vkTest->setupDrawSemaphores();

    uint32_t frame = 0;
    float total_time = 0.0f;
#define TIME_INTERVAL 1.0f

    //Initial position and orientation of model
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::vec3 translate = glm::vec3(0.0f);

    modelMatrix = glm::translate(modelMatrix, model_centre);
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), camera->get_right());
    modelMatrix = glm::translate(modelMatrix, -model_centre);

    model->setAnimation(0);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        static auto startTime = std::chrono::high_resolution_clock::now();
        auto startFrameTime = std::chrono::high_resolution_clock::now();

        float time = std::chrono::duration_cast<std::chrono::microseconds>(startFrameTime - startTime).count() / 1e6f;

        //Check inputs
        if (mouse.pressed) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            switch (mouse.button) {
                case GLFW_MOUSE_BUTTON_LEFT:
                    translate = glm::vec3(float(xpos - mouse.xpos), float(mouse.ypos - ypos), 0.0f) * 0.1f;
                    modelMatrix = glm::translate(modelMatrix, translate);
                    model_centre += translate;
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
                    modelMatrix = glm::translate(modelMatrix, model_centre);
                    modelMatrix = glm::rotate(modelMatrix, float(xpos - mouse.xpos) * 0.01f, glm::vec3(0.0f, 1.0f, 0.0f));
                    modelMatrix = glm::rotate(modelMatrix, float(ypos - mouse.ypos) * 0.01f, glm::vec3(1.0f, 0.0f, 0.0f));
                    modelMatrix = glm::translate(modelMatrix, -model_centre);
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    lightMatrix = glm::rotate(lightMatrix, float(xpos - mouse.xpos) * 0.005f, glm::vec3(0.0f, 1.0f, 0.0f));
                    lightMatrix = glm::rotate(lightMatrix, float(ypos - mouse.ypos) * 0.005f, glm::vec3(1.0f, 0.0f, 0.0f));
                    break;
            }
            mouse.xpos = xpos;
            mouse.ypos = ypos;
        }

        //Update uniform buffer
        UniformBufferObject ubo = {};
        ubo.mv = camera->get_view_matrix() * modelMatrix;
        ubo.mvp = camera->get_proj_matrix() * ubo.mv;
        ubo.l = lightMatrix * light;

        vkTest->copyDataToBuffer(stagingUniformBufferMemory, &ubo, sizeof(ubo));
        vkTest->copyBufferToBuffer(stagingUniformBuffer, uniformBuffer, sizeof(ubo));

        //Update material buffers with animation data
        model->update(time);

        //Draw frame
        vkTest->draw();

        auto endFrameTime = std::chrono::high_resolution_clock::now();
        frameTime = std::chrono::duration_cast<std::chrono::microseconds>(endFrameTime - startFrameTime).count() / 1e6f;
        total_time += frameTime;
        frame++;
        if (total_time >= TIME_INTERVAL) {
            std::stringstream stream;
            stream << WINDOW_TITLE << " - " << std::fixed << std::setprecision(0) << (float)frame / total_time << " fps - ";
            stream << sampleCountMSAA << "x MSAA";
            glfwSetWindowTitle(window, stream.str().c_str());
            frame = 0;
            total_time = 0.0f;
        }
    }

    //Cleanup
    vkTest->savePipelineCacheToDisk(pipelineCacheFilename);
    vkDeviceWaitIdle(device);
    
    for (auto shaderStage: shaderStages) {
      vkDestroyShaderModule(device, shaderStage.module, nullptr);
    }
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkDestroyBuffer(device, stagingUniformBuffer, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    vkFreeMemory(device, stagingUniformBufferMemory, nullptr);

    delete model;
    delete vkTest;
    
    glfwDestroyWindow(window);
    glfwTerminate();

  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void VkTest::createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStages, VkSampleCountFlagBits samples)
{
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Model::Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 7> attributeDescriptions = {};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = 0;

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = 3 * sizeof(float);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = 6 * sizeof(float);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[3].offset = 8 * sizeof(float);

    attributeDescriptions[4].binding = 0;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[4].offset = 11 * sizeof(float);

    attributeDescriptions[5].binding = 0;
    attributeDescriptions[5].location = 5;
    attributeDescriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[5].offset = 14 * sizeof(float);

    attributeDescriptions[6].binding = 0;
    attributeDescriptions[6].location = 6;
    attributeDescriptions[6].format = VK_FORMAT_R32G32B32A32_SINT;
    attributeDescriptions[6].offset = 18 * sizeof(float);

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
    viewport.width = (float)m_swapChainExtent.width;
    viewport.height = (float)m_swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = m_swapChainExtent;

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
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.rasterizationSamples = samples;
    multisampling.minSampleShading = 0.2f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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
    colorBlending.blendConstants[0] = 1.0f;
    colorBlending.blendConstants[1] = 1.0f;
    colorBlending.blendConstants[2] = 1.0f;
    colorBlending.blendConstants[3] = 1.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
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
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (m_graphicsPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    }

    if (vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
}

void VkTest::createFrameBuffers(VkSampleCountFlagBits samples)
{      
    //Create image and image view for the depth buffer
    depth.destroy(m_device);
    depthMSAA.destroy(m_device);
    colorMSAA.destroy(m_device);

    create2DImageAndView(depthFormat, m_swapChainExtent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth);

    create2DImageAndView(depthFormat, m_swapChainExtent, 1, 1, samples, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depthMSAA);

    create2DImageAndView(m_swapchain_image_format, m_swapChainExtent, 1, 1, samples, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, colorMSAA);

    //Create framebuffer
    m_swapChainFramebuffers.resize(m_swapChainImageView.size());
    for (size_t i = 0; i < m_swapChainImageView.size(); i++) {
      createFramebuffer({ colorMSAA.view, m_swapChainImageView[i], depthMSAA.view, depth.view }, m_swapChainExtent, 1, m_swapChainFramebuffers[i]);
    }
}

void VkTest::createRenderPass(VkSampleCountFlagBits samples)
{
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

    VkAttachmentDescription colorAttachmentMSAA = {};
    colorAttachmentMSAA.format = vkTest->swapchainImageFormat();
    colorAttachmentMSAA.samples = samples;
    colorAttachmentMSAA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentMSAA.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentMSAA.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentMSAA.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentMSAA.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentMSAA.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRefMSAA = {};
    colorAttachmentRefMSAA.attachment = 1;
    colorAttachmentRefMSAA.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachmentMSAA = {};
    depthAttachmentMSAA.format = depthFormat;
    depthAttachmentMSAA.samples = samples;
    depthAttachmentMSAA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachmentMSAA.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachmentMSAA.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachmentMSAA.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachmentMSAA.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachmentMSAA.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 2;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkSubpassDependency> subpass_dependencies = {
      {
	VK_SUBPASS_EXTERNAL,
	0,
	VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	VK_ACCESS_MEMORY_READ_BIT,
	VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	VK_DEPENDENCY_BY_REGION_BIT
      },
      {
	0,
	VK_SUBPASS_EXTERNAL,
	VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
	VK_ACCESS_MEMORY_READ_BIT,
	VK_DEPENDENCY_BY_REGION_BIT
      }
    };

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentRefMSAA;
    VkBase::createRenderPass({ colorAttachmentMSAA, colorAttachment, depthAttachmentMSAA, depthAttachment }, { subpass }, subpass_dependencies);
}

void VkTest::allocateDescriptorSets(Model *model)
{
    descriptorSets.resize(2);
    VkBase::allocateDescriptorSets(descriptorSetLayouts[0], descriptorSets[0]);
    meshDescriptorSets.resize(model->getNumMeshes());
    for (uint32_t mesh = 0; mesh < model->getNumMeshes(); mesh++)
        VkBase::allocateDescriptorSets(descriptorSetLayouts[1], meshDescriptorSets[mesh]);
};

void VkTest::createDescriptorSetLayouts()
{
    //set 0 is for the uniform buffer, set 1 for the material and 3 samplers
    descriptorSetLayouts.resize(2);
    
    //One uniform buffer accessible by the vertex shader
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    //One uniform buffer accessible by the fragment shader
    VkDescriptorSetLayoutBinding matLayoutBinding = {};
    matLayoutBinding.binding = 0;
    matLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    matLayoutBinding.descriptorCount = 1;
    matLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    //three samplers accessible by the fragment shader
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

    VkDescriptorSetLayoutBinding samplerLayoutBinding3 = {};
    samplerLayoutBinding3.binding = 3;
    samplerLayoutBinding3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding3.descriptorCount = 1;
    samplerLayoutBinding3.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    vkTest->createDescriptorSetLayout({ uboLayoutBinding }, descriptorSetLayouts[0]);
    vkTest->createDescriptorSetLayout({ matLayoutBinding, samplerLayoutBinding1, samplerLayoutBinding2, samplerLayoutBinding3}, descriptorSetLayouts[1]); 

}

//Update the descriptor set for set 0 (uniform buffer)
void VkTest::updateDescriptorSets(const VkBase::Buffer &uniformBuffer)
{
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = uniformBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = uniformBuffer.size;

    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSets[0];
    descriptorWrite.dstBinding = uniformBuffer.binding;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);
}

//Update the descriptor set for set 1 (material buffer + 3 samplers for all meshes)
void VkTest::updateDescriptorSets(Model *model)
{
    VkDescriptorBufferInfo bufferInfo = {};
    std::vector<VkDescriptorImageInfo> imageInfo(3);
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    for (uint32_t mesh = 0; mesh < model->getNumMeshes(); mesh++ ) {
        bufferInfo.buffer = model->getMaterialBuffers(mesh).buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = model->getMaterialBuffers(mesh).size;

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = meshDescriptorSets[mesh];
        descriptorWrite.dstBinding = model->getMaterialBuffers(mesh).binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(descriptorWrite);

        uint32_t index = 0;
        for (auto image : { model->getDiffuse(mesh), model->getNormals(mesh), model->getSpeculars(mesh) }) {
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = image.view;
            imageInfo[index].sampler = image.sampler;

            VkWriteDescriptorSet descriptorWrite = {};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = meshDescriptorSets[mesh];
            descriptorWrite.dstBinding = image.binding;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pImageInfo = &imageInfo[index++];
            descriptorWrites.push_back(descriptorWrite);
        }
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        descriptorWrites.clear();
    }
}

void VkTest::recordCommandBuffers(Model *model)
{   
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_renderPass;
    renderPassBeginInfo.renderArea.offset = { 0, 0 };
    renderPassBeginInfo.renderArea.extent = m_swapChainExtent;
    std::array<VkClearValue, 4> clearValues = {};
    clearValues[0].color = { {0.1176f, 0.5647f, 1.0f, 1.0f} };
    clearValues[1].color = { {0.1176f, 0.5647f, 1.0f, 1.0f} };
    clearValues[2].depthStencil = { 1.0f, 0 };
    clearValues[3].depthStencil = { 1.0f, 0 };
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pInheritanceInfo = nullptr;

    for (size_t i = 0; i < m_commandBuffers.size(); i++) {
        for (size_t j = 0; j < m_commandBuffers[i].size(); j++) {

            //Set target frame buffer
            renderPassBeginInfo.framebuffer = m_swapChainFramebuffers[j];

            //Start the render pass
            vkBeginCommandBuffer(m_commandBuffers[i][j], &beginInfo);

            //Render pass
            vkCmdBeginRenderPass(m_commandBuffers[i][j], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(m_commandBuffers[i][j], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

            for (size_t mesh = 0; mesh < model->getNumMeshes(); ++mesh) {
                const VkDeviceSize offset = { 0 };
                vkCmdBindVertexBuffers(m_commandBuffers[i][j], 0, 1, model->getVertexBufferPointer(mesh), &offset);
                vkCmdBindIndexBuffer(m_commandBuffers[i][j], model->getIndexBuffer(mesh), 0, VK_INDEX_TYPE_UINT32);
                descriptorSets[1] = meshDescriptorSets[mesh];
                vkCmdBindDescriptorSets(m_commandBuffers[i][j], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
                vkCmdDrawIndexed(m_commandBuffers[i][j], model->getNumIndices(mesh), 1, 0, 0, 0);
            }

            vkCmdEndRenderPass(m_commandBuffers[i][j]);

            //End render pass
            if (vkEndCommandBuffer(m_commandBuffers[i][j]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }
}

//https://github.com/SaschaWillems/Vulkan/blob/master/examples/multisampling/multisampling.cpp
VkSampleCountFlagBits VkTest::getMaxSampleCount(void)
{
    VkSampleCountFlags counts = std::min(m_devProperties.limits.framebufferColorSampleCounts, m_devProperties.limits.framebufferDepthSampleCounts);
    
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }
    return VK_SAMPLE_COUNT_1_BIT;
}
