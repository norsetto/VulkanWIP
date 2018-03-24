/*
 * Based on https://github.com/PacktPublishing/Vulkan-Cookbook
 * Drawing_particles_using_compute_and_graphics_pipelines
 * by Pawel Lapinski (https://www.linkedin.com/in/pawel-lapinski-84522329)
 * 
 */
#include "config.h"
#include "vkBase.hpp"
#include "vkTools.hpp"
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
#endif

#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

#define NUM_PARTICLES 2000

class VkTest: public VkBase
{
public:
    using VkBase::createFrameBuffers;
    void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStage);
    void createComputePipeline(VkPipelineShaderStageCreateInfo shaderStage);
    void createCommandPools(VkCommandPool &computeCommandPool);
    void createRenderPass();
    virtual void createFrameBuffers();
    void createDescriptorSetLayouts(void);
    void allocateDescriptorSets();
    void updateDescriptorSets(VkBuffer uniformBuffer, VkBufferView vertexBufferView);
    void recordCommandBuffers(uint32_t frameIndex, float deltaTime, float time);
    void createStorageTexelBuffer(VkDeviceSize size, VkBuffer & storage_texel_buffer, VkDeviceMemory & memory_object, VkBufferView & storage_texel_buffer_view);
    void draw(float frameTime, float time);
    
    //Setters
    void setDepthFormat(VkFormat depthFormat) { this->depthFormat = depthFormat; };
    
    ~VkTest(void)
    {
        depth.destroy(m_device);
        for (auto descriptorSetLayout : descriptorSetLayouts) {
            if (descriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_device, descriptorSetLayout, nullptr);
                descriptorSetLayout = VK_NULL_HANDLE;
            }
            descriptorSetLayouts.clear();
        }
        vkDestroyPipelineLayout(m_device, computePipelineLayout, nullptr);
        vkDestroyPipeline(m_device, computePipeline, nullptr);
        vkDestroySemaphore(m_device, computeSemaphore, nullptr);
        vkDestroyFence(m_device, computeFence, nullptr);
    };
    
private:
    Image depth;
    VkFormat depthFormat;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    VkSemaphore computeSemaphore;
    VkFence computeFence;
};

VkTest *vkTest;

VkDevice device = VK_NULL_HANDLE;

const VkBase::VideoBuffer videoBuffer = VkBase::TRIPLE_BUFFER;
int width = 800;
int height = 600;
std::string WINDOW_TITLE = "Vulkan Test";
const std::string pipelineCacheFilename = "particles_pipeline_cache.bin";
const VkPresentModeKHR mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
std::vector<std::vector<VkPipelineShaderStageCreateInfo>> shaderStages = {};
float frameTime;
Camera *camera;
struct PARTICLE {
    glm::vec4 position;
    glm::vec4 color;
};
std::vector<PARTICLE> particles;
 
struct UniformBufferObject {
  glm::mat4 mv;
  glm::mat4 p;
  glm::vec4 l;
};

std::vector<UniformBufferObject> uniforms;
VkBuffer uniformBuffer, stagingUniformBuffer, vertexBuffer;
VkDeviceMemory uniformBufferMemory, stagingUniformBufferMemory, vertexBufferMemory;
VkBufferView vertexBufferView;
VkCommandBuffer computeCommandBuffer;
VkCommandPool computeCommandPool;
VkPipelineCache computePipelineCache, graphicsPipelineCache, pipelineCache;

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
  vkTest->createGraphicsPipeline(shaderStages[0]);
  vkTest->createFrameBuffers();
  
  width = w;
  height = h;
  camera->set_proj_matrix(width / (float)height, 0.01f, 100.0f);
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
    deviceFeatures.geometryShader = VK_TRUE;
    vkTest->setLogicalDevice(deviceFeatures);

    //Create the command pools
    vkTest->createCommandPools(computeCommandPool);

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

    //Create swapchain
    vkTest->createSwapchain({static_cast<uint32_t>(width), static_cast<uint32_t>(height)}, videoBuffer, mode);

    //Create render pass with two color and two depth attachments
    vkTest->createRenderPass();
    
    //Load shaders
    shaderStages.resize(2);
    shaderStages[0].resize(3);
    vkTest->createShaderStage(SHADERS_LOCATION "particles/shader.vert", VK_SHADER_STAGE_VERTEX_BIT, shaderStages[0][0]);
    vkTest->createShaderStage(SHADERS_LOCATION "particles/shader.geom", VK_SHADER_STAGE_GEOMETRY_BIT, shaderStages[0][1]);
    vkTest->createShaderStage(SHADERS_LOCATION "particles/shader.frag", VK_SHADER_STAGE_FRAGMENT_BIT, shaderStages[0][2]);
    
    shaderStages[1].resize(1);
    vkTest->createShaderStage(SHADERS_LOCATION "particles/shader.comp", VK_SHADER_STAGE_COMPUTE_BIT, shaderStages[1][0]);

    //Create the uniform buffer
    uniforms.resize(1);
    vkTest->createDataDoubleBuffer(uniforms, stagingUniformBuffer, uniformBuffer, stagingUniformBufferMemory, uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    //Create the vertex buffer
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for( uint32_t i = 0; i < NUM_PARTICLES; ++i ) {
        PARTICLE particle;
        float azimuth = 360.0f * dist(mt);
        float elevation = -90.0f * dist(mt) * 180.0f;
		float radius = 1.0f + dist(mt);
        particle.position = glm::vec4(radius * cos(elevation)* cos(azimuth),
                                      radius * sin(elevation),
                                      radius * cos(elevation) * sin(azimuth), 1.0f);
        particle.color = glm::vec4(0.50f + dist(mt) * 0.5f,
								   0.25f + dist(mt) * 0.5f,
								   0.00f + dist(mt) * 0.5f,
								   0.50f + dist(mt) * 1.5f);
        particles.push_back(particle);
    }
    VkDeviceSize particlesSize = sizeof(PARTICLE)*particles.size();
    vkTest->createStorageTexelBuffer(particlesSize, vertexBuffer, vertexBufferMemory, vertexBufferView);
    
    VkBuffer stagingVertexBuffer;
    VkDeviceMemory stagingVertexBufferMemory;
    
    vkTest->createAllocateAndBindBuffer(particlesSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingVertexBuffer, stagingVertexBufferMemory);
    vkTest->copyDataToBuffer(stagingVertexBufferMemory, particles.data(), particlesSize);
    vkTest->copyBufferToBuffer(stagingVertexBuffer, vertexBuffer, particlesSize);
    
    //Create the descriptor pool
    vkTest->createDescriptorPool(false, 2, { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1 } });

    //Create Descriptor Set layouts
    vkTest->createDescriptorSetLayouts();
    
    //Allocate the descriptor sets
    vkTest->allocateDescriptorSets();
    
    //Update the descriptor sets
    vkTest->updateDescriptorSets(uniformBuffer, vertexBufferView);

    //Set the graphics pipeline
    size_t cacheSize = vkTest->loadPipelineCacheFromDisk(pipelineCacheFilename, true, graphicsPipelineCache);
    vkTest->createGraphicsPipeline(shaderStages[0]);

    //Set up the compute pipeline
    if (cacheSize == 0) {
        vkTest->createPipelineCache(&computePipelineCache);
        vkTest->createPipelineCache(&pipelineCache);
    }
    else
        computePipelineCache = graphicsPipelineCache;
    vkTest->createComputePipeline(shaderStages[1][0]);
    if (cacheSize == 0) {
        VkPipelineCache pipelineCaches[2] = { computePipelineCache, graphicsPipelineCache };
        vkMergePipelineCaches(device, pipelineCache, 2, pipelineCaches);
    }
    else
        pipelineCache = graphicsPipelineCache;
    
    //Setup framebuffer
    vkTest->createFrameBuffers();

    //Set camera
    camera = new Camera(glm::vec3(0.0f, 0.0f, -4.0f));
    camera->lookAt(glm::vec3(0.0f));
    camera->set_proj_matrix(width / (float)height, 0.01f, 100.0f);
    camera->set_speed(1000.0f);

    //Set light
    float lightRadius = 1000.0f;
    glm::vec4 light = lightRadius * glm::vec4(0.0f, 0.7071f, -0.7071f, 1.0f);
    glm::mat4 lightMatrix = glm::mat4(1.0f);

    //Setup the command buffers
    vkTest->allocateCommandBuffers(2);
    std::vector<VkCommandBuffer> commandBuffers;
    vkTest->allocateCommandBuffers(computeCommandPool, 1, commandBuffers);
    computeCommandBuffer = commandBuffers[0];
    
    //Setup semaphores
    vkTest->setupDrawSemaphores();

    uint32_t frame = 0;
    float total_time = 0.0f;
#define TIME_INTERVAL 1.0f

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
                    camera->rotate((float)(xpos - mouse.xpos), (float)(ypos - mouse.ypos));
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
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
        ubo.mv = camera->get_view_matrix();
        ubo.p = camera->get_proj_matrix();
        ubo.l = lightMatrix * light;

        vkTest->copyDataToBuffer(stagingUniformBufferMemory, &ubo, sizeof(ubo));
        vkTest->copyBufferToBuffer(stagingUniformBuffer, uniformBuffer, sizeof(ubo));

        //Draw frame
        vkTest->draw(frameTime, time);

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
    vkTest->savePipelineCacheToDisk(pipelineCacheFilename, pipelineCache);
    vkDeviceWaitIdle(device);
    
    for (auto shaderStage: shaderStages) {
        for (auto shader: shaderStage) {
            vkDestroyShaderModule(device, shader.module, nullptr);
        }
    }
    if (computePipelineCache == graphicsPipelineCache) {
        vkDestroyPipelineCache(device, graphicsPipelineCache, nullptr);
    } else {
        vkDestroyPipelineCache(device, graphicsPipelineCache, nullptr);
        vkDestroyPipelineCache(device, computePipelineCache, nullptr);
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
    }
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkDestroyBuffer(device, stagingUniformBuffer, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkDestroyBuffer(device, stagingVertexBuffer, nullptr);
    vkDestroyBufferView(device, vertexBufferView, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    vkFreeMemory(device, stagingUniformBufferMemory, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkFreeMemory(device, stagingVertexBufferMemory, nullptr);
    vkFreeCommandBuffers(device, computeCommandPool, 1, &computeCommandBuffer);
    vkDestroyCommandPool(device, computeCommandPool, nullptr);
    
    delete vkTest;
    
    glfwDestroyWindow(window);
    glfwTerminate();

  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void VkTest::createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStage)
{
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(PARTICLE);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[0].offset = 0;

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[1].offset = sizeof(glm::vec4);

    //Set state for the fixed functionality pipeline stages
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
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
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 0.0f;
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
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts[0];

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStage.size());
    pipelineInfo.pStages = shaderStage.data();
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

    if (vkCreateGraphicsPipelines(m_device, graphicsPipelineCache, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
}

void VkTest::createComputePipeline(VkPipelineShaderStageCreateInfo shaderStage)
{
    VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(float)};

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts[1];
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("could not create compute pipeline layout!");
    }
 
    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = shaderStage;
    computePipelineCreateInfo.layout = computePipelineLayout;

    if (vkCreateComputePipelines(m_device, computePipelineCache, 1, &computePipelineCreateInfo, nullptr, &computePipeline) != VK_SUCCESS) {
      throw std::runtime_error("could not create compute pipeline!");
    }
    
    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    if(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &computeSemaphore) != VK_SUCCESS) {
      throw std::runtime_error("could not create a compute semaphore!");
    }
    
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(m_device, &fenceCreateInfo, nullptr, &computeFence) != VK_SUCCESS) {
      throw std::runtime_error("could not create a compute fence!");
    }
}

void VkTest::createFrameBuffers()
{      
    //Create image and image view for the depth buffer
    depth.destroy(m_device);

    create2DImageAndView(depthFormat, m_swapChainExtent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth);

    //Create framebuffer
    m_swapChainFramebuffers.resize(m_swapChainImageView.size());
    for (size_t i = 0; i < m_swapChainImageView.size(); i++) {
      createFramebuffer({ m_swapChainImageView[i], depth.view }, m_swapChainExtent, 1, m_swapChainFramebuffers[i]);
    }
}

void VkTest::createRenderPass()
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
    VkBase::createRenderPass({ colorAttachment, depthAttachment }, { subpass }, subpass_dependencies);
}

void VkTest::createDescriptorSetLayouts()
{
    //Set 0 is for the uniform buffer, set 1 for the texel storage buffer
    descriptorSetLayouts.resize(2);
    
    //One uniform buffer accessible by the vertex and geometry shader
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT;

    //One storage texel buffer accessible by the compute shader
    VkDescriptorSetLayoutBinding compLayoutBinding = {};
    compLayoutBinding.binding = 0;
    compLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    compLayoutBinding.descriptorCount = 1;
    compLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    vkTest->createDescriptorSetLayout({ uboLayoutBinding }, descriptorSetLayouts[0]);
    vkTest->createDescriptorSetLayout({ compLayoutBinding }, descriptorSetLayouts[1]); 
}

//Allocate descriptor sets
void VkTest::allocateDescriptorSets()
{
    descriptorSets.resize(2);
    VkBase::allocateDescriptorSets(descriptorSetLayouts, descriptorSets);
}

//Update the descriptor sets
void VkTest::updateDescriptorSets(VkBuffer uniformBuffer, VkBufferView vertexBufferView)
{
    BufferDescriptorInfo buffer_descriptor_update = {
        descriptorSets[0],
        0,
        0,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        {
            {
                uniformBuffer,
                0,
                VK_WHOLE_SIZE
            }
        }
    };

    TexelBufferDescriptorInfo storage_texel_buffer_descriptor_update = {
        descriptorSets[1],
        0,
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
        {
            {
                vertexBufferView
            }
        }
    };

    VkBase::updateDescriptorSets({}, { buffer_descriptor_update }, { storage_texel_buffer_descriptor_update }, {});    
}

void VkTest::createStorageTexelBuffer(VkDeviceSize size, VkBuffer & storage_texel_buffer, VkDeviceMemory & memory_object, VkBufferView & storage_texel_buffer_view)
{
    VkBase::createStorageTexelBuffer(VK_FORMAT_R32G32B32A32_SFLOAT, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, false, storage_texel_buffer, memory_object, storage_texel_buffer_view);
}

void VkTest::createCommandPools(VkCommandPool &computeCommandPool)
{
    //Create graphics queue command pool
    createCommandPool();
    
    //Create compute queue command pool
    createCommandPool(computeCommandPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, m_computeFamily);
}

void VkTest::recordCommandBuffers(uint32_t frameIndex, float deltaTime, float time)
{   
    //Record command buffer with compute shader dispatch
    vkWaitForFences(m_device, 1, &computeFence, VK_FALSE, std::numeric_limits<uint64_t>::max());
    vkResetFences(m_device, 1, &computeFence);

    VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

    vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descriptorSets[1], 0, nullptr);
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    float pushConstants[2] = {deltaTime, time};
    vkCmdPushConstants(computeCommandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(float), pushConstants);
    vkCmdDispatch(computeCommandBuffer, NUM_PARTICLES / 32 + 1, 1, 1);
    vkEndCommandBuffer(computeCommandBuffer);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeSemaphore;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffer;

    vkQueueSubmit(m_computeQueue, 1, &submitInfo, computeFence);

    //Prepare drawing functions
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_renderPass;
    renderPassBeginInfo.renderArea.offset = { 0, 0 };
    renderPassBeginInfo.renderArea.extent = m_swapChainExtent;
    std::array<VkClearValue, 2> clearValues = {};
    clearValues[0].color = { {0.1176f, 0.5647f, 1.0f, 1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    for (size_t i = 0; i < m_commandBuffers[frameIndex].size(); i++) {

      //Set target frame buffer
      renderPassBeginInfo.framebuffer = m_swapChainFramebuffers[i];
/*
      //Transfer ownership of texel storage buffer if graphics and compute queues are from different families
      if (m_computeQueue != m_graphicsQueue) {
        VkBufferMemoryBarrier bufferMemoryBarrier = {};
        bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufferMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferMemoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        bufferMemoryBarrier.srcQueueFamilyIndex = m_computeFamily;
        bufferMemoryBarrier.dstQueueFamilyIndex = m_graphicsFamily;
        bufferMemoryBarrier.buffer = vertexBuffer;
        bufferMemoryBarrier.size = VK_WHOLE_SIZE;
      
        vkCmdPipelineBarrier(m_commandBuffers[frameIndex][i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &bufferMemoryBarrier, 0, nullptr);
      }
*/
      //Start the render pass
      vkBeginCommandBuffer(m_commandBuffers[frameIndex][i], &beginInfo);

      //Render pass
      vkCmdBeginRenderPass(m_commandBuffers[frameIndex][i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(m_commandBuffers[frameIndex][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

      const VkDeviceSize offset = { 0 };
      vkCmdBindVertexBuffers(m_commandBuffers[frameIndex][i], 0, 1, &vertexBuffer, &offset);
      vkCmdBindDescriptorSets(m_commandBuffers[frameIndex][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);
      
      vkCmdDraw(m_commandBuffers[frameIndex][i], static_cast<uint32_t>(particles.size()), 1, 0, 0);
      
      vkCmdEndRenderPass(m_commandBuffers[frameIndex][i]);

      //End render pass
      if (vkEndCommandBuffer(m_commandBuffers[frameIndex][i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
}

void VkTest::draw(float frameTime, float time)
{
  static uint32_t frameIndex = 0;
  
  vkWaitForFences(m_device, 1, &m_drawingFinishedFence[frameIndex], VK_FALSE, std::numeric_limits<uint64_t>::max());
  vkResetFences(m_device, 1, &m_drawingFinishedFence[frameIndex]);
  
  uint32_t imageIndex;
  vkAcquireNextImageKHR(m_device, m_swapChain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphore[frameIndex], VK_NULL_HANDLE, &imageIndex);

  //Record commands
  recordCommandBuffers(frameIndex, frameTime, time);
  
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphore[frameIndex], computeSemaphore };
  VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT };
  submitInfo.waitSemaphoreCount = 2;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_commandBuffers[frameIndex][imageIndex];

  VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphore[frameIndex] };
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_drawingFinishedFence[frameIndex]) != VK_SUCCESS) {
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
  
  frameIndex = (frameIndex + 1) % m_imageAvailableSemaphore.size();
}

