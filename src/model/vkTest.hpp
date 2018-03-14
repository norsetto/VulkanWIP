#pragma once

#include "vkTools.hpp"

#include <array>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//Mouse struct for mouse handling
struct MOUSE {
  bool pressed;
  int button;
  double xpos;
  double ypos;
} mouse;

struct UniformBufferObject {
  glm::mat4 mv;
  glm::mat4 mvp;
  glm::vec4 light;
};

struct Buffer {
  VkBuffer buffer;
  VkDeviceSize size;
  uint32_t binding;
};

struct Image {
  VkImageView view;
  VkSampler sampler;
  uint32_t binding;
};

std::vector<UniformBufferObject> uniforms;
VkBuffer vertexBuffer, uniformBuffer, stagingUniformBuffer;
VkDeviceMemory vertexBufferMemory, uniformBufferMemory, stagingUniformBufferMemory;

VkImage depthImage = VK_NULL_HANDLE;
VkDeviceMemory depthImageMemoryObject = VK_NULL_HANDLE;
VkImageView depthImageView = VK_NULL_HANDLE;
VkFormat depthFormat;

class VkTest: public VkBase
{
public:

  ~VkTest() {
    if (m_device != VK_NULL_HANDLE)
      {
	vkDeviceWaitIdle(m_device);
        
	if (depthImageView != VK_NULL_HANDLE) {
	  vkDestroyImageView(m_device, depthImageView, nullptr);

	  depthImageView = VK_NULL_HANDLE;
	}
	if (depthImageMemoryObject != VK_NULL_HANDLE) {

	  vkFreeMemory(m_device, depthImageMemoryObject, nullptr);

	  depthImageMemoryObject = VK_NULL_HANDLE;

	}
	if (depthImage != VK_NULL_HANDLE) {
	  vkDestroyImage(m_device, depthImage, nullptr);

	  depthImage = VK_NULL_HANDLE;
	}
	vkFreeMemory(m_device, vertexBufferMemory, nullptr);
	vkFreeMemory(m_device, uniformBufferMemory, nullptr);
	vkFreeMemory(m_device, stagingUniformBufferMemory, nullptr);
	vkDestroyBuffer(m_device, vertexBuffer, nullptr);
	vkDestroyBuffer(m_device, uniformBuffer, nullptr);
	vkDestroyBuffer(m_device, stagingUniformBuffer, nullptr);
      }
  }
    
  void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> shaderStages)
  {

    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = 14 * sizeof(float);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions = {};
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
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
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
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
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
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

    /*		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = 4;

		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    */
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

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
  }

  void createFrameBuffers()
  {
    //Create image and image view for the depth buffer
    if (depthImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, depthImageView, nullptr);

      depthImageView = VK_NULL_HANDLE;
    }
    if (depthImageMemoryObject != VK_NULL_HANDLE) {

      vkFreeMemory(m_device, depthImageMemoryObject, nullptr);

      depthImageMemoryObject = VK_NULL_HANDLE;

    }
    if (depthImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, depthImage, nullptr);

      depthImage = VK_NULL_HANDLE;
    }
    create2DImageAndView(depthFormat, m_swapChainExtent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depthImage, depthImageMemoryObject, depthImageView);

    //Create framebuffer
    m_swapChainFramebuffers.resize(m_swapChainImageView.size());

    for (size_t i = 0; i < m_swapChainImageView.size(); i++) {
      createFramebuffer({ m_swapChainImageView[i], depthImageView }, m_swapChainExtent, 1, m_swapChainFramebuffers[i]);
    }
  }

  void createDescriptorSets(std::vector<Buffer> buffers, std::vector<Image> images)
  {
    allocateDescriptorSets();
    std::vector<VkWriteDescriptorSet> descriptorWrites = {};
    std::vector<VkDescriptorBufferInfo> bufferInfo(buffers.size());
    std::vector<VkDescriptorImageInfo> imageInfo(images.size());

    uint32_t index = 0;

    for (auto buffer : buffers) {
      bufferInfo[index].buffer = buffer.buffer;
      bufferInfo[index].offset = 0;
      bufferInfo[index].range = buffer.size;

      VkWriteDescriptorSet descriptorWrite = {};
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = m_descriptorSet;
      descriptorWrite.dstBinding = buffer.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = &bufferInfo[index++];
      descriptorWrites.push_back(descriptorWrite);
    }

    index = 0;
    for (auto image : images) {
      imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo[index].imageView = image.view;
      imageInfo[index].sampler = image.sampler;

      VkWriteDescriptorSet descriptorWrite = {};
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = m_descriptorSet;
      descriptorWrite.dstBinding = image.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pImageInfo = &imageInfo[index++];
      descriptorWrites.push_back(descriptorWrite);
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }

  void recordCommandBuffers(VkTools::Mesh &model)
  {
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

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pInheritanceInfo = nullptr;

    for (size_t i = 0; i < m_commandBuffers.size(); i++) {

      //Set target frame buffer
      renderPassBeginInfo.framebuffer = m_swapChainFramebuffers[i];

      //Start the render pass
      vkBeginCommandBuffer(m_commandBuffers[i], &beginInfo);

      //Render pass
      vkCmdBeginRenderPass(m_commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

      VkBuffer vertexBuffers[] = { vertexBuffer };
      VkDeviceSize offsets[] = { 0 };
      vkCmdBindVertexBuffers(m_commandBuffers[i], 0, 1, vertexBuffers, offsets);
      vkCmdBindDescriptorSets(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

      for (size_t j = 0; j < model.Parts.size(); ++j) {
        vkCmdDraw(m_commandBuffers[i], model.Parts[j].VertexCount, 1, model.Parts[j].VertexOffset, 0);
      }

      vkCmdEndRenderPass(m_commandBuffers[i]);

      //End render pass
      if (vkEndCommandBuffer(m_commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }
};
