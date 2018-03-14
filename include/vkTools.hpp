#pragma once

#include "config.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <fstream>
#include <vector>
#include <array>
#include <iostream>

#include <memory>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace VkTools {
  std::vector<char> readFile(const std::string& filename)
  {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (file.fail()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
  }

  uint32_t findMemoryType(uint32_t typeFilter, VkPhysicalDeviceMemoryProperties memProperties, VkMemoryPropertyFlags properties)
  {
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
	return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  bool hasStencilComponent(VkFormat format) {

    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT;

  }

  struct Mesh {

    std::vector<float>  Data;
    struct Part {
      uint32_t  VertexOffset;
      uint32_t  VertexCount;
    };
    std::vector<Part> Parts;

    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    uint32_t stride;
  };

  // Based on:
  // Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh". Terathon Software 3D Graphics Library, 2001.
  // http://www.terathon.com/code/tangent.html

  void calculateTangentAndBitangent(float const * normal_data, glm::vec3 const & face_tangent, glm::vec3 const & face_bitangent, float * tangent_data, float * bitangent_data)
  {
    // Gram-Schmidt orthogonalize
    glm::vec3 const normal = { normal_data[0], normal_data[1], normal_data[2] };
    glm::vec3 const tangent = glm::normalize(face_tangent - normal * glm::dot(normal, face_tangent));

    // Calculate handedness
    float handedness = (glm::dot(glm::cross(normal, tangent), face_bitangent) < 0.0f) ? -1.0f : 1.0f;

    glm::vec3 const bitangent = handedness * glm::cross(normal, tangent);

    tangent_data[0] = tangent[0];
    tangent_data[1] = tangent[1];
    tangent_data[2] = tangent[2];

    bitangent_data[0] = bitangent[0];
    bitangent_data[1] = bitangent[1];
    bitangent_data[2] = bitangent[2];
  }

  void generateTangentSpaceVectors(Mesh & mesh)
  {
    size_t const normal_offset = 3;
    size_t const texcoord_offset = 6;
    size_t const tangent_offset = 8;
    size_t const bitangent_offset = 11;
    size_t const stride = bitangent_offset + 3;
    
    for (auto & part : mesh.Parts) {
      _unused(part);
      for (size_t i = 0; i < mesh.Data.size(); i += stride * 3) {
	size_t i1 = i;
	size_t i2 = i1 + stride;
	size_t i3 = i2 + stride;

	glm::vec3 const v1 = { mesh.Data[i1], mesh.Data[i1 + 1], mesh.Data[i1 + 2] };
	glm::vec3 const v2 = { mesh.Data[i2], mesh.Data[i2 + 1], mesh.Data[i2 + 2] };
	glm::vec3 const v3 = { mesh.Data[i3], mesh.Data[i3 + 1], mesh.Data[i3 + 2] };

	std::array<float, 2> const w1 = { {mesh.Data[i1 + texcoord_offset], mesh.Data[i1 + texcoord_offset + 1]} };
	std::array<float, 2> const w2 = { {mesh.Data[i2 + texcoord_offset], mesh.Data[i2 + texcoord_offset + 1]} };
	std::array<float, 2> const w3 = { {mesh.Data[i3 + texcoord_offset], mesh.Data[i3 + texcoord_offset + 1]} };

	float x1 = v2[0] - v1[0];
	float x2 = v3[0] - v1[0];
	float y1 = v2[1] - v1[1];
	float y2 = v3[1] - v1[1];
	float z1 = v2[2] - v1[2];
	float z2 = v3[2] - v1[2];

	float s1 = w2[0] - w1[0];
	float s2 = w3[0] - w1[0];
	float t1 = w2[1] - w1[1];
	float t2 = w3[1] - w1[1];

	float r = 1.0f / (s1 * t2 - s2 * t1);

	glm::vec3 face_tangent = { (t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r };
	glm::vec3 face_bitangent = { (s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r };

	calculateTangentAndBitangent(&mesh.Data[i1 + normal_offset], face_tangent, face_bitangent, &mesh.Data[i1 + tangent_offset], &mesh.Data[i1 + bitangent_offset]);

	calculateTangentAndBitangent(&mesh.Data[i2 + normal_offset], face_tangent, face_bitangent, &mesh.Data[i2 + tangent_offset], &mesh.Data[i2 + bitangent_offset]);

	calculateTangentAndBitangent(&mesh.Data[i3 + normal_offset], face_tangent, face_bitangent, &mesh.Data[i3 + tangent_offset], &mesh.Data[i3 + bitangent_offset]);
      }
    }
  }
 
  void load3DModelFromObjFile(char const * filename, Mesh & mesh, bool computeTB = false)
  {
    // Load model
    tinyobj::attrib_t                attribs;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      error;

    if (!tinyobj::LoadObj(&attribs, &shapes, &materials, &error, filename)) {
      throw std::runtime_error("could not open object file!");
    }
    if (!error.empty()) {
      std::cerr << error << std::endl;
    }

#ifdef VK_DEBUG
    std::cout << "# of vertices  : " << (attribs.vertices.size() / 3) << std::endl;
    std::cout << "# of normals   : " << (attribs.normals.size() / 3) << std::endl;
    std::cout << "# of texcoords : " << (attribs.texcoords.size() / 2) << std::endl;
    std::cout << "# of shapes    : " << shapes.size() << std::endl;
    std::cout << "# of materials : " << materials.size() << std::endl;
#endif

    computeTB = computeTB && attribs.normals.size() * attribs.texcoords.size() > 0;

    mesh = {};
    mesh.min_x = attribs.vertices[0];
    mesh.max_x = attribs.vertices[0];
    mesh.min_y = attribs.vertices[1];
    mesh.max_y = attribs.vertices[1];
    mesh.min_z = attribs.vertices[2];
    mesh.max_z = attribs.vertices[2];

    uint32_t offset = 0;

    //Loop over shapes
    for (auto & shape : shapes) {

      //Loop over indices
      uint32_t part_offset = offset;

      for (auto & index : shape.mesh.indices) {

	//Load vertices
	mesh.Data.emplace_back(attribs.vertices[3 * index.vertex_index + 0]);
	mesh.Data.emplace_back(attribs.vertices[3 * index.vertex_index + 1]);
	mesh.Data.emplace_back(attribs.vertices[3 * index.vertex_index + 2]);

	//Find boundaries
	if (attribs.vertices[3 * index.vertex_index + 0] < mesh.min_x) {
	  mesh.min_x = attribs.vertices[3 * index.vertex_index + 0];
	}
	if (attribs.vertices[3 * index.vertex_index + 0] > mesh.max_x) {
	  mesh.max_x = attribs.vertices[3 * index.vertex_index + 0];
	}
	if (attribs.vertices[3 * index.vertex_index + 1] < mesh.min_y) {
	  mesh.min_y = attribs.vertices[3 * index.vertex_index + 1];
	}
	if (attribs.vertices[3 * index.vertex_index + 1] > mesh.max_y) {
	  mesh.max_y = attribs.vertices[3 * index.vertex_index + 1];
	}
	if (attribs.vertices[3 * index.vertex_index + 2] < mesh.min_z) {
	  mesh.min_z = attribs.vertices[3 * index.vertex_index + 2];
	}
	if (attribs.vertices[3 * index.vertex_index + 2] > mesh.max_z) {
	  mesh.max_z = attribs.vertices[3 * index.vertex_index + 2];
	}

	//Load normals
	if (attribs.normals.size() > 0) {
	  mesh.Data.emplace_back(attribs.normals[3 * index.normal_index + 0]);
	  mesh.Data.emplace_back(attribs.normals[3 * index.normal_index + 1]);
	  mesh.Data.emplace_back(attribs.normals[3 * index.normal_index + 2]);
	}

	//Load texcoords
	if (attribs.texcoords.size() > 0) {
	  mesh.Data.emplace_back(attribs.texcoords[2 * index.texcoord_index + 0]);
	  mesh.Data.emplace_back(attribs.texcoords[2 * index.texcoord_index + 1]);
	}

	// Insert temporary tangent space vectors data
	if (computeTB) {
	  for (int i = 0; i < 6; ++i) {
	    mesh.Data.emplace_back(0.0f);
	  }
	}

	++offset;
      }

      uint32_t part_vertex_count = offset - part_offset;
      
      if (part_vertex_count > 0) {
	mesh.Parts.push_back({ part_offset, part_vertex_count });
      }
    }

    if (computeTB) {
      generateTangentSpaceVectors(mesh);
    }

    mesh.stride = 3 + ((attribs.normals.size() > 0) ? 3 : 0) + (attribs.texcoords.size() ? 2 : 0) + (computeTB ? 6 : 0);
  }


  void loadTextureDataFromFile(char const * filename, int num_requested_components, std::vector<unsigned char> & image_data, int * image_width, int * image_height, int * image_num_components, int * image_data_size)
  {
    int width = 0;
    int height = 0;
    int num_components = 0;

    std::unique_ptr<unsigned char, void(*)(void*)> stbi_data(stbi_load(filename, &width, &height, &num_components, num_requested_components), stbi_image_free);

    if ((!stbi_data) ||
	(0 >= width) ||
	(0 >= height) ||
	(0 >= num_components)) {
      throw std::runtime_error("could not read image!");
    }

    int data_size = width * height * (0 < num_requested_components ? num_requested_components : num_components);

    if (image_data_size) {
      *image_data_size = data_size;
    }

    if (image_width) {
      *image_width = width;
    }

    if (image_height) {
      *image_height = height;
    }

    if (image_num_components) {
      *image_num_components = num_components;
    }

    image_data.resize(data_size);
    std::memcpy(image_data.data(), stbi_data.get(), data_size);
  }
}
