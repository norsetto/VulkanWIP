#pragma once

#include "vkBase.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <map>
#include <string>
#include <cfloat>
#include <assert.h>
#include <vector>

#include <assimp/Importer.hpp>                              // C++ importer interface
#include <assimp/scene.h>                                   // Output data structure
#include <assimp/postprocess.h>                             // Post processing flags

class Model {

public:

    //For the time being, we only do this
	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec3 tangent;
		glm::vec3 bitangent;
    };
    
    Model(VkBase *vkBase) : vkBase(vkBase) {};
    ~Model(void)
    {
        VkDevice device = vkBase->device();
        for (auto textures :  {textureDiffuseMaps, textureNormalMaps, textureSpecularMaps})
            for (auto texture : textures)
            {
                if (texture.second.sampler !=  VK_NULL_HANDLE) {
                    vkDestroySampler(device, texture.second.sampler, nullptr);
                    texture.second.sampler = VK_NULL_HANDLE;
                }
                if (texture.second.view !=  VK_NULL_HANDLE) {
                    vkDestroyImageView(device, texture.second.view, nullptr);
                    texture.second.view = VK_NULL_HANDLE;
                }
                if (texture.second.image !=  VK_NULL_HANDLE) {
                    vkDestroyImage(device, texture.second.image, nullptr);
                    texture.second.image = VK_NULL_HANDLE;
                }
                if (texture.second.imageMemory !=  VK_NULL_HANDLE) {
                    vkFreeMemory(device, texture.second.imageMemory, nullptr);
                    texture.second.imageMemory = VK_NULL_HANDLE;
                }
            }
        for (auto buffers : { vertexBuffer, indexBuffer, materialBuffer, stagingMaterialBuffer } )
            for (auto buffer :  buffers)
            {
                if (buffer !=  VK_NULL_HANDLE) {
                    vkDestroyBuffer(device, buffer, nullptr);
                    buffer = VK_NULL_HANDLE;
                }
            }
        for (auto memories : { vertexBufferMemory, indexBufferMemory, materialBufferMemory, stagingMaterialBufferMemory } )
            for (auto memory :  memories)
            {
                if (memory !=  VK_NULL_HANDLE) {
                    vkFreeMemory(device, memory, nullptr);
                    memory = VK_NULL_HANDLE;
                }
            }
    };
    
    void load(const std::string &filename, bool compute_bounding_box = true, bool compute_tangent_bitangent = false);

    //Getters
    uint32_t getNumMeshes(void) { return num_meshes; };
    float scale(void) { return bb->scale; };
    glm::vec3 min(void) { return bb->min; };
    glm::vec3 max(void) { return bb->max; };
    VkBase::Buffer getMaterialBuffers(uint32_t index) { return materialBuffers[index]; };
    VkBase::Texture getDiffuse(uint32_t index) { return diffuse[index]; };
    VkBase::Texture getNormals(uint32_t index) { return normals[index]; };
    VkBase::Texture getSpeculars(uint32_t index) { return speculars[index]; };
    VkBuffer* getVertexBufferPointer(uint32_t index) { return &vertexBuffer[index]; };
    VkBuffer getIndexBuffer(uint32_t index) { return indexBuffer[index]; };
    uint32_t getNumIndices(uint32_t index) { return num_indices[index]; };

private:

    void findUniqueTextures(const aiScene *scene);
    
    //Base Class
    VkBase *vkBase;
    
    //Vertex and Index buffers and memories
    std::vector<VkBuffer> vertexBuffer, indexBuffer;
    std::vector<VkDeviceMemory> vertexBufferMemory, indexBufferMemory;

    //Textures
    std::map<std::string, VkBase::Texture> textureDiffuseMaps;	
    std::map<std::string, VkBase::Texture> textureNormalMaps;	
    std::map<std::string, VkBase::Texture> textureSpecularMaps;	
    std::vector<VkBase::Texture> diffuse;
    std::vector<VkBase::Texture> normals;
    std::vector<VkBase::Texture> speculars;
    
    //Number of meshes
    uint32_t num_meshes = 0;
    
    //Indices buffer
    uint32_t *num_indices = nullptr;
    std::vector<uint32_t> *indices = nullptr;

    //Material properties
    struct MATERIALS_BLOCK {
        glm::vec4 diffuse;
        glm::vec4 ambient;
        glm::vec4 specular;
        glm::vec4 auxilary;
      };
    std::vector<MATERIALS_BLOCK> materials;
    std::vector<VkBuffer> materialBuffer, stagingMaterialBuffer;
    std::vector<VkDeviceMemory> materialBufferMemory, stagingMaterialBufferMemory;
    std::vector<VkBase::Buffer> materialBuffers;
/*
    // Descriptors
    std::vector<VkVertexInputAttributeDescription> *attributeDescriptions;
    VkVertexInputBindingDescription *bindingDescriptions;
    VkPipelineVertexInputStateCreateInfo *vertexInputInfo;
*/
    //Bounding box
    struct BOUNDING_BOX {
        glm::vec3 min;
        glm::vec3 max;
        float scale;
      }* bb = nullptr;

};

void Model::load(const std::string &filename, bool compute_bounding_box, bool compute_tangent_bitangent) {
    Assimp::Importer importer;
    unsigned int aiFlags = aiProcess_GenSmoothNormals |
    aiProcess_JoinIdenticalVertices    |
    aiProcess_ImproveCacheLocality     |
    aiProcess_RemoveRedundantMaterials |
    aiProcess_Triangulate              |
    aiProcess_GenUVCoords              |
    aiProcess_SortByPType              |
    aiProcess_FindDegenerates          |
    aiProcess_FindInvalidData          |
    aiProcess_FlipUVs                  | 
    aiProcess_ValidateDataStructure;

    if (compute_tangent_bitangent) aiFlags = aiFlags | aiProcess_CalcTangentSpace;

    const aiScene* scene = importer.ReadFile(filename, aiFlags);

    assert(scene);
    assert(scene->HasMeshes());

    //Find unique textures
    findUniqueTextures(scene);

    //Load textures
    for (std::map<std::string, VkBase::Texture>::iterator it = textureDiffuseMaps.begin(); it != textureDiffuseMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first);
    for (std::map<std::string, VkBase::Texture>::iterator it = textureNormalMaps.begin(); it != textureNormalMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first);
    for (std::map<std::string, VkBase::Texture>::iterator it = textureSpecularMaps.begin(); it != textureSpecularMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first);
      
    if (compute_bounding_box) {
        bb = new BOUNDING_BOX;
        bb->min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        bb->max = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
        bb->scale = 1.0f;
      }
      
    num_meshes = scene->mNumMeshes;
    
    materials.resize(num_meshes);
    stagingMaterialBuffer.resize(num_meshes);
    materialBuffer.resize(num_meshes);
    stagingMaterialBufferMemory.resize(num_meshes);
    materialBufferMemory.resize(num_meshes);
    materialBuffers.resize(num_meshes);
    
    diffuse.resize(num_meshes);
    normals.resize(num_meshes);
    speculars.resize(num_meshes);
    /*
    attributeDescriptions = new std::vector<VkVertexInputAttributeDescription> [num_meshes];
    bindingDescriptions = new VkVertexInputBindingDescription [num_meshes];
    vertexInputInfo = new VkPipelineVertexInputStateCreateInfo [num_meshes];
    */
    vertexBuffer.resize(num_meshes);
    vertexBufferMemory.resize(num_meshes);
    indexBuffer.resize(num_meshes);
    indexBufferMemory.resize(num_meshes);
    indices = new std::vector<uint32_t> [num_meshes];
    num_indices = new uint32_t [num_meshes];

    for (uint32_t i = 0; i < num_meshes; i++) {

        const aiMesh* mesh = scene->mMeshes[i];

        //Set indices buffer
        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            for (uint32_t k = 0; k < 3; k++) {
                indices[i].push_back(mesh->mFaces[j].mIndices[k]);
            }
        }
        num_indices[i] = static_cast<uint32_t>(indices[i].size());
        
        //Set attribute descriptors
/*        uint32_t stride = 0;
        
        if (mesh->HasPositions()) {
            VkVertexInputAttributeDescription attributeDescription = {};
            attributeDescription.binding = 0;
            attributeDescription.location = 0;
            attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescription.offset = offsetof(Vertex, position);
            attributeDescriptions[i].push_back(attributeDescription);
        
            stride += 3;
          }

        if (mesh->HasNormals()) {
            VkVertexInputAttributeDescription attributeDescription = {};
            attributeDescription.binding = 0;
            attributeDescription.location = 1;
            attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescription.offset = offsetof(Vertex, normal);
            attributeDescriptions[i].push_back(attributeDescription);
            
            stride += 3;
          }

        if (mesh->HasTextureCoords(0)) {
            VkVertexInputAttributeDescription attributeDescription = {};
            attributeDescription.binding = 0;
            attributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
            if (mesh->HasNormals()) {
                attributeDescription.location = 2;
            }
            else {
                attributeDescription.location = 1;
            }
            attributeDescription.offset = offsetof(Vertex, uv);
            attributeDescriptions[i].push_back(attributeDescription);
            
            stride += 2;
          }

        if (mesh->HasTangentsAndBitangents()) {
            VkVertexInputAttributeDescription attributeDescription = {};
                attributeDescription.binding = 0;
                attributeDescription.location = 3;
                attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescription.offset = offsetof(Vertex, tangent);
                attributeDescriptions[i].push_back(attributeDescription);
                attributeDescription.binding = 0;
                attributeDescription.location = 4;
                attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescription.offset = offsetof(Vertex, bitangent);
                attributeDescriptions[i].push_back(attributeDescription);
                
                stride += 6;
          }

        assert(stride == 14);
        bindingDescriptions[i].binding = 0;
        bindingDescriptions[i].stride = stride * sizeof(float);
        bindingDescriptions[i].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo[i] = {};
        vertexInputInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo[i].vertexBindingDescriptionCount = 1;
        vertexInputInfo[i].pVertexBindingDescriptions = &bindingDescriptions[i];
        vertexInputInfo[i].vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions[i].size());
        vertexInputInfo[i].pVertexAttributeDescriptions = attributeDescriptions[i].data();
*/
        //Set Vertex Buffers
        std::vector<Vertex> vertices;
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            
            Vertex vertex;
            vertex.position  = glm::make_vec3(&mesh->mVertices[j].x);
            vertex.normal    = glm::make_vec3(&mesh->mNormals[j].x);
            vertex.uv        = glm::make_vec2(&mesh->mTextureCoords[0][j].x);
            vertex.tangent   = glm::make_vec3(&mesh->mTangents[j].x);
            vertex.bitangent = glm::make_vec3(&mesh->mBitangents[j].x);
        
            vertices.push_back(vertex);
        
            if (compute_bounding_box) {
                bb->min.x = vk_min(bb->min.x, mesh->mVertices[j].x);
                bb->min.y = vk_min(bb->min.y, mesh->mVertices[j].y);
                bb->min.z = vk_min(bb->min.z, mesh->mVertices[j].z);

                bb->max.x = vk_max(bb->max.x, mesh->mVertices[j].x);
                bb->max.y = vk_max(bb->max.y, mesh->mVertices[j].y);
                bb->max.z = vk_max(bb->max.z, mesh->mVertices[j].z);
            }
        }
        
        vkBase->createDataBuffer(vertices, vertexBuffer[i], vertexBufferMemory[i], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        vkBase->createDataBuffer(indices[i], indexBuffer[i], indexBufferMemory[i], VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        //Process materials
        aiMaterial *mtl = scene->mMaterials[mesh->mMaterialIndex];

        aiString texPath;
        if(AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath))
        {
            materials[i].auxilary[1] = 1.0f;
            diffuse[i] = textureDiffuseMaps[texPath.data];
            if(AI_SUCCESS == mtl->GetTexture(aiTextureType_NORMALS, 0, &texPath))
            {
                materials[i].auxilary[1] = 2.0f;
                normals[i] = textureNormalMaps[texPath.data];
                if(AI_SUCCESS == mtl->GetTexture(aiTextureType_SPECULAR, 0, &texPath))
                {
                    materials[i].auxilary[1] = 3.0f;
                    speculars[i] = textureSpecularMaps[texPath.data];
                }
            } else
            {
                normals[i] = {};
                if(AI_SUCCESS == mtl->GetTexture(aiTextureType_SPECULAR, 0, &texPath))
                {
                    materials[i].auxilary[1] = 4.0f;
                    speculars[i] = textureSpecularMaps[texPath.data];
                }
            }
        }
        else
        {
            materials[i].auxilary[1] = 0.0f;
        }
        diffuse[i].binding = 1;
        normals[i].binding = 2;
        speculars[i].binding = 3;

        aiColor4D diffuseColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor))
            materials[i].diffuse = glm::vec4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);
        else
            materials[i].diffuse = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);

        aiColor4D ambientColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambientColor))
            materials[i].ambient = glm::vec4(ambientColor.r, ambientColor.g, ambientColor.b, ambientColor.a);
        else
            materials[i].ambient = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);

        aiColor4D specularColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specularColor))
            materials[i].specular = glm::vec4(specularColor.r, specularColor.g, specularColor.b, specularColor.a);
        else
            materials[i].specular = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        
        float shininess = 0.0;
        unsigned int max;
        aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
        materials[i].auxilary[0] = shininess;
        
        std::vector<MATERIALS_BLOCK> materialsData = { materials[i] };
        vkBase->createDataDoubleBuffer(materialsData, stagingMaterialBuffer[i], materialBuffer[i], stagingMaterialBufferMemory[i], materialBufferMemory[i], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        vkBase->copyDataToBuffer(stagingMaterialBufferMemory[i], &materials[i], sizeof(MATERIALS_BLOCK));
        vkBase->copyBufferToBuffer(stagingMaterialBuffer[i], materialBuffer[i], sizeof(MATERIALS_BLOCK));

        materialBuffers[i] = {materialBuffer[i], sizeof(MATERIALS_BLOCK), 0};
      }

    if (compute_bounding_box) {
        float tmp = bb->max.x - bb->min.x;
        tmp = bb->max.y - bb->min.y > tmp?bb->max.y - bb->min.y:tmp;
        tmp = bb->max.z - bb->min.z > tmp?bb->max.z - bb->min.z:tmp;
        bb->scale = 1.f / tmp;
      }
}

void Model::findUniqueTextures(const aiScene *scene)
{
    for (unsigned int i = 0; i < scene->mNumMaterials; i++) {
        int texIndex = 0;
        aiString path;

        aiReturn texFound = scene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
        while (texFound == AI_SUCCESS) {
            textureDiffuseMaps[path.data] = {};
            texIndex++;
            texFound = scene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
          }

        texIndex = 0;

        texFound = scene->mMaterials[i]->GetTexture(aiTextureType_NORMALS, texIndex, &path);
        while (texFound == AI_SUCCESS) {
            textureNormalMaps[path.data] = {};
            texIndex++;
            texFound = scene->mMaterials[i]->GetTexture(aiTextureType_NORMALS, texIndex, &path);
          }

        texIndex = 0;

        texFound = scene->mMaterials[i]->GetTexture(aiTextureType_SPECULAR, texIndex, &path);
        while (texFound == AI_SUCCESS) {
            textureSpecularMaps[path.data] = {};
            texIndex++;
            texFound = scene->mMaterials[i]->GetTexture(aiTextureType_SPECULAR, texIndex, &path);
          }
      }
}