/*
 * The bone and animation stuff is mainly taken from:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/skeletalanimation/skeletalanimation.cpp
 */
#pragma once

#include "vkBase.hpp"
#include "config.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifdef VK_DEBUG
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#endif

#include <map>
#include <string>
#include <cfloat>
#include <cstring>
#include <assert.h>
#include <vector>
#ifdef VK_DEBUG
#include <queue>
#endif
#include <assimp/Importer.hpp>                              // C++ importer interface
#include <assimp/scene.h>                                   // Output data structure
#include <assimp/postprocess.h>                             // Post processing flags

class Model {

public:

    // Maximum number of bones per mesh
    // Must not be higher than same const in skinning shader
    #define MAX_BONES_PER_MESH 64

    // Maximum number of bones per vertex
    #define MAX_BONES_PER_VERTEX 4

    //For the time being, we only do this
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        glm::vec3 bitangent;
        float boneWeights[MAX_BONES_PER_VERTEX];
        uint32_t boneIDs[MAX_BONES_PER_VERTEX];
    };
    
    struct VertexBoneData {
        std::array<uint32_t, MAX_BONES_PER_VERTEX> IDs{};
        std::array<float, MAX_BONES_PER_VERTEX> weights{};

        //Add bone weighting to vertex info
        void add(uint32_t boneID, float weight)
        {
            for (uint32_t i = 0; i < MAX_BONES_PER_VERTEX; i++)
            {
                //We only take the bigger weights into consideration
                if (weight > weights[i])
                {
                    std::swap(IDs[i], boneID);
                    std::swap(weights[i], weight);
                }
            }
        }
    };

    Model(VkBase *vkBase) : vkBase(vkBase) {};
    ~Model(void)
    {
        VkDevice device = vkBase->device();
        for (auto textures :  {textureDiffuseMaps, textureNormalMaps, textureSpecularMaps})
            for (auto texture : textures)
            {
                texture.second.destroy(vkBase->device());
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
    
    void load(const std::string &filename, bool compute_bounding_box, bool compute_tangent_bitangent, bool anisotropy_enable, float max_anisotropy, bool load_bones = false);

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
    void loadBones(const aiScene *scene);
    void transformBoneOffsets(const aiScene *scene, aiNode* pNode, const aiMatrix4x4& ParentTransform);
#ifdef VK_DEBUG
    void printLevelOrder(aiNode* pNode);
# endif
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
        glm::mat4 bones[MAX_BONES_PER_MESH];
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

    //Per-mesh/per-vertex bone info (boneID, weight)
    std::vector<VertexBoneData>* bones;

    //Per-mesh/per-bone bone offsets
    std::vector<aiMatrix4x4>* boneOffset;

    //Per-mesh/per-bone transformed bone offsets
    std::vector<aiMatrix4x4>* transformedBoneOffset;

    //Root inverse transform matrix
    aiMatrix4x4 globalInverseTransform;
};

void Model::load(const std::string &filename, bool compute_bounding_box, bool compute_tangent_bitangent, bool anisotropy_enable, float max_anisotropy, bool load_bones) {
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
    num_meshes = scene->mNumMeshes;

    //Find unique textures
    findUniqueTextures(scene);

    //Load bones
    bool has_bones = false;
    for (uint32_t m = 0; m < num_meshes && has_bones ==  false; m++) {
        if (scene->mMeshes[m]->HasBones())
            has_bones = true;
    };
    load_bones = load_bones && has_bones;
    if (load_bones) {
        //Store global inverse transform matrix of root node
		globalInverseTransform = scene->mRootNode->mTransformation;
        globalInverseTransform.Inverse();

        //Setup bones
        loadBones(scene);

        //Fill out initial transformations
        aiMatrix4x4 identity = aiMatrix4x4();
        transformBoneOffsets(scene, scene->mRootNode, identity);
#ifdef VK_DEBUG
        printLevelOrder(scene->mRootNode);
# endif
    }

    //Load textures
    for (std::map<std::string, VkBase::Texture>::iterator it = textureDiffuseMaps.begin(); it != textureDiffuseMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first, anisotropy_enable, max_anisotropy);
    for (std::map<std::string, VkBase::Texture>::iterator it = textureNormalMaps.begin(); it != textureNormalMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first, anisotropy_enable, max_anisotropy);
    for (std::map<std::string, VkBase::Texture>::iterator it = textureSpecularMaps.begin(); it != textureSpecularMaps.end(); it++)
        vkBase->loadTexture(it->second, it->first, anisotropy_enable, max_anisotropy);
      
    if (compute_bounding_box) {
        bb = new BOUNDING_BOX;
        bb->min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        bb->max = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
        bb->scale = 1.0f;
      }
      
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

    for (uint32_t mesh = 0; mesh < num_meshes; mesh++) {

        const aiMesh* pMesh = scene->mMeshes[mesh];
        std::cout <<  "Loading " <<  pMesh->mName.C_Str() << " for " << pMesh->mNumVertices <<  " vertices." << std::endl;

        //Set indices buffer
        for (uint32_t j = 0; j < pMesh->mNumFaces; j++) {
            for (uint32_t k = 0; k < 3; k++) {
                indices[mesh].push_back(pMesh->mFaces[j].mIndices[k]);
            }
        }
        num_indices[mesh] = static_cast<uint32_t>(indices[mesh].size());
        
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
        for (unsigned int v = 0; v < pMesh->mNumVertices; v++) {
            
            Vertex vertex;
            vertex.position  = glm::make_vec3(&pMesh->mVertices[v].x);
            vertex.normal    = glm::make_vec3(&pMesh->mNormals[v].x);
            vertex.uv        = glm::make_vec2(&pMesh->mTextureCoords[0][v].x);
            vertex.tangent   = glm::make_vec3(&pMesh->mTangents[v].x);
            vertex.bitangent = glm::make_vec3(&pMesh->mBitangents[v].x);

            //Fetch bone weights and IDs
            if (load_bones) {
                for (uint32_t k = 0; k < MAX_BONES_PER_VERTEX; k++) {
                    vertex.boneWeights[k] = bones[mesh].at(v).weights[k];
                    vertex.boneIDs[k] = bones[mesh].at(v).IDs[k];
                }
            }

            vertices.push_back(vertex);
        
            if (compute_bounding_box) {
                bb->min.x = vk_min(bb->min.x, pMesh->mVertices[v].x);
                bb->min.y = vk_min(bb->min.y, pMesh->mVertices[v].y);
                bb->min.z = vk_min(bb->min.z, pMesh->mVertices[v].z);

                bb->max.x = vk_max(bb->max.x, pMesh->mVertices[v].x);
                bb->max.y = vk_max(bb->max.y, pMesh->mVertices[v].y);
                bb->max.z = vk_max(bb->max.z, pMesh->mVertices[v].z);
            }
        }
        
        vkBase->createDataBuffer(vertices, vertexBuffer[mesh], vertexBufferMemory[mesh], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        vkBase->createDataBuffer(indices[mesh], indexBuffer[mesh], indexBufferMemory[mesh], VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        //Process materials
        aiMaterial *mtl = scene->mMaterials[pMesh->mMaterialIndex];

        //Process textures
        aiString texPath;
        if(AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath))
        {
            materials[mesh].auxilary[1] = 1.0f;
            diffuse[mesh] = textureDiffuseMaps[texPath.data];
            std::cout << "With diffuse texture " << texPath.C_Str() << std::endl;
            if(AI_SUCCESS == mtl->GetTexture(aiTextureType_NORMALS, 0, &texPath))
            {
                materials[mesh].auxilary[1] = 2.0f;
                normals[mesh] = textureNormalMaps[texPath.data];
                std::cout << "With normal texture " << texPath.C_Str() << std::endl;
                if(AI_SUCCESS == mtl->GetTexture(aiTextureType_SPECULAR, 0, &texPath))
                {
                    materials[mesh].auxilary[1] = 3.0f;
                    speculars[mesh] = textureSpecularMaps[texPath.data];
                    std::cout << "With specular texture " << texPath.C_Str() << std::endl;
                }
            } else
            {
                normals[mesh] = {};
                if(AI_SUCCESS == mtl->GetTexture(aiTextureType_SPECULAR, 0, &texPath))
                {
                    materials[mesh].auxilary[1] = 4.0f;
                    speculars[mesh] = textureSpecularMaps[texPath.data];
                    std::cout << "With specular texture " << texPath.C_Str() << std::endl;
                }
            }
        }
        else
        {
            materials[mesh].auxilary[1] = 0.0f;
        }
        diffuse[mesh].binding = 1;
        normals[mesh].binding = 2;
        speculars[mesh].binding = 3;

        //Process material data
        aiColor4D diffuseColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor))
            materials[mesh].diffuse = glm::vec4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);
        else
            materials[mesh].diffuse = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);

        aiColor4D ambientColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambientColor))
            materials[mesh].ambient = glm::vec4(ambientColor.r, ambientColor.g, ambientColor.b, ambientColor.a);
        else
            materials[mesh].ambient = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);

        aiColor4D specularColor;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specularColor))
            materials[mesh].specular = glm::vec4(specularColor.r, specularColor.g, specularColor.b, specularColor.a);
        else
            materials[mesh].specular = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        
        float shininess = 0.0;
        unsigned int max;
        aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
        materials[mesh].auxilary[0] = shininess;
        
        //Process bone data
        if (load_bones) {
            for (uint32_t bone = 0; bone < pMesh->mNumBones; bone++) {
                aiMatrix4x4 boneTransform = transformedBoneOffset[mesh][bone];
                materials[mesh].bones[bone] = glm::transpose(glm::make_mat4(&boneTransform.a1));
#ifdef VK_DEBUG
                std::cout << "[" << mesh << "," << bone << "]";
                std::cout << "\t" << glm::to_string(materials[mesh].bones[bone]) << std::endl;
                std::cout << "\t\t" << glm::to_string(glm::transpose(glm::make_mat4(&boneOffset[mesh][bone].a1))) << std::endl;
#endif
            }

#ifdef VK_DEBUG
        for (unsigned int v = 0; v < pMesh->mNumVertices; v++) {
                glm::mat4 matrix = glm::mat4(0.0f);
                for (uint32_t k = 0; k < MAX_BONES_PER_VERTEX; k++) {
                    matrix += materials[mesh].bones[bones[mesh].at(v).IDs[k]] * bones[mesh].at(v).weights[k];
                }
                std::cout << "[" << mesh << "," << v << "]";
                std::cout << "\t" << glm::to_string(matrix) << std::endl;
            }
#endif
        }

        //Finally pack everything into the Buffer struct and load it to the GPU
        std::vector<MATERIALS_BLOCK> materialsData = { materials[mesh] };
        vkBase->createDataDoubleBuffer(materialsData, stagingMaterialBuffer[mesh], materialBuffer[mesh], stagingMaterialBufferMemory[mesh], materialBufferMemory[mesh], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        vkBase->copyDataToBuffer(stagingMaterialBufferMemory[mesh], &materials[mesh], sizeof(MATERIALS_BLOCK));
        vkBase->copyBufferToBuffer(stagingMaterialBuffer[mesh], materialBuffer[mesh], sizeof(MATERIALS_BLOCK));

        materialBuffers[mesh] = {materialBuffer[mesh], sizeof(MATERIALS_BLOCK), 0};
    }

    std::cout <<  "Loaded " <<  num_meshes <<  " meshes." <<  std::endl;

    if (compute_bounding_box) {
        float tmp = bb->max.x - bb->min.x;
        tmp = bb->max.y - bb->min.y > tmp?bb->max.y - bb->min.y:tmp;
        tmp = bb->max.z - bb->min.z > tmp?bb->max.z - bb->min.z:tmp;
        bb->scale = 1.f / tmp;
    }
}

//Load bone information from ASSIMP mesh
void Model::loadBones(const aiScene *scene)
{
    bones = new std::vector<VertexBoneData> [num_meshes];
    boneOffset = new std::vector<aiMatrix4x4> [num_meshes];
    transformedBoneOffset = new std::vector<aiMatrix4x4> [num_meshes];

    for (uint32_t mesh = 0; mesh < num_meshes; mesh++) {
        aiMesh *pMesh = scene->mMeshes[mesh];
        const uint32_t numBones = pMesh->mNumBones;

        if (numBones > MAX_BONES_PER_MESH) {
            std::stringstream errorMessage;
            errorMessage << "too many bones [" << numBones <<  "] for mesh " << mesh;
            errorMessage << " (" << pMesh->mName.C_Str() << ")!";
            throw std::runtime_error(errorMessage.str().c_str());
        }

        //Bone details
        bones[mesh].resize(pMesh->mNumVertices);
        transformedBoneOffset[mesh].resize(numBones);
        for (uint32_t bone = 0; bone < numBones; bone++)
        {
            aiBone *pBone = pMesh->mBones[bone];
            boneOffset[mesh].push_back(pBone->mOffsetMatrix);

            for (uint32_t weight = 0; weight < pBone->mNumWeights; weight++)
            {
                uint32_t vertexID = pBone->mWeights[weight].mVertexId;
                bones[mesh].at(vertexID).add(bone, pBone->mWeights[weight].mWeight);
            }
        }
        std::cout << "Loaded " << numBones << " bones for mesh " << mesh;
        std::cout << " (" << pMesh->mName.C_Str() << ")" << std::endl;
    }
}

void Model::transformBoneOffsets(const aiScene* scene, aiNode* pNode, const aiMatrix4x4& ParentTransform)
{
    std::string nodeName(pNode->mName.data);
    aiMatrix4x4 NodeTransformation(pNode->mTransformation);
    aiMatrix4x4 GlobalTransformation = ParentTransform * NodeTransformation;

    //Compute transformation for current node
    for (uint32_t mesh = 0; mesh < num_meshes; mesh++) {
        const aiMesh* pMesh = scene->mMeshes[mesh];

        for (uint32_t bone = 0; bone < pMesh->mNumBones; bone++) {
            if (std::string(pMesh->mBones[bone]->mName.data) == nodeName) {
                transformedBoneOffset[mesh][bone] = globalInverseTransform * GlobalTransformation * boneOffset[mesh][bone];
#ifdef VK_DEBUG
                std::cout << "[" << mesh << "," << bone << "]" << std::endl;
#endif
                break;
            }
        }
    }

    //Recurse through all child nodes
    for (uint32_t i = 0; i < pNode->mNumChildren; i++) {
        transformBoneOffsets(scene, pNode->mChildren[i], GlobalTransformation);
    }
}

#ifdef VK_DEBUG
void Model::printLevelOrder(aiNode* pNode)
{
    //Create an empty queue for level order traversal
    std::queue<aiNode*> q;

    //Enqueue Root and initialize height
    q.push(pNode);

    while (1)
    {
        // nodeCount (queue size) indicates number of nodes
        // at current lelvel.
        int nodeCount = q.size();
        if (nodeCount == 0)
            break;

        // Dequeue all nodes of current level and Enqueue all
        // nodes of next level
        while (nodeCount > 0)
        {
            aiNode *node = q.front();
            std::cout << node->mName.C_Str() << " ";
            q.pop();
            for (uint32_t i = 0; i < node->mNumChildren; i++) {
                q.push(node->mChildren[i]);
            }
            nodeCount--;
        }
        std::cout << std::endl;
    }
}
#endif

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
