#version 420

#define MAX_BONES 64

layout (std140, set = 1, binding = 0) uniform Material {
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
    vec4 auxilary;
    mat4 bones[MAX_BONES];
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;
layout (location = 5) in vec4 weight;
layout (location = 6) in ivec4 ID;

layout (location = 0) out vec3 N;
layout (location = 1) out vec3 L;
layout (location = 2) out vec3 V;
layout (location = 3) out vec2 TexCoord;
layout (location = 4) out vec3 T;
layout (location = 5) out vec3 B;

layout (std140, set = 0, binding = 0) uniform Matrices
{
    mat4 modelview_matrix;
    mat4 modelviewproj_matrix;
    vec4 light_pos;
};

void main()
{

  mat4 boneTransform = bones[ID[0]] * weight[0];
  boneTransform     += bones[ID[1]] * weight[1];
  boneTransform     += bones[ID[2]] * weight[2];
  boneTransform     += bones[ID[3]] * weight[3];

  vec4 P = modelview_matrix * vec4(position, 1.0);
  mat3 transformationMatrix = mat3(inverse(transpose(modelview_matrix * boneTransform)));
  N = transformationMatrix * normal;

  L = (light_pos - P).xyz;
  V = -P.xyz;
  TexCoord = texCoord;
  T = transformationMatrix * tangent;
  B = transformationMatrix * bitangent;
  gl_Position = modelviewproj_matrix * boneTransform * vec4(position, 1.0);
}
