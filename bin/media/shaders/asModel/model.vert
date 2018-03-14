#version 420

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

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
  vec4 P = modelview_matrix * vec4(position, 1.0);
  N = mat3(modelview_matrix) * normal;
  L = (light_pos - P).xyz;
  V = -P.xyz;
  TexCoord = texCoord;
  T = mat3(modelview_matrix) * tangent;
  B = mat3(modelview_matrix) * bitangent;
  gl_Position = modelviewproj_matrix * vec4(position, 1.0);
}
