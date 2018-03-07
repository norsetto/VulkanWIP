#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec4 position;
layout(location = 1) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
};

void main() {

	position = vec4(inPosition, 1.0);
	gl_Position = mvp * position;
	fragColor = inColor;
}