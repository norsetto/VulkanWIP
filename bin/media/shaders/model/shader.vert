#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBitangent;
layout(location = 5) out vec4 outLight;

layout(binding = 0) uniform UniformBufferObject {
	mat4 mv;
    mat4 mvp;
	vec4 light;
};

void main() {

	outPosition    = mat3(mv) * inPosition;
	outNormal      = mat3(mv) * inNormal;
	outTexCoord    = vec2(inTexCoord.x, 1.0 - inTexCoord.y);
	outTangent     = mat3(mv) * inTangent;
	outBitangent   = mat3(mv) * inBitangent;
	outLight       = light;

    gl_Position = mvp * vec4(inPosition, 1.0);
}