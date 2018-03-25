#version 450

layout(binding = 1) uniform sampler2D diffuseSampler;
layout(binding = 2) uniform sampler2D normalSampler;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;
layout(location = 5) in vec4 inLight;

layout(location = 0) out vec4 outColor;

void main() {
	vec4 texel = texture(diffuseSampler, inTexCoord);
	vec3 diffuse_texture = texel.rgb;
	float specularity = texel.a;
	vec3 normal = 2.0 * texture(normalSampler, inTexCoord).rgb - 1.0;

	vec3 N = normalize(mat3(inTangent, inBitangent, inNormal) * normal);
	vec3 L = normalize(inLight.xyz);
	vec3 V = -normalize(inPosition);
	L += V;
	L = normalize(L);
	vec3 R = reflect(-L, N);

	float ambient = 0.2;
	float diffuse = max(0.0, dot(N, L));
	float specular = pow(max(dot(R, V), 0.0), 100.0);

	outColor = vec4(min(1.0, ambient + diffuse + specular * specularity) * diffuse_texture, 1.0 );
}
