#version 420
 
#define MAX_BONES 64

layout (std140, set = 1, binding = 0) uniform Material {
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
    vec4 auxilary;
    mat4 bones[MAX_BONES];
};

layout (set = 1, binding = 1) uniform sampler2D texUnit;
layout (set = 1, binding = 2) uniform sampler2D nrmUnit;
layout (set = 1, binding = 3) uniform sampler2D spcUnit;

layout (location = 0) in vec3 N;
layout (location = 1) in vec3 L;
layout (location = 2) in vec3 V;
layout (location = 3) in vec2 TexCoord;
layout (location = 4) in vec3 T;
layout (location = 5) in vec3 B;

layout (location = 0) out vec4 color;

void main()
{
    vec4 diffuse_color;
    vec4 specular_color;

    //Compute normal vector
    vec3 normal = 2.0 * texture(nrmUnit, TexCoord).rgb - 1.0;
	vec3 normDir = normalize(mat3(T, B, N) * normal);

    //Compute and normalize light, view and reflection vectors
    vec3 lightDir = normalize(L);
    vec3 viewDir  = normalize(V);
    vec3 reflDir  = normalize(reflect(-lightDir, normDir));

    //Lighting equations
    float NdotL = dot(normDir, lightDir);
    float RdotV = dot(reflDir, viewDir);
 
    diffuse_color = texture(texUnit, TexCoord);
    color.a = diffuse_color.a;
    diffuse_color *= max(0.0, NdotL);
    
    specular_color = texture(spcUnit, TexCoord);
    specular_color *= pow(max(0.0, RdotV), auxilary[0]);
    color.rgb = (specular_color * 0.35 + diffuse_color  + ambient).rgb;
}
