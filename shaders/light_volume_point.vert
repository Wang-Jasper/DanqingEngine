// light_volume_point.vert — instanced point-light volume proxy. gl_InstanceIndex
// indexes PointLightSSBO; the icosphere unit-sphere mesh is scaled to lightRadius
// and translated to lightPos; the index is forwarded to the fragment shader.
#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PC {
    mat4 viewProj;
} pc;

struct PointLight {
    vec3  position;
    float radius;
    vec3  color;
    float intensity;
    int   shadowSlot;
    float _pad[3];
};
layout(std430, set = 0, binding = 4) readonly buffer PointLightSSBO {
    PointLight pointLights[];
};

layout(location = 0) out flat int vInstanceID;

void main() {
    vec3 lightPos = pointLights[gl_InstanceIndex].position;
    float r       = pointLights[gl_InstanceIndex].radius;
    vec3 worldPos = lightPos + inPosition * r;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
    vInstanceID = gl_InstanceIndex;
}
