// shadow.vert — directional shadow pass. Transforms scene geometry into light
// space; Vulkan writes the depth attachment directly.
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // unused
layout(location = 2) in vec3 inColor;    // unused
layout(location = 3) in vec2 inTexCoord; // unused

layout(set = 0, binding = 0) uniform ShadowVPUBO {
    mat4 lightViewProj;
} shadow;

struct InstanceData {
    mat4 model;
    vec4 albedoAndMetallic;
    vec4 roughnessAndFlags;
};
layout(set = 0, binding = 1, std430) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceBuf;

layout(push_constant) uniform PushConstants {
    uint instanceOffset;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} pc;

void main() {
    InstanceData inst = instanceBuf.instances[pc.instanceOffset + gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = shadow.lightViewProj * worldPos;
}
