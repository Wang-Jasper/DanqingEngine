// shadow_point.vert — point-light cubemap shadow pass. Renders 6 faces with
// perspective depth-only projection; each face selects cubeVP[faceIndex] via
// push constant.
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // unused
layout(location = 2) in vec3 inColor;    // unused
layout(location = 3) in vec2 inTexCoord; // unused

layout(set = 0, binding = 0) uniform PointShadowVPUBO {
    mat4 cubeVP[6];
    vec4 lightPosRange; // xyz = lightPos, w = range
} ubo;

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
    uint faceIndex;
} pc;

layout(location = 0) out vec3 vWorldPos;

void main() {
    InstanceData inst = instanceBuf.instances[pc.instanceOffset + gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = ubo.cubeVP[pc.faceIndex] * worldPos;
}
