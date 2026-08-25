// geometry.vert — Geometry Pass vertex shader (PBR + instancing).
#version 450

// Vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

// UBO: view-projection matrix
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Per-instance data (model matrix + PBR material) lives in an SSBO. Each batch
// is dispatched with vkCmdDrawIndexed(..., instanceCount, ...); the vertex
// shader reads instanceOffset + gl_InstanceIndex to fetch its slot.
struct InstanceData {
    mat4 model;
    vec4 albedoAndMetallic;
    vec4 roughnessAndFlags;
};
layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceBuf;

// Push Constant: per-batch metadata (small, refreshed each draw call)
layout(push_constant) uniform PushConstants {
    uint instanceOffset;   // base index of this batch inside instances[]
    float hasAlbedoTex;    // forwarded to fragment for runtime branch
    uint _pad0;
    uint _pad1;
} pc;

// Outputs to the fragment shader
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out vec4 fragAlbedoMetallic;
layout(location = 5) out float fragRoughness;

void main() {
    InstanceData inst = instanceBuf.instances[pc.instanceOffset + gl_InstanceIndex];

    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    fragWorldPos = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(inst.model))) * inNormal;
    fragColor  = inColor;
    fragTexCoord = inTexCoord;
    fragAlbedoMetallic = inst.albedoAndMetallic;
    fragRoughness = inst.roughnessAndFlags.r;
}
