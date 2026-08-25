// light_volume_spot.vert — instanced spot-light volume proxy. gl_InstanceIndex
// indexes SpotLightSSBO. A sphere bounding box (icosphere at spotPos, radius
// spotRadius) is conservative but simple; the fragment shader discards by cone falloff.
#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PC {
    mat4 viewProj;
} pc;

struct SpotLight {
    vec3  position;
    float radius;
    vec3  direction;
    float intensity;
    vec3  color;
    float innerCos;
    float outerCos;
    int   shadowSlot;
    float _pad[2];
};
layout(std430, set = 0, binding = 6) readonly buffer SpotLightSSBO {
    SpotLight spotLights[];
};

layout(location = 0) out flat int vInstanceID;

void main() {
    vec3 lightPos = spotLights[gl_InstanceIndex].position;
    float r       = spotLights[gl_InstanceIndex].radius;
    vec3 worldPos = lightPos + inPosition * r;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
    vInstanceID = gl_InstanceIndex;
}
