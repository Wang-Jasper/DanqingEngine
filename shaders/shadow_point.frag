// shadow_point.frag — point-light cubemap shadow pass. Writes linear
// distance/range to gl_FragDepth so lighting.frag can compare length(L)/range
// against the cubemap depth for hardware PCF visibility.
#version 450

layout(location = 0) in vec3 vWorldPos;

layout(set = 0, binding = 0) uniform PointShadowVPUBO {
    mat4 cubeVP[6];
    vec4 lightPosRange; // xyz = lightPos, w = range
} ubo;

void main() {
    float distToLight = length(vWorldPos - ubo.lightPosRange.xyz);
    float depth = distToLight / max(ubo.lightPosRange.w, 1e-4);
    // Writing gl_FragDepth disables early-Z, but this depth-only pass has no
    // color writes, so the cost is negligible.
    gl_FragDepth = clamp(depth, 0.0, 1.0);
}
