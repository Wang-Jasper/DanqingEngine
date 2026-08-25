// ============================================================================
// ssao_blur.frag — Phase 11.9.4-improve  separable bilateral SSAO blur
// ============================================================================
// 5-tap depth-aware Gaussian blur, applied as 2 passes (X then Y) controlled
// by `dir` push constant.
//
// Inputs:
//   set=0 binding=0: aoIn      (R8_UNORM, single-channel AO)
//   set=0 binding=1: gPosition (RGBA32F, A=mask) — used as a depth proxy:
//                   we compare absolute view-space Z (length of (V·P_world))
//                   to weight neighbors. Same-surface neighbors keep weight,
//                   silhouette neighbors get suppressed → preserves edges
//                   while collapsing per-pixel SSAO noise.
//
// The Y pass also UPSAMPLES bilinearly: aoIn is half-resolution and the
// fragment runs at full resolution, so `texture(aoIn, uv)` performs a
// 2× bilinear lookup as a side effect. The bilateral weights then prevent
// half-res edges from leaking onto full-res silhouettes.
// ============================================================================
#version 450

layout(set = 0, binding = 0) uniform sampler2D aoIn;
layout(set = 0, binding = 1) uniform sampler2D gPosition;
layout(set = 0, binding = 2) uniform BlurUBO {
    mat4 view;        // for view-space Z computation
    vec2 invInputSize; // 1.0 / size(aoIn) — half-res pixel size
    vec2 _pad;
} ubo;

layout(push_constant) uniform BlurPush {
    int dir;       // 0 = horizontal, 1 = vertical
    int _p0;
    int _p1;
    int _p2;
} pc;

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out float outAO;

// 5-tap Gaussian sigma=1.5 (normalized later by total weight).
const float G[5] = float[](0.06136, 0.24477, 0.38774, 0.24477, 0.06136);

float viewZ(vec2 uv) {
    vec4 p = texture(gPosition, uv);
    if (p.a == 0.0) return 1e30; // sky → effectively infinite distance
    return (ubo.view * vec4(p.rgb, 1.0)).z;
}

void main() {
    vec2 stepUV = pc.dir == 0 ? vec2(ubo.invInputSize.x, 0.0)
                              : vec2(0.0, ubo.invInputSize.y);

    float centerZ = viewZ(fragTexCoord);
    float ao   = 0.0;
    float wSum = 0.0;
    // Edge-aware sigma in view-space units. Tighter than radius (we want
    // neighbors on the SAME surface only).
    const float depthSigma = 0.2;

    for (int i = -2; i <= 2; ++i) {
        vec2 uv = fragTexCoord + stepUV * float(i);
        float gw = G[i + 2];
        float aoSample = texture(aoIn, uv).r;
        float zSample  = viewZ(uv);
        float dz = (zSample - centerZ) / depthSigma;
        float bw = exp(-dz * dz);
        ao   += aoSample * gw * bw;
        wSum += gw * bw;
    }
    outAO = wSum > 0.0 ? ao / wSum : 1.0;
}
