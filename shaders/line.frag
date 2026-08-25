#version 450

layout(location = 0) in vec3 fragColor;

// Outputs to the G-Buffer MRT (same layout as geometry.frag)
layout(location = 0) out vec4 outPosition;  // RT0: Position (w=1 marks a valid pixel)
layout(location = 1) out vec4 outNormal;    // RT1: Normal
layout(location = 2) out vec4 outAlbedo;    // RT2: Albedo

void main() {
    // Mark as a valid pixel (w=1) with position 0 (near)
    outPosition = vec4(0.0, 0.0, 0.0, 1.0);
    // Zero normal and roughness so the lighting pass still adds ambient
    outNormal   = vec4(0.0, 0.0, 0.0, 0.0);
    // Use the light color, brightened to stay visible
    outAlbedo   = vec4(fragColor * 8.0, 1.0);
}
