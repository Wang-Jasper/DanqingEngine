// geometry.frag — Geometry Pass fragment shader: PBR material + texture
// sampling, MRT output.
#version 450

// Inputs from the vertex shader
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 fragAlbedoMetallic;
layout(location = 5) in float fragRoughness;

// set=1, binding=0: Albedo texture sampler
layout(set = 1, binding = 0) uniform sampler2D albedoMap;

// MRT outputs
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;

void main() {
    outPosition = vec4(fragWorldPos, 1.0);
    outNormal   = vec4(normalize(fragNormal), fragRoughness);

    vec3 albedo = fragAlbedoMetallic.rgb;
    float metallic = fragAlbedoMetallic.a;

    // Sample the albedo texture and modulate material color
    vec4 texColor = texture(albedoMap, fragTexCoord);
    albedo *= texColor.rgb;

    // Vertex color modulation
    albedo *= fragColor;

    outAlbedo = vec4(albedo, metallic);
}
