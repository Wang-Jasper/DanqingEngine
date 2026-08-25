// shadow.frag — empty fragment shader for the directional shadow pass.
// Depth-only: no color attachments, so the driver elides this stage, but
// PipelineBuilder requires a paired vert + frag.
#version 450

void main() {
    // No color writes; depth is written automatically by the rasterizer.
}
