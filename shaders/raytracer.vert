#version 450

// Fullscreen triangle without any vertex buffer.
// gl_VertexIndex: 0 -> (0,0), 1 -> (2,0), 2 -> (0,2)
// These positions cover the screen when mapped to clip space below.

layout(location = 0) out vec2 vUv;

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
