#version 330 core

layout (location = 0) in vec3 vertexPosition;
layout(location = 1) in vec4 vertexColor;
uniform mat4 mvp;
uniform vec2 scr_offset;

out vec4 fragmentColor;

void main() {
    vec4 clip = mvp * vec4(vertexPosition, 1.0);
    gl_Position = vec4(clip.xyz + vec3(clip.w * scr_offset, 0.0), clip.w);

    fragmentColor = vertexColor;
}