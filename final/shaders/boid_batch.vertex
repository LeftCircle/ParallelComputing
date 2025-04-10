#version 330 core

// Input vertex data (local boid model)
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;

// Per-instance data
layout(location = 2) in vec3 instancePosition;    // Boid position
layout(location = 3) in vec4 instanceRotation;    // Rotation as quaternion
layout(location = 4) in vec3 instanceColor;       // Boid color

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec3 fragNormal;
out vec3 fragPosition;
out vec3 fragColor;

// Quaternion rotation function
mat3 quatToMat3(vec4 q) {
    float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    float qx2 = qx * qx, qy2 = qy * qy, qz2 = qz * qz;
    
    return mat3(
        1.0 - 2.0 * (qy2 + qz2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy),
        2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx2 + qz2), 2.0 * (qy * qz - qw * qx),
        2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx2 + qy2)
    );
}

void main() {
    // Apply instance transformation to vertex
    mat3 rotMatrix = quatToMat3(instanceRotation);
    vec3 worldPos = rotMatrix * vertexPosition + instancePosition;
    
    // Calculate world-space normal
    vec3 worldNormal = normalize(rotMatrix * vertexNormal);
    
    // Pass data to fragment shader
    fragPosition = worldPos;
    fragNormal = worldNormal;
    fragColor = instanceColor;
    
    // Calculate final position
    gl_Position = projection * view * vec4(worldPos, 1.0);
}