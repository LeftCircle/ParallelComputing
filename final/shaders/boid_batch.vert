#version 330 core

// Input vertex data (local boid model)
layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in vec2 vertex_texcoord;

// Per-instance data
layout(location = 3) in vec3 instancePosition;    // Boid position
layout(location = 4) in vec4 instanceRotation;    // Rotation as quaternion
layout(location = 5) in vec3 instanceColor;       // Boid color

// Uniforms
uniform mat4 view;
uniform mat4 projection;

out vec3 vNormal;
out vec3 vViewSpacePos;
out vec3 vTexCoord;

// These might replace the instance position and rotation?
// It's the model view projection matrix for the boid
//uniform mat4 mvp;
//uniform mat4 mv_points;
//uniform mat3 mv_normals;

// // Outputs to fragment shader
// out vec3 fragNormal;
// out vec3 fragPosition;
// out vec3 fragColor;

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

void main()
{
	vec3 world_pos = vertex_position + instancePosition;
    gl_Position = projection * view * vec4(world_pos, 1.0);
	
	// Might be missing rotation and stuff here?
	mat4 mv_points = view * instancePosition;
	
	// the mv_normals are relative to the camera view
	mat3 mv_normals = mat3(mv_points).inverse().transpose();

	vNormal = normalize(mv_normals * vertex_normal);
	vViewSpacePos = vec3(mv_points * vec4(vertex_position, 1.0));
	vTexCoord = textCoord;
}


// void main() {
//     // Apply instance transformation to vertex
//     mat3 rotMatrix = quatToMat3(instanceRotation);
//     vec3 worldPos = rotMatrix * vertexPosition + instancePosition;
    
//     // Calculate world-space normal
//     vec3 worldNormal = normalize(rotMatrix * vertexNormal);
    
//     // Pass data to fragment shader
//     fragPosition = worldPos;
//     fragNormal = worldNormal;
//     fragColor = instanceColor;
    
//     // Calculate final position
//     gl_Position = projection * view * vec4(worldPos, 1.0);
// }