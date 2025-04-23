#version 330 core

layout(location = 0) out vec4 fragColor;

in vec3 vNormal;
in vec3 vViewSpacePos;
in vec2 vTexCoord;

uniform sampler2D tex;
uniform sampler2D diffuse_map;
uniform sampler2D specular_map;

// Light direction might have to be passed by the vertex shader to 
// be interpolated. But with directional light, it should be fine. 
uniform vec3 light_direction;

uniform vec3 intensity_k_diffuse;
uniform vec3 intensity_k_specular;
uniform vec3 intensity_k_ambient;

uniform float shininess;

void main()
{
	// Texture
	vec4 tex_color = texture(tex, vTexCoord);
	vec4 diffuse_color = texture(diffuse_map, vTexCoord);
	vec4 specular_color = texture(specular_map, vTexCoord);

	vec3 diffuse = intensity_k_diffuse * diffuse_color.rgb;
	vec3 specular = intensity_k_specular * specular_color.rgb;
	vec3 ambient = intensity_k_ambient * tex_color.rgb;

	// Shading
	vec3 N = normalize(vNormal);
	vec3 omega = normalize(-light_direction);
	vec3 view_direction = normalize(-vViewSpacePos);
	vec3 h = normalize(omega + view_direction);
	
	vec3 diffuse_shade = diffuse * max(dot(N, omega), 0.0);
	vec3 specular_shade = specular * pow(max(dot(N, h), 0.0), shininess);
	if (dot(N, omega) < 0.0)
	{
		specular_shade = vec3(0.0);
	}
	vec3 color = ambient + diffuse_shade + specular_shade;
	fragColor = vec4(color, 1.0);
}

// // Inputs from vertex shader
// in vec3 fragNormal;
// in vec3 fragPosition;
// in vec3 fragColor;

// // Lighting uniforms
// uniform vec3 viewPos;        // Camera position
// uniform vec3 lightPos;       // Light position
// uniform vec3 lightColor;     // Light color
// uniform float ambientStrength = 0.3;
// uniform float specularStrength = 0.5;
// uniform float shininess = 32.0;

// // Output color
// out vec4 outColor;

// void main() {
//     // Ambient lighting
//     vec3 ambient = ambientStrength * lightColor;
    
//     // Diffuse lighting
//     vec3 lightDir = normalize(lightPos - fragPosition);
//     float diff = max(dot(fragNormal, lightDir), 0.0);
//     vec3 diffuse = diff * lightColor;
    
//     // Specular lighting
//     vec3 viewDir = normalize(viewPos - fragPosition);
//     vec3 reflectDir = reflect(-lightDir, fragNormal);
//     float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
//     vec3 specular = specularStrength * spec * lightColor;
    
//     // Combine all lighting with object color
//     vec3 result = (ambient + diffuse + specular) * fragColor;
//     outColor = vec4(result, 1.0);
// }