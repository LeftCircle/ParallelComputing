/*
 CPSC 8170 Physically Based Animation
 Donald H. House     8/23/2018
 
 Interface for Teapot-Bubble Viewer
*/

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>

#include "camera.h"
#include "boids_oop.h"
#include "rcObjModifier.h"
#include "rcTexture.h"
#include "cyGL.h"

#ifndef __VIEW_H__
#define __VIEW_H__

class View{
public:
	Vector3d world_min;
	Vector3d world_max;

private:
	const int width;                // initial window dimensions
	const int height;               // proportions match 1080p proportions but 1/2 scale

	const float near_plane;               // distance of near clipping plane
	const float far_plane;                // distance of far clipping plane
	const float fov;                // camera vertical field of view

	const float modelsize;          // dimension of the teapot
	const float modeldepth;         // initial distance of center of teapot from camera

	const float diffuse_fraction;   // fraction of reflected light that is diffuse
	const float specular_fraction;  // fraction of reflected light that is specular
	const float shininess;          // specular exponent

	// colors used for lights, and screen background
	const float white[4];
	const float dim_white[4];
	const float grey_background[4];

	// colors used for materials
	const float base_color[3];
	const float highlight_color[3];

	// The perspective camera
	Camera* camera;

	// The simulation model
	//Model *themodel;
	std::vector<BoidOOP>* boids;

	// Switches to turn lights on and off
	bool KeyOn;
	bool FillOn;
	bool BackOn;

	// Switch to determine background color
	bool BackgroundGrey;

	// Current window dimensions
	int Width;
	int Height;

	// cy program for handling shader compilation
	cy::GLSLProgram program;

	std::vector<rc::rcTriMeshForGL> _meshes;

	GLuint instanceVBO;
	GLuint vertexVBO;
	GLuint normalVBO;
	GLuint texVBO;
	GLuint boidVAO;

	std::vector<Eigen::Vector3f> positions;

	// position the lights, never called outside of this class
	void setLights();

	// draw the model, never called outside of this class
	void drawModel();
	void _bind_mesh(rc::rcTriMeshForGL& mesh);
	void _bind_buffers(rc::rcTriMeshForGL& mesh);
	void _bind_textures(rc::rcTriMeshForGL& mesh);
	void _bind_material(rc::rcTriMeshForGL& mesh, rc::Texture& texture, const int texture_id, const char* sampler_name);


public:
	View();
	View(std::vector<BoidOOP>* boids_ptr);
	~View();

	// initialize the state of the viewer to start-up defaults
	void setInitialView();

	// Toggle lights on/off
	void toggleKeyLight();
	void toggleFillLight();
	void toggleBackLight();

	// Toggle background color grey/black
	void toggleBackColor();

	// Handlers for mouse events
	void handleButtons(int button, int state, int x, int y, bool shiftkey);
	void handleMotion(int x, int y);

	// redraw the display
	void updateDisplay();

	// handle window resizing, to keep consistent viewscreen proportions
	void reshapeWindow(int width, int height);

	// accessors to determine current screen width and height
	int getWidth(){return Width;}
	int getHeight(){return Height;}

	void register_obj_mesh(const char* obj_path);

	void init_boid_rendering(const int n_boids);
	void draw_bounding_box(const Vector3d& min, const Vector3d& max);
	
};

#endif

