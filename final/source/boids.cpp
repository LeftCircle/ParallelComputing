#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <string.h>
//#include <GL/gl.h>
//#include <GL/glut.h>
#include <Eigen/Dense>
#include <vector>

#include "tests.h"
#include "boids_oop.h"
#include "view.h"
#include "controller.h"
#include "scene.h"

const float dt = 1.0 / 60.0; // time step for simulation

Scene* scene_ptr = nullptr; // Global pointer to the scene


void doSimulation() {
	// Update the boid's position and velocity
	for (BoidOOP& boid : scene_ptr->boids) {
		boid.flock(scene_ptr->boids);
		boid.update(dt);
	}
	static int count = 0;
  
	//if(count == 0)         // only update the display after every displayInterval time steps
	glutPostRedisplay();
	
	//count = (count + 1) % BubbleModel.displayInterval();
}

void doDisplay(){
	scene_ptr->view->updateDisplay();
}

void doReshape(int width, int height){
	scene_ptr->view->reshapeWindow(width, height);
}

void handleKey(unsigned char key, int x, int y){
	scene_ptr->controller->handleKey(key, x, y);
}
void handleButtons(int button, int state, int x, int y){
	scene_ptr->controller->handleButtons(button, state, x, y);
}

void handleMotion(int x, int y) {
	scene_ptr->view->handleMotion(x, y);
	glutPostRedisplay();
}

int main(int argc, char** argv) {

	std::cout << "Running Boids simulation..." << std::endl;
	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests();
		return 0;
	}

	glutInit(&argc, argv);
	std::cout << "GLUT initialized..." << std::endl;
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(600, 800);
	glutCreateWindow("Boids");
	
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
		return -1;
	}
	//CY_GL_REGISTER_DEBUG_CALLBACK;
	
	std::cout << "GLEW initialized..." << std::endl;
	
	Scene scene(100); // Create a scene with 100 boids
	scene_ptr = &scene; // Set the global pointer to the scene
	

	// register callback to handle events
	glutDisplayFunc(doDisplay);
	glutReshapeFunc(doReshape);
	glutKeyboardFunc(handleKey);
	glutMouseFunc(handleButtons);
	glutMotionFunc(handleMotion);
  
	// idle function is called whenever there are no other events to process
	glutIdleFunc(doSimulation);
	
	// set up the camera viewpoint, materials, and lights
	scene.view->setInitialView();
	scene.view->register_obj_mesh(default_obj_path);
	
	glutMainLoop();

	std::cout << "Exiting..." << std::endl;
	// Clean up
	scene_ptr = nullptr; // Reset the global pointer to the scene

    return 0;
}