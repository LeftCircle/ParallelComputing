//#include <GL/glew.h>
//#include <GL/freeglut.h>

#include "boids.h"

const float dt = 1.0 / 60.0; // time step for simulation

// Eigen::Vector3d position(0, 0, 0);
// Eigen::Vector3d velocity(1, 0, 0);

// BoidOOP boid(position, velocity, 10.0, 1.0);

// std::vector<BoidOOP> boids = {boid};

// View view(&boids);

// Controller controller(&view, &boids);
Scene scene(100); // Create a scene with 100 boids


//
// idle callback: let the Model handle simulation timestep events
//
void doSimulation() {
	// Update the boid's position and velocity
	for (BoidOOP& boid : scene.boids) {
		boid.update(dt);
	}
	static int count = 0;
  
	if(count == 0)         // only update the display after every displayInterval time steps
	  glutPostRedisplay();
	
	//count = (count + 1) % BubbleModel.displayInterval();
}

//
// let the View handle display events
//
void doDisplay(){
	scene.view->updateDisplay();
}

void doReshape(int width, int height){
	scene.view->reshapeWindow(width, height);
}

void handleKey(unsigned char key, int x, int y){
	scene.controller->handleKey(key, x, y);
}
void handleButtons(int button, int state, int x, int y){
	scene.controller->handleButtons(button, state, x, y);
}

void handleMotion(int x, int y) {
	scene.view->handleMotion(x, y);
	glutPostRedisplay();
}

int main(int argc, char** argv) {

	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests();
		return 0;
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(scene.view->getWidth(), scene.view->getHeight());
	glutCreateWindow("Boids");
	glewInit();
	
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
	
	glutMainLoop();

    return 0;
}