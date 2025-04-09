#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <GL/glut.h>

#include <vector>

#include "boids_oop.h"
#include "view.h"

class Controller {
private:
	View* view; // Pointer to the View object
	std::vector<BoidOOP>* boids; // Pointer to the vector of BoidOOP objects
	
public:
	// Constructor
	Controller(View* scene_view, std::vector<BoidOOP>* scene_boids);

	// Destructor
	~Controller();

	// Handle keyboard input
	void handleKey(unsigned char key, int x, int y);

	// Handle mouse button events
	void handleButtons(int button, int state, int x, int y);

};

#endif