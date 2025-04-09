#include "controller.h"


Controller::Controller(View* scene_view, std::vector<BoidOOP>* scene_boids){
	view = scene_view;
	boids = scene_boids;
}

Controller::~Controller(){
	// Destructor
	// No dynamic memory to free in this case
}

//
// Keyboard callback routine.
// Send model and view commands based on key presses
//
void Controller::handleKey(unsigned char key, int x, int y){
	const int ESC = 27;
	
	switch(key){
	  case 'k':           // toggle key light on and off
		view->toggleKeyLight();
		break;
		
	  case 'f':           // toggle fill light on and off
		view->toggleFillLight();
		break;
		
	  case 'r':           // toggle back light on and off
		view->toggleBackLight();
		break;
		
	  case 'g':           // toggle background color from grey to black
		view->toggleBackColor();
		break;
  
	  case 'i':			// I -- reinitialize view
	  case 'I':
		view->setInitialView();
		break;
		
	  case 'q':			// Q or Esc -- exit program
	  case 'Q':
	  case ESC:
		exit(0);
	}
	
	// always refresh the display after a key press
	glutPostRedisplay();
  }
  
//
// let the View handle mouse button events
// but pass along the state of the shift key also
//
void Controller::handleButtons(int button, int state, int x, int y) {
	bool shiftkey = (glutGetModifiers() == GLUT_ACTIVE_SHIFT);

	view->handleButtons(button, state, x, y, shiftkey);
	glutPostRedisplay();
}
