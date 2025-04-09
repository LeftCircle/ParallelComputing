#include <iostream>
#include <string.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include "tests.h"

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
        glColor3f(1.0, 0.0, 0.0); glVertex2f(0.0, 1.0);
        glColor3f(0.0, 1.0, 0.0); glVertex2f(-1.0, -1.0);
        glColor3f(0.0, 0.0, 1.0); glVertex2f(1.0, -1.0);
    glEnd();
    glutSwapBuffers();
}

int main(int argc, char** argv) {

	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests();
		return 0;
	}

	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("OpenGL Test");
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}