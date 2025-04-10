#include "scene.h"


Scene::Scene(int n_boids){
	// Initialize the view and controller
	view = new View(&boids);
	controller = new Controller(view, &boids);
	for (int i = 0; i < n_boids; ++i) {
		// Give each boid a random position and velocity
		Eigen::Vector3d position(rand() % 100, rand() % 100, rand() % 100);
		Eigen::Vector3d velocity((rand() % 200 - 100) / 100.0, (rand() % 200 - 100) / 100.0, (rand() % 200 - 100) / 100.0);
		boids.emplace_back(position, velocity, 10.0, 1.0);
	}
}

Scene::~Scene(){
	delete view;
	delete controller;
}
