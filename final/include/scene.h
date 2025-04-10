#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <Eigen/Dense>

#include "tests.h"
#include "boids_oop.h"
#include "view.h"
#include "controller.h"


// Eigen::Vector3d position(0, 0, 0);
// Eigen::Vector3d velocity(1, 0, 0);

// BoidOOP boid(position, velocity, 10.0, 1.0);

// std::vector<BoidOOP> boids = {boid};

// View view(&boids);

// Controller controller(&view, &boids);


class Scene {
public:
	Scene(int n_boids);
	~Scene();

public:
	std::vector<BoidOOP> boids;
	View *view;
	Controller *controller;

};


#endif