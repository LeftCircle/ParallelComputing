#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <Eigen/Dense>

#include "tests.h"
#include "boids_oop.h"
#include "view.h"
#include "controller.h"
#include "vector.h"


const Vector3d WORLD_MIN(-100, -100, -100);
const Vector3d WORLD_MAX(100, 100, 100);
const float MAX_SPEED = 0.25;
const float MAX_FORCE = 1.0;


float randf_range(float low, float high);
Vector3d rand_vec3d(const Vector3d& min, const Vector3d& max);

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