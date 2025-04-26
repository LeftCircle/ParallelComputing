#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <Eigen/Dense>

#include "tests.h"
#include "boids_oop.h"
#include "view.h"
#include "controller.h"
#include "vector.h"



const float MAX_FORCE = 1.0;


float randf_range(float low, float high);
Vector3d rand_vec3d(const Vector3d& min, const Vector3d& max);

class Scene {
public:
	Vector3d WORLD_MIN = Vector3d(-100, -100, -100);
	Vector3d WORLD_MAX = Vector3d(100, 100, 100);
	float MAX_SPEED = 3.0f;
	Scene(int n_boids, const Vector3d& world_min, const Vector3d& world_max, float max_speed);
	~Scene();

public:
	std::vector<BoidOOP> boids;
	View *view;
	Controller *controller;

};


#endif