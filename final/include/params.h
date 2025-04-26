#pragma once
#include <string>
#include "vector.h"

struct Params {
    float max_speed = 3.0f;
    float max_force = 1.0f;
    Vector3d world_min = Vector3d(-100, -100, -100);
    Vector3d world_max = Vector3d(100, 100, 100);
	std::string obj_path = "assets/cone.obj";
	int n_boids = 10;

    bool load(const std::string& filename);
};