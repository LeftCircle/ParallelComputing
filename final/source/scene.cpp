#include "scene.h"


Scene::Scene(int n_boids, const Vector3d& world_min, const Vector3d& world_max, float max_speed){
	// Initialize the view and controller
	MAX_SPEED = max_speed;
	WORLD_MAX = world_max;
	WORLD_MIN = world_min;
	view = new View(&boids);
	controller = new Controller(view, &boids);
	const float speed_range = 300.0;
	for (int i = 0; i < n_boids; ++i) {
		// Give each boid a random position and velocity
		Vector3d rand_pos = rand_vec3d(world_min / 2.0, world_max / 2.0);
		Vector3d rand_vel = Vector3d(randf_range(-speed_range, speed_range), randf_range(-speed_range, speed_range), randf_range(-speed_range, speed_range));
		rand_vel = rand_vel.normalize() * randf_range(0, max_speed);
		Eigen::Vector3f position(rand_pos.x, rand_pos.y, rand_pos.z);
		Eigen::Vector3f velocity(rand_vel.x, rand_vel.y, rand_vel.z);
		// Create a new boid and add it to the vector
		boids.emplace_back(position, velocity, MAX_SPEED, MAX_FORCE);
	}
	view->init_boid_rendering(n_boids);
	view->world_max = WORLD_MAX;
	view->world_min = WORLD_MIN;
	
}

Scene::~Scene(){
	delete view;
	delete controller;
}


Vector3d rand_vec3d(const Vector3d& min, const Vector3d& max) {
	Vector3d rand_vec;
	rand_vec.x = ((double)rand() / RAND_MAX) * (max.x - min.x) + min.x;
	rand_vec.y = ((double)rand() / RAND_MAX) * (max.y - min.y) + min.y;
	rand_vec.z = ((double)rand() / RAND_MAX) * (max.z - min.z) + min.z;
	return rand_vec;
}

float randf_range(float low, float high) {
	return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}