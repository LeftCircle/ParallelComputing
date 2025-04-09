#include "boids_oop.h"


BoidOOP::BoidOOP(Eigen::Vector3d& new_position, Eigen::Vector3d& new_velocity, double maxSpeed, double maxForce){
	position = new_position;
	velocity = new_velocity;
	acceleration.setZero();
	max_speed = maxSpeed;
	max_force = maxForce;
}

BoidOOP::~BoidOOP() {
	// Destructor
}

void BoidOOP::update(float dt) {
	// Update the position and velocity of the boid
	velocity += acceleration * dt;
	float squared_speed = velocity.squaredNorm();
	if (squared_speed > max_speed * max_speed) {
		velocity = velocity.normalized() * max_speed; // Limit speed
	}
	position += velocity * dt;
	acceleration.setZero(); // Reset acceleration
}

std::vector<Eigen::Vector3d> BoidOOP::get_global_coordinates() {
	std::vector<Eigen::Vector3d> global_coords;
	for (const Eigen::Vector3d& local_coord : model_local_coords) {
		global_coords.push_back(position + local_coord);
	}
	return global_coords;
}
