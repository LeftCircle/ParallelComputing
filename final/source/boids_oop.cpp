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


void BoidOOP::applyForce(const Eigen::Vector3d& force) {
    // F = ma, but we assume mass = 1, so F = a
    acceleration += force;
}

// Seek behavior: steer towards a target location
Eigen::Vector3d BoidOOP::seek(const Eigen::Vector3d& target) {
    Eigen::Vector3d desired = target - position;
    desired.normalize();
    desired *= max_speed;
    
    Eigen::Vector3d steer = desired - velocity;
    
    // Limit the force
    if (steer.norm() > max_force) {
        steer = steer.normalized() * max_force;
    }
    
    return steer;
}

// Separation: steer to avoid crowding local flockmates
Eigen::Vector3d BoidOOP::separate(const std::vector<BoidOOP>& boids, double desiredSeparation) {
    Eigen::Vector3d steer = Eigen::Vector3d::Zero();
    int count = 0;
    
    // Check each boid to see if it's too close
    for (const BoidOOP& other : boids) {
        double d = (position - other.getPosition()).norm();
        
        // If the boid is too close
        if ((d > 0) && (d < desiredSeparation)) {
            Eigen::Vector3d diff = position - other.getPosition();
            diff.normalize();
            diff /= d;  // Weight by distance
            steer += diff;
            count++;
        }
    }
    
    // Average
    if (count > 0) {
        steer /= count;
    }
    
    // As long as the vector is greater than 0
    if (steer.norm() > 0) {
        steer.normalize();
        steer *= max_speed;
        steer -= velocity;
        
        // Limit the force
        if (steer.norm() > max_force) {
            steer = steer.normalized() * max_force;
        }
    }
    
    return steer;
}

void BoidOOP::flock(const std::vector<BoidOOP>& boids) {
    Eigen::Vector3d sep = separate(boids, 25.0) * 1.5; // Separation has higher weight
    
    // Apply the forces
    applyForce(sep);
    
    // Add boundary forces to keep boids in the world
    Eigen::Vector3d boundary_force = Eigen::Vector3d::Zero();
    
    // Simple boundary handling - push away from world edges
    if (position.x() < -100 + 10) boundary_force.x() += 0.1;
    if (position.x() > 100 - 10) boundary_force.x() -= 0.1;
    if (position.y() < -100 + 10) boundary_force.y() += 0.1;
    if (position.y() > 100 - 10) boundary_force.y() -= 0.1;
    if (position.z() < -100 + 10) boundary_force.z() += 0.1;
    if (position.z() > 100 - 10) boundary_force.z() -= 0.1;
    
    applyForce(boundary_force);
}