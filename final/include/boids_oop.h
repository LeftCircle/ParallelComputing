/*
This is the basic object oriented apporach to boids. In this simulation, 
a boid will be an object that has a position, velocity, and acceleration.

Each boid will implement its steering behavior in a method called `update()`.
The result of the simulation will be an updated position of the boid, which can 
be used to draw the boid on the screen either with OpenGL or as a plugin to Godot. 
*/

#ifndef BOIDS_OOP_H
#define BOIDS_OOP_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>


inline const char* default_obj_path = "assets/cone.obj";

class BoidOOP {
public:
	// The boids model, which is just a triangular prism
	std::vector<Eigen::Vector3f> model_local_coords{
		// Draw the triange for the base, which is a square
		// on the XY plane
		Eigen::Vector3f(-1, 1, 0),
		Eigen::Vector3f(1, 1, 0),
		Eigen::Vector3f(1, -1, 0),
		Eigen::Vector3f(-1, -1, 0),
		Eigen::Vector3f(-1, 1, 0),
		Eigen::Vector3f(1, -1, 0),

		// Now draw the four triangles for the prism,
		// Where the z coordinate is (0, 0, -1)
		Eigen::Vector3f(-1, 1, 0),
		Eigen::Vector3f(0, 0, -1),
		Eigen::Vector3f(1, 1, 0),

		Eigen::Vector3f(1, 1, 0),
		Eigen::Vector3f(0, 0, -1),
		Eigen::Vector3f(1, -1, 0),

		Eigen::Vector3f(1, -1, 0),
		Eigen::Vector3f(0, 0, -1),
		Eigen::Vector3f(-1, -1, 0),

		Eigen::Vector3f(-1, -1, 0),
		Eigen::Vector3f(0, 0, -1),
		Eigen::Vector3f(-1, 1, 0)
	};

	

private:
	Eigen::Vector3f position; 
	Eigen::Vector3f velocity; 
	Eigen::Vector3f acceleration; 
	double max_speed; 
	double max_force;

	void apply_boundary_conditions(Eigen::Vector3f& min, Eigen::Vector3f& max);

	

public:
	BoidOOP(Eigen::Vector3f& new_position, Eigen::Vector3f& new_velocity, double maxSpeed, double maxForce);
	~BoidOOP();

	void update(float dt);

	void applyForce(const Eigen::Vector3f& force);
	Eigen::Vector3f seek(const Eigen::Vector3f& target);
	Eigen::Vector3f separate(const std::vector<BoidOOP>& boids, double desiredSeparation);
	void flock(const std::vector<BoidOOP>& boids);

	std::vector<Eigen::Vector3f> get_global_coordinates();

	const Eigen::Vector3f getPosition() const {
		return position;
	}

	const Eigen::Vector3f getVelocity() const {
		return velocity;
	}

	const std::vector<Eigen::Vector3f> getModelLocalCoords() const {
		return model_local_coords;
	}
};


#endif

