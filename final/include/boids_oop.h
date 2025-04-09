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


class BoidOOP {
private:
	Eigen::Vector3d position; 
	Eigen::Vector3d velocity; 
	Eigen::Vector3d acceleration; 
	double maxSpeed; 
	double maxForce; 
};


#endif

