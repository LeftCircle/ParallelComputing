#include "params.h"
#include <fstream>
#include <sstream>

bool Params::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                if (key == "max_speed") max_speed = std::stof(value);
                else if (key == "max_force") max_force = std::stof(value);
                else if (key == "world_min") {
                    float x, y, z;
                    sscanf(value.c_str(), "%f,%f,%f", &x, &y, &z);
                    world_min = Vector3d(x, y, z);
                }
                else if (key == "world_max") {
                    float x, y, z;
                    sscanf(value.c_str(), "%f,%f,%f", &x, &y, &z);
                    world_max = Vector3d(x, y, z);
                }
				else if (key == "obj_path") {
					obj_path = value;
				}
				else if (key == "n_boids") n_boids = std::stoi(value);
				else {
					std::cerr << "Unknown parameter: " << key << std::endl;
					return false;
				}
			}
			else {
				std::cerr << "Error reading value for key: " << key << std::endl;
				return false;
            }
        }
    }
    return true;
}