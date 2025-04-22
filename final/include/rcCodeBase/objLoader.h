#ifndef OBJLOADER_H
#define OBJLOADER_H

#include "objForGL.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>

bool is_obj_file(const std::string& path) {
	return path.substr(path.size() - 4, 4) == ".obj";
}

size_t split_string(const std::string& line, std::vector<std::string>& tokens, char delimiter) {
	tokens.clear();
	std::string token;
	std::istringstream tokenStream(line);
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens.size();
}

class ObjLoader
{
	// OBJ File format:
	// mtllib [filename].mtl
	// v [x] [y] [z]
	// vt [u] [v]
	// vn [x] [y] [z]
	// f [v1]/[vt1]/[vn1] [v2]/[vt2]/[vn2] [v3]/[vt3]/[vn3]
	// usemtl [material_name]
	// s [number]
	// o [object_name]
	// g [group_name]
	// [comment]
private:
	std::vector<rc::ObjMesh> _meshes;

	void _write_three_token_face_to_mesh(rc::ObjMesh& mesh, const std::vector<std::string>& tokens, std::vector<std::string>& scratch) {
		rc::ObjFaceIndeces v_face, vt_face, vn_face;
		for (int i = 0; i < 3; i++) {
			split_string(tokens[i], scratch, '/');
			v_face[i] = std::stoi(scratch[0]);
			vt_face[i] = std::stoi(scratch[1]);
			vn_face[i] = std::stoi(scratch[2]);
		}
		mesh.vertex_faces.push_back(v_face);
		mesh.tex_coord_faces.push_back(vt_face);
		mesh.normal_faces.push_back(vn_face);
	}

	void _add_faces_to_mesh(rc::ObjMesh& mesh, const std::vector<std::string>& tokens) {
		// The face format is f [v1]/[vt1]/[vn1] [v2]/[vt2]/[vn2] [v3]/[vt3]/[vn3]
		// or f [v1]/[vt1]/[vn1] [v2]/[vt2]/[vn2] [v3]/[vt3]/[vn3] [v4]/[vt4]/[vn4]
		// This creates two triangles like so:
		// 3-------2
		// |      /|
		// |    /  |
		// |  /    |
		// |/      |
		// 0-------1
		// and creates two faces [0, 1, 2] and [0, 2, 3]
		// split the line by spaces and create a objFaceIndeces from the next 3 values
		// If the first token is 'f', then disregard it
		int f_start_index = tokens[0] == "f" ? 1 : 0;
		int n_face_tokens = tokens.size() - f_start_index;
		assert(n_face_tokens == 3 or n_face_tokens == 4, "Only support for triangles and quads");
		static std::vector<std::string> v_n_t_faces;
		static std::vector<std::string> first_triangle_tokens = { tokens[f_start_index], tokens[f_start_index + 1], tokens[f_start_index + 2] };
		static std::vector<std::string> second_triangle_tokens;

		_write_three_token_face_to_mesh(mesh, first_triangle_tokens, v_n_t_faces);

		if (n_face_tokens == 4)
		{
			second_triangle_tokens = { tokens[f_start_index], tokens[f_start_index + 2], tokens[f_start_index + 3] };
			_write_three_token_face_to_mesh(mesh, second_triangle_tokens, v_n_t_faces);
		}
	}

	// Reads the file and adds obj meshes to the _meshes vector.
	void _build_obj_meshes(std::ifstream& file) {
		rc::ObjMesh mesh;
		bool reading_f = false;
		std::string line;
		double x, y, z;
		std::vector<std::string> v_n_t_faces = { "", "", "", "" };
		while (std::getline(file, line)) {
			if (line.size() < 2 or line[0] == '#') {
				continue;
			}

			std::istringstream iss(line);
			std::string id;
			iss >> id;
			if (id != "f") {
				iss >> x >> y >> z;
			}

			if (reading_f and id != "f") {
				// create a new mesh and continue
				_meshes.push_back(mesh);
				mesh = rc::ObjMesh();
				reading_f = false;
			}

			if (id == "f") {
				reading_f = true;
				add_face_indexes(mesh, line);
			}
			else if (id == "v") {
				mesh.vertices.push_back(rc::Vector3(x, y, z));
			}
			else if (id == "vt") {
				mesh.tex_coords.push_back(rc::Vector3(x, y, z));
			}
			else if (id == "vn") {
				mesh.normals.push_back(rc::Vector3(x, y, z));
			}
		}
		// add the last mesh if it has any data
		if (mesh.vertices.size() > 0) {
			_meshes.push_back(mesh);
		}
	}

	void _add_value_to_v_or_t_or_n(rc::ObjFaceIndeces& v_face, rc::ObjFaceIndeces& vt_face, rc::ObjFaceIndeces& vn_face, const int value, const int vtn_case, const int face_xyz)
	{
		if (vtn_case == 0) {
			v_face[face_xyz] = value;
		}
		else if (vtn_case == 1) {
			vt_face[face_xyz] = value;
		}
		else if (vtn_case == 2) {
			vn_face[face_xyz] = value;
		}
	}

	void _add_face_indexes_from_vtn(rc::ObjFaceIndeces& v_face, rc::ObjFaceIndeces& vt_face, rc::ObjFaceIndeces& vn_face, const std::string& line, const int face_xyz)
	{
		// We have been given the string of either v/vt/vn or v//vn or v/vt or v
		// We need to split the string by / and convert the values to integers, then add to the correct face.
		int vtn_case = 0; // 0 = vertex, 1 = texture, 2 = normal
		unsigned int value = 0;
		for (char c : line) {
			if (c == '/') {
				vtn_case = (vtn_case + 1) % 3;
				value = 0;
			}
			else if (c >= '0' and c <= '9') {
				int current_number = c - '0';
				value = value * 10 + current_number;
				_add_value_to_v_or_t_or_n(v_face, vt_face, vn_face, value, vtn_case, face_xyz);
			}
		}

	}

	void _add_faces_to_mesh_with_vtn_strings(rc::ObjMesh& mesh, const std::string& s1, const std::string& s2, const std::string& s3) {
		rc::ObjFaceIndeces v_face, vt_face, vn_face;
		_add_face_indexes_from_vtn(v_face, vt_face, vn_face, s1, 0);
		_add_face_indexes_from_vtn(v_face, vt_face, vn_face, s2, 1);
		_add_face_indexes_from_vtn(v_face, vt_face, vn_face, s3, 2);
		mesh.vertex_faces.push_back(v_face);
		mesh.tex_coord_faces.push_back(vt_face);
		mesh.normal_faces.push_back(vn_face);
	}


public:
	ObjLoader() {};
	~ObjLoader() {}; // Remove this line

	bool loadObjFile(const std::string& path)
	{
		if (!is_obj_file(path)) {
			return false;
		}
		// add the current path to the file name

		std::ifstream file(path);
		if (!file.is_open()) {
			std::cerr << "Failed to open file: " << path << std::endl;
			return false;
		}
		_build_obj_meshes(file);
		return true;
	}

	std::vector<rc::ObjMesh>& getObjMeshes() { return _meshes; }
	const std::vector<rc::ObjMesh>& getObjMeshes() const { return _meshes; }

	void add_face_indexes(rc::ObjMesh& mesh, const std::string& line) {
		std::istringstream iss(line);
		std::string id, f0, f1, f2, f3;
		iss >> id >> f0 >> f1 >> f2;
		_add_faces_to_mesh_with_vtn_strings(mesh, f0, f1, f2);
		if (iss >> f3) {
			_add_faces_to_mesh_with_vtn_strings(mesh, f0, f2, f3);
		}
	}
};


#endif // OBJLOADER_H