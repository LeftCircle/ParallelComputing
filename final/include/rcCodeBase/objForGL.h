#ifndef OBJ_FOR_GL_HPP
#define OBJ_FOR_GL_HPP

#include <vector>
#include <unordered_map>
#include <string>

namespace rc
{
	struct Vector3
	{
		Vector3() { x = 0; y = 0; z = 0; };
		Vector3(float x_, float y_, float z_) { x = x_; y = y_; z = z_; };
		float x, y, z;
	};

	struct Material
	{
		Material() { name = ""; ka = Vector3(); kd = Vector3(); ks = Vector3(); ns = 0; };

		std::string name;
		// ambient
		Vector3 ka;
		// diffuse
		Vector3 kd;
		// specular
		Vector3 ks;
		// specular exponent
		float ns;
	};

	/*struct Vertex
	{
		Vertex() { position = Vector3(); normal = Vector3(); tex_coord = Vector3(); };
		Vertex(Vector3 position_, Vector3 normal_, Vector3 tex_coord_) { position = position_; normal = normal_; tex_coord = tex_coord_; };
		Vector3 position;
		Vector3 normal;
		Vector3 tex_coord;
	};*/

	// This holds the index into the array for either vertices, normals, or textures for each face.
	struct ObjFaceIndeces {
		int index[3];

		int& operator[](int i) {
			return index[i];
		}
	};

	struct pointIndeces {
		const unsigned int vert_index;
		const unsigned int normal_index;
		const unsigned int texture_index;

		bool operator==(const pointIndeces& other) const {
			return (vert_index == other.vert_index && normal_index == other.normal_index && texture_index == other.texture_index);
		}
	};

} // namespace rc

// Specialize std::hash for pointIndeces
namespace std {
	template <> struct hash<rc::pointIndeces> {
		size_t operator()(const rc::pointIndeces& point) const {
			return std::hash<unsigned int>()(point.vert_index) ^ std::hash<unsigned int>()(point.normal_index) ^ std::hash<unsigned int>()(point.texture_index);
		}
	};
}

namespace rc {
	// This holds the data in the obj file to be converted into a glMesh. 
	class ObjMesh
	{
	private:

	public:
		ObjMesh() { group_name = ""; };
		std::string group_name;
		std::vector<Vector3> vertices;
		std::vector<Vector3> normals;
		std::vector<Vector3> tex_coords;
		std::vector<ObjFaceIndeces> vertex_faces;
		std::vector<ObjFaceIndeces> normal_faces;
		std::vector<ObjFaceIndeces> tex_coord_faces;

		const int NV() const { return vertices.size(); };
		const int NF() const { return vertex_faces.size(); };

		const ObjFaceIndeces& F(const int i) const { return vertex_faces[i]; };
		const ObjFaceIndeces& FN(const int i) const { return normal_faces[i]; };
		const ObjFaceIndeces& FT(const int i) const { return tex_coord_faces[i]; };

		const Vector3& V(const int i) const { return vertices[i]; };
		const Vector3& VN(const int i) const { return normals[i]; };
		const Vector3& VT(const int i) const { return tex_coords[i]; };
	};


	// Class for a mesh in OpenGL. Each vertex, normal, and position index corresponds to the 
	// same vertice in the mesh.

	// Might be better to create an obj to glMesh class that takes an objMesh and converts it to a glMesh
	// This way we can keep the GlMesh class simple and not have to worry about conversion functions
	class GLMesh
	{
	private:
		std::vector<Vector3> _vertices;
		std::vector<Vector3> _normals;
		std::vector<Vector3> _tex_coords;
		std::vector<int> _elements;

		void _add_new_vtn_and_element(
			const Vector3& v,
			const Vector3& n,
			const Vector3& t,
			const int element)
		{
			_vertices.push_back(v);
			_normals.push_back(n);
			_tex_coords.push_back(t);
			_elements.push_back(element);
		};

		void _add_vtn_values_or_index(const ObjMesh& mesh,
		std::unordered_map<pointIndeces, int>& locations,
		const pointIndeces& point)
		{
			// check if the hash exists and if not add it
			size_t new_index = _vertices.size();
			std::pair<std::unordered_map<pointIndeces, int>::iterator, bool> result =
				locations.emplace(point, new_index);
			
			if (result.second) {
				const Vector3& v = mesh.V(point.vert_index - 1);
				const Vector3& n = mesh.VN(point.normal_index - 1);
				const Vector3& t = mesh.VT(point.texture_index - 1);
				_add_new_vtn_and_element(v, n, t, new_index);
			}
			else {
				int existingElementIndex = result.first->second;
				_elements.push_back(existingElementIndex);
			}
		};

		void _create_faces_and_add_values(
			const ObjMesh& mesh,
			std::unordered_map<pointIndeces, int>& locations,
			const int i)
			{
			ObjFaceIndeces face = mesh.F(i);
			ObjFaceIndeces normal_face = mesh.FN(i);
			ObjFaceIndeces texture_face = mesh.FT(i);
			for (int j = 0; j < 3; j++) {
				pointIndeces point = { 
					face.index[j],
					normal_face.index[j],
					texture_face.index[j]
				};
				_add_vtn_values_or_index(mesh, locations, point);
			}
		};

	public:
		void transformObjToGL(const ObjMesh& mesh) {
			std::unordered_map<pointIndeces, int> locations;
			locations.reserve(3 * mesh.NV());

			const unsigned int n_faces = mesh.NF();

			for (int i = 0; i < n_faces; ++i) {
				_create_faces_and_add_values(mesh, locations, i);
			}
		};

		const std::vector<Vector3>& getVertices() const { return _vertices; };
		const std::vector<Vector3>& getNormals() const { return _normals; };
		const std::vector<Vector3>& getTexCoords() const { return _tex_coords; };
		const std::vector<int>& getElements() const { return _elements; };

		const int NV() const { return _vertices.size(); };
		const int NE() const { return _elements.size(); };
		const int NN() const { return _normals.size(); };
		const int NT() const { return _tex_coords.size();
		};
	};
}
#endif // OBJ_FOR_GL_HPP