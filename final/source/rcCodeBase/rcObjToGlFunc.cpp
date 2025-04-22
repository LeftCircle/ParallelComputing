#include "rcObjToGlFunc.h"

GlElementArrayData transformObjToGL(const cy::TriMesh &mesh){
	GlElementArrayData output;
	std::unordered_map<pointIndeces,int> locations;
	locations.reserve(3 * mesh.NV());

	const unsigned int n_faces = mesh.NF();

	for (int i = 0; i < n_faces; ++i) {
		cy::TriMesh::TriFace face = mesh.F(i);
		cy::TriMesh::TriFace normal_face = mesh.FN(i);
		cy::TriMesh::TriFace texture_face = mesh.FT(i);
		for (int j = 0; j < 3; j++){
			pointIndeces point = {face.v[j], normal_face.v[j], texture_face.v[j]};
			// check if the hash exists and if not add it
			size_t new_index = output._v_vbo.size();
			auto result = locations.emplace(point, new_index);
			if (result.second) {
				// a new unique combination of v, n, t exists. 
				output._v_vbo.push_back(mesh.V(face.v[j]));
				output._vn_vbo.push_back(mesh.VN(normal_face.v[j]));
				output._vt_vbo.push_back(mesh.VT(texture_face.v[j]));
				output._elements.push_back(new_index);
			} else {
				int existingElementIndex = result.first->second;
				output._elements.push_back(existingElementIndex);
			}
		}
	}

	return output;
}