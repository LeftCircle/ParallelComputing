#ifndef RC_OBJ_TO_GL_FUNC_H
#define RC_OBJ_TO_GL_FUNC_H
#include <unordered_map>

#include "cyTriMesh.h"

struct GlElementArrayData {
    std::vector<cy::Vec3f> _v_vbo;
    std::vector<cy::Vec3f> _vn_vbo;
    std::vector<cy::Vec3f> _vt_vbo;
	std::vector<int> _elements;
};

struct FaceData {
    const cy::Vec3f &vert;
    const unsigned int *normal;
    const unsigned int *texture;
};


struct pointIndeces {
    const unsigned int vert_index;
    const unsigned int normal_index;
    const unsigned int texture_index;

    bool operator==(const pointIndeces &other) const {
        return (vert_index == other.vert_index && normal_index == other.normal_index && texture_index == other.texture_index);
    }
};

namespace std {
    template <> struct hash<pointIndeces> {
        size_t operator()(const pointIndeces& point) const {
            return std::hash<unsigned int>()(point.vert_index) ^ std::hash<unsigned int>()(point.normal_index) ^ std::hash<unsigned int>()(point.texture_index);
        }
    };
}

GlElementArrayData transformObjToGL(const cy::TriMesh &mesh);

#endif