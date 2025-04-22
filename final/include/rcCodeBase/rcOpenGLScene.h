#ifndef RC_OPENGL_SCENE_HPP
#define RC_OPENGL_SCENE_HPP

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "../cyCodeBase/cyVector.h"
#include "../cyCodeBase/cyGL.h"
#include "rcCamera.h"
#include "rcObjModifier.h"

namespace rc
{
class GLScene
{
protected:
	cy::Matrix4f mvp = cy::Matrix4f::Identity();
	cy::Matrix4f mv_points = cy::Matrix4f::Identity();
	cy::Matrix3f mv_normals = cy::Matrix3f::Identity();
	rcTriMeshForGL* _mesh;
public:
	GLScene() : _mesh(nullptr) {}

	cy::Matrix4f const & MVP() const { return mvp; }
	cy::Matrix4f& MVP() { return mvp; }
	cy::Matrix4f const & MV_points() const { return mv_points; }
	cy::Matrix4f& MV_points() { return mv_points; }
	cy::Matrix3f const & MV_normals() const { return mv_normals; }
	cy::Matrix3f& MV_normals() { return mv_normals; }

	GLfloat bg[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const char* window_name = "Window Name";
	cy::GLSLProgram program;
	float FOV = 90.0f;
	float previous_frame_time = 0.0f;
	rc::Camera camera;
	cy::Vec2f mouse_position = cy::Vec2f(0, 0);
	
	cy::Matrix4f point_transform = cy::Matrix4f::Identity();

	unsigned int n_elements = 0;
	unsigned int n_points = 0;

	void set_mesh(rcTriMeshForGL* mesh) { _mesh = mesh; }
	rcTriMeshForGL* get_mesh() { return _mesh; }

	void set_mvp()
	{
		mvp = camera.projection_matrix * camera.view_matrix * point_transform;
		mv_points = camera.view_matrix * point_transform;
		mv_normals = mv_points.GetSubMatrix3();
		mv_normals.Invert();
		mv_normals.Transpose();
	}

	void set_mvp_and_update_uniforms()
	{
		set_mvp();
		program.SetUniform("mvp", mvp);
		program.SetUniform("mv_points", mv_points);
		program.SetUniform("mv_normals", mv_normals);
	}

	void position_in_mv(cy::Vec3f& position)
	{
		position = cy::Vec3f(mv_points * position);
	}

};
}; // namespace rc
#endif // !RC_OPENGL_SCENE_HPP