#ifndef RC_CAMERA_HPP
#define RC_CAMERA_HPP

#include "../cyCodeBase/cyMatrix.h"
#include "rcCore.h"
#include <iostream>


namespace rc{

class Camera
{
private:
	cy::Vec3f _og_up = cy::Vec3f(0, 1, 0);
public:
	cy::Vec3f position = cy::Vec3f(0, 0, 60);
	cy::Vec3f target = cy::Vec3f(0, 0, 0);
	cy::Matrix4f view_matrix;
	cy::Matrix4f projection_matrix;
	float fov;

	cy::Vec3f get_up_vector() { return cy::Vec3f(view_matrix.cell[1], view_matrix.cell[5], view_matrix.cell[9]); };
	cy::Vec3f get_tangent_vector() { return cy::Vec3f(view_matrix.cell[0], view_matrix.cell[4], view_matrix.cell[8]); };

	void set_perspective_projection(float new_fov, float aspect, float znear, float zfar)
	{
		fov = new_fov;
		projection_matrix.SetPerspective(DEG2RAD(new_fov), aspect, znear, zfar);
	}

	void lookat(cy::Vec3f& pos, cy::Vec3f& targ_v, cy::Vec3f& up_v)
	{
		position = pos, target = targ_v;
		view_matrix.SetView(pos, targ_v, up_v);
		_og_up = up_v;
	}

	//! Rotates the position of the camera by theta about the up vector and phi about the tangent vector
	//! while continuing to look at the target. 
	void rotate(float t, float p)
	{
		cy::Vec3f old_up = get_up_vector();
		cy::Vec3f old_tangent = get_tangent_vector();

		cy::Matrix4f rot_p_general = cy::Matrix4f::Rotation(old_tangent, DEG2RAD(p));
		cy::Matrix4f rot_t_general = cy::Matrix4f::Rotation(old_up, DEG2RAD(t));

		//cy::Vec3f up = cy::Vec3f(rot_p_general * old_up);
		cy::Vec3f up = cy::Vec3f(rot_t_general * rot_p_general * old_up);
		position = cy::Vec3f(rot_t_general * rot_p_general * position);
		view_matrix.SetView(position, target, up);
	}
	void rotate_about_og_up(float t, float p)
	{
		// Rotate the position and up vector about the original up vector
		cy::Vec3f up = get_up_vector();
		cy::Vec3f tangent = get_tangent_vector();

		cy::Matrix4f rot_p_general = cy::Matrix4f::Rotation(tangent, DEG2RAD(p));
		cy::Matrix4f rot_t_general = cy::Matrix4f::Rotation(_og_up, DEG2RAD(t));

		position = cy::Vec3f(rot_t_general * rot_p_general * position);
		up = cy::Vec3f(rot_t_general * rot_p_general * up);
		view_matrix.SetView(position, target, up);
	}

	void rotate_other_position_about_og_up(cy::Vec3f& pos, float t, float p)
	{
		cy::Vec3f up = get_up_vector();
		cy::Vec3f tangent = get_tangent_vector();

		cy::Matrix4f rot_p_general = cy::Matrix4f::Rotation(tangent, DEG2RAD(p));
		cy::Matrix4f rot_t_general = cy::Matrix4f::Rotation(_og_up, DEG2RAD(t));

		pos = cy::Vec3f(rot_t_general * rot_p_general * position);
	}
	
	void zoom(float z)
	{
		cy::Vec3f direction = position - target;
		direction.Normalize();
		position += direction * z;
		view_matrix.SetView(position, target, get_up_vector());
	}
};

} // namespace rc
#endif