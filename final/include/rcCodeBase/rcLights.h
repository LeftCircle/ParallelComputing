#ifndef RC_LIGHTS_HPP
#define RC_LIGHTS_HPP

#include "../cyCodeBase/cyVector.h"
#include "../rcCodeBase/rcMatrix.h"
#include "rcCore.h"


namespace rc
{

	class Light
	{
	protected:
		cy::Vec3f _position = cy::Vec3f(0, 0, 0);
		cy::Vec3f _color = cy::Vec3f(1, 1, 1);
		float _intensity = 1.0f;
		cy::Matrix4f _rotation = cy::Matrix4f::Identity();

	public:
		virtual void SetRotat(cy::Matrix4f const& rotation) { _rotation = rotation; }
		Light() {};
		Light(cy::Vec3f position, cy::Vec3f color, float intensity)
		{
			_position = position;
			_color = color;
			_intensity = intensity;
		}
		~Light() {};

		cy::Vec3f const& position() const { return _position; }
		cy::Vec3f& position() { return _position; }
		cy::Vec3f const& color() const { return _color; }
		cy::Vec3f& color() { return _color; }
		float const& intensity() const { return _intensity; }
		float& intensity() { return _intensity; }
	};

	class DirectionalLight : public Light
	{
	protected:
		cy::Vec3f _direction = cy::Vec3f(0, -1, 0);
		cy::Vec3f _ambient_intensity= cy::Vec3f(0.1f, 0.1f, 0.1f);
		cy::Vec3f _diffuse_intensity= cy::Vec3f(0.5f, 0.5f, 0.5f);
		cy::Vec3f _specular_intensity= cy::Vec3f(0.5f, 0.5f, 0.5f);

	public:
		DirectionalLight() {};
		DirectionalLight(cy::Vec3f position, cy::Vec3f color, float intensity, cy::Vec3f direction) : Light(position, color, intensity)
		{
			_direction = direction;
		}
		~DirectionalLight() {};

		void set_direction(cy::Vec3f direction) { _direction = direction; }
		cy::Vec3f const& direction() const { return _direction; }
		cy::Vec3f& direction() { return _direction; }
		
		// Intensities
		void set_ambient_intensity(cy::Vec3f const& intensity) { _ambient_intensity = intensity; }
		void set_diffuse_intensity(cy::Vec3f const& intensity) { _diffuse_intensity = intensity; }
		void set_specular_intensity(cy::Vec3f const& intensity) { _specular_intensity = intensity; }
		cy::Vec3f const& ambient_intensity() const { return _ambient_intensity; }
		cy::Vec3f const& diffuse_intensity() const { return _diffuse_intensity; }
		cy::Vec3f const& specular_intensity() const { return _specular_intensity; }
		cy::Vec3f& ambient_intensity() { return _ambient_intensity; }
		cy::Vec3f& diffuse_intensity() { return _diffuse_intensity; }
		cy::Vec3f& specular_intensity() { return _specular_intensity; }
		
		//void rotate_direction(float theta, float phi)
		//{
		//	cy::Matrix4f rot_p_general = cy::Matrix4f::Rotation(cy::Vec3f(0, 0, 1), DEG2RAD(theta));
		//	//cy::Matrix4f rot_t_general = cy::Matrix4f::Rotation(cy::Vec3f(0, 1, 0), DEG2RAD(phi));
		//	cy::Matrix4f rot_t_general = cy::Matrix4f::Identity();
		//	_direction = cy::Vec3f(rot_t_general * rot_p_general * _direction);
		//}
	};

	class SphericalDirectionalLight : public DirectionalLight
	{
		// Like a directional light but the direction is determined by the vector from the spherical 
		// position to the origin
	private:
		cy::Vec3f _origin = cy::Vec3f(0.0f);
		float theta_mod = 0.0f;
		bool rotating = false;
		cy::Vec3f _spherical_naught = cy::Vec3f(0.0f, 0.0f, 1.0f);
		cy::Vec3f _delta_spherical = cy::Vec3f(0.0f, 0.0f, 0.0f);
		cy::Vec3f _spherical_pos = cy::Vec3f(0.0f, 0.0f, 1.0f);

		void _spherical_to_cartesian(cy::Vec3f& spherical)
		{
			float t = DEG2RAD(spherical[0]);
			float p = DEG2RAD(spherical[1]);
			float r = spherical[2];
			spherical.Set(r * sin(t) * cos(p), r * sin(t) * sin(p), r * cos(p));
		}

	public:
		SphericalDirectionalLight() {};
		SphericalDirectionalLight(float t, float p) { _spherical_pos.Set(t, p, 1.0f); calculate_direction(t, p); }

		void set_spherical_position(float t, float p) { _spherical_pos.Set(t, p, 1.0f); calculate_direction(t, p); }
		virtual void SetRotation(cy::Matrix4f const& rotation) 
		{
			_rotation = rotation; 
			_direction = cy::Vec3f(_rotation * _direction);

		}


		cy::Vec3f spherical_to_cartesian(float t, float p)
		{
			t = DEG2RAD(t);
			p = DEG2RAD(p);
			return cy::Vec3f(sin(p) * cos(t), sin(t) * sin(p), cos(p));
		}

		void calculate_direction(float theta, float phi)
		{
			// direction is normalized vector from the spherical position to the origin. 
			cy::Vec3f spherical_pos = spherical_to_cartesian(theta, phi) + _origin;
			_direction = cy::Vec3f(_rotation * (_origin - spherical_pos).GetNormalized());
		}

		void start_rotating()
		{
			_delta_spherical.Set(0.0f, 0.0f, 0.0f);
			_spherical_naught.Set(_spherical_pos[0], _spherical_pos[1], _spherical_pos[2]);
			rotating = true;
		}

		void stop_rotating() 
		{
			rotating = false; 
			cy::Vec3f new_sphere = _spherical_naught + _delta_spherical;
			_spherical_pos.Set(new_sphere[0], new_sphere[1], new_sphere[2]);
		}

		cy::Vec3f add_rotation_and_get_direction(float dt, float dp)
		{
			_delta_spherical += cy::Vec3f(dt, dp, 0.0f);
			cy::Vec3f new_sphere = _spherical_naught + _delta_spherical;
			
			calculate_direction(new_sphere[0], new_sphere[1]);
			return cy::Vec3f(_direction);
		}

	};
}; // namespace rc

#endif