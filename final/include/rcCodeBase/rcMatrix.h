#ifndef RC_MATRIX_H
#define RC_MATRIX_H

// an extension of the cyMatrix class to add some new functions
#include "../cyCodeBase/cyMatrix.h"

namespace rc
{

template <typename T>
class Matrix4 : public cy::Matrix4<T>
{
public:
// Create a 4x4 matrix that converts from polar coordinates to cartesian 
// coordinates. From https://math.stackexchange.com/questions/162177/building-transformation-matrix-from-spherical-to-cartesian-coordinate-system
void SetSphericalToCartensian( T theta, T phi, T r )
{
	T sinTheta = sin(theta);
	T cosTheta = cos(theta);
	T sinPhi = sin(phi);
	T cosPhi = cos(phi);

	T sinThetaSinPhi = sinTheta * sinPhi;
	T sinThetaCosPhi = sinTheta * cosPhi;
	T cosThetaCosPhi = cosTheta * cosPhi;
	T cosThetaSinPhi = cosTheta * sinPhi;

	this->Column(0).Set(sinThetaCosPhi, sinThetaSinPhi, cosTheta, 0);
	this->Column(1).Set(r * cosThetaCosPhi, r * cosThetaSinPhi, -r * sinTheta, 0);
	this->Column(2).Set(-r * sinThetaSinPhi, r * sinThetaCosPhi, 0, 0);
	this->Column(3).Set(0, 0, 0, 1);

}
}; // class Matrix4
};
#endif