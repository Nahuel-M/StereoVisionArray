#include "Camera.h"
#include <iostream>

using namespace cv;

Camera::Camera(double focal_length, Point3d position, double pixel_size):
	f{ focal_length }, pos3D{position}, pixel_size{ pixel_size }
{
}

Camera::~Camera()
{
}

Point2i Camera::project(Point3d Pos3D)
{
	Point2i pixel;
	float mult = f / (Pos3D.z - this->pos3D.z) / pixel_size;
	pixel.x = int( (Pos3D.x - this->pos3D.x) * mult );
	pixel.y = int( (Pos3D.y - this->pos3D.y) * mult );
	return pixel;
}


Point3d Camera::inv_project(Point2i pixel)
{
	Point3d vector{
		pixel.x * pixel_size / f,
		pixel.y * pixel_size / f,
		1
	};
	return vector;

}