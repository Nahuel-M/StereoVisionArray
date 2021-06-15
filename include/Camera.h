#pragma once
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <vector>
#include <array>
#include <opencv2/core.hpp>
#pragma warning (pop)


class Camera
{
public:
	Camera(double focal_length, double _fx, double _fy, cv::Point3d position, double pixel_size, cv::Point2i principalPoint);
	Camera(double focal_length, cv::Point3d position, double pixel_size);
	~Camera();

	cv::Point3d pos3D;
	cv::Point2i project(cv::Point3d Pos3D);
	cv::Point3d inv_project(cv::Point2i pixel);

	double f;
	double fx;
	double fy;
	double pixelSize;
	cv::Point2i principalPoint;

private:

};
