#pragma once
#include <vector>
#include <array>
#include <opencv2/core.hpp>


class Camera
{
public:
	Camera(double focal_length, cv::Point3d position, double pixel_size);
	Camera();
	~Camera();

	cv::Point3d pos3D;
	cv::Point2i project(cv::Point3d Pos3D);
	cv::Point3d inv_project(cv::Point2i pixel);

	double f;
	double pixelSize;

private:

};
