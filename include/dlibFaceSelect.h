#pragma once
#include <opencv2/core.hpp>
#include "Camera.h"

cv::Mat getFaceMask(cv::Mat& sampleImage);

cv::Mat drawMask(cv::Mat& sampleImage, std::vector<cv::Point2i>& PointIndices);

std::vector<cv::Point2i> getFaceMaskPoints(cv::Mat& sampleImage);

std::vector<cv::Point3d> Points2DtoPoints3D(cv::Mat& depth, std::vector<cv::Point2i>& PointIndices, Camera cam);

std::vector<cv::Point2i> Points3DtoPoints2D(std::vector<cv::Point3d>& PointIndices, Camera cam);

cv::Mat getFaceCircle(cv::Mat& image);