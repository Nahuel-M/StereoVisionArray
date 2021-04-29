#pragma once
#include <opencv2/core.hpp>

std::vector<cv::Point2i> bresenham(cv::Point2i point1, cv::Point2i point2);

std::vector<cv::Point2i> bresenhamDxDy(cv::Point2i point1, cv::Point2f delta, cv::Size imageSize);