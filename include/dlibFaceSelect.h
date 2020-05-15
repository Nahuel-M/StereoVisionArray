#pragma once
#include <opencv2/core.hpp>


cv::Mat getFaceMask(cv::Mat& sampleImage);

cv::Mat getFaceCircle(cv::Mat& image);