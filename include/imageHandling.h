#pragma once
#include <string>
#include <opencv2/core.hpp>

void showImage(std::string name, cv::Mat image);

cv::Mat getIdealRef();

void saveImage(std::string filename, cv::Mat image);

cv::Mat loadImage(std::string filename);
