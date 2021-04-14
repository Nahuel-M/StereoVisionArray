#pragma once
#include <string>
#include <opencv2/core.hpp>

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

void showImage(std::string name, cv::Mat image, double multiplier = 1, bool hold = true, float scale = 0.3);

void showDifference(std::string name, cv::Mat image1, cv::Mat image2, double multiplier = 1);

cv::Mat getIdealRef();

void saveImage(std::string filename, cv::Mat image);

cv::Mat loadImage(std::string filename);
