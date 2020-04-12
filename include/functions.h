#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

enum pairType {
    ORTHOGONAL,
    DIAGONAL,
    TO_CENTER,
    LINE_HORIZONTAL,
    LINE_VERTICAL,
    CROSS
};
class Camera;

std::vector<std::array<int, 2>> getCameraPairs(std::vector<Camera>& cameras, pairType pairs);

void showImage(std::string name, cv::Mat image);

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

std::vector<cv::Point2i> bresenham(cv::Point2i point1, cv::Point2i point2);
