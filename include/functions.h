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
    CROSS,
    JUMP_CROSS,
    TO_CENTER_SMALL,
    MID_LEFT,
    MID_TOP
};
class Camera;

std::vector<std::array<int, 2>> getCameraPairs(std::vector<Camera>& cameras, pairType pairs);

double getAbsDiff(cv::Mat& mat1, cv::Mat& mat2);


void showImage(std::string name, cv::Mat image);

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

std::vector<cv::Point2i> bresenham(cv::Point2i point1, cv::Point2i point2);

cv::Mat getIdealRef();

void saveImage(std::string filename, cv::Mat image);

cv::Mat loadImage(std::string filename);

double calculateAverageError(cv::Mat &image);
