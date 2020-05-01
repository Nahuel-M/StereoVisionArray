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

cv::Mat shiftPerspective(Camera inputCam, Camera outputCam, cv::Mat depthMap);

cv::Mat shiftPerspective2(Camera inputCam, Camera outputCam, cv::Mat depthMap);

std::vector<std::vector<std::array<int, 2>>> getGroups(std::vector<Camera>& cameras, std::string groupType);

cv::Mat Points3DToDepthMap(std::vector<cv::Point3d>& points, Camera camera, cv::Size resolution);

std::vector<cv::Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, cv::Size resolution);

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pairs);

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair, int cameraNum);

double getAbsDiff(cv::Mat& mat1, cv::Mat& mat2);


void showImage(std::string name, cv::Mat& image);

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

std::vector<cv::Point2i> bresenham(cv::Point2i point1, cv::Point2i point2);

cv::Mat getIdealRef();

void saveImage(std::string filename, cv::Mat image);

cv::Mat loadImage(std::string filename);

double calculateAverageError(cv::Mat &image);
