#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"

//#include "bresenham.h"
#include "imageHandling.h"

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
    MID_RIGHT,
    MID_TOP,
    MID_BOTTOM,
    LEFT_LEFTER,
    RIGHT_RIGHTER,
};
class Camera;

cv::Mat depth2Disparity(cv::Mat& depth, Camera camera1, Camera camera2);

cv::Mat disparity2Depth(cv::Mat& disparity, Camera camera1, Camera camera2);

cv::Mat getDisparityFromPair(std::vector<cv::Mat>& images, std::vector<Camera>& cameras, cv::Mat& mask, std::array<int, 2> pair);

cv::Mat getDisparityFromPair2(std::vector<cv::Mat>& images, std::vector<Camera>& cameras, cv::Mat& mask, std::array<int, 2> pair);

cv::Mat getDisparityFromPairSGM(cv::Mat& image1, cv::Mat& image2, cv::Mat& mask, Camera cam1, Camera cam2);

std::vector<Camera> getCameras(cv::Size resolution);

std::vector<cv::Mat> getImages(std::string folderName, double scale);

cv::Mat improveWithDisparity(cv::Mat& disparity, cv::Mat centerImage, std::vector<cv::Mat>& images, std::vector<std::array<Camera, 2>>& cameras, int windowSize);

cv::Mat improveWithDisparitySGM(cv::Mat& disparity, cv::Mat centerImage, std::vector<cv::Mat>& images, std::vector<std::array<Camera, 2>>& cameras, int windowSize);

cv::Mat shiftPerspective(Camera inputCam, Camera outputCam, cv::Mat &depth);

cv::Mat shiftDisparityPerspective(Camera inputCam, Camera outputCam, cv::Mat& disparity);

void fillHoles(cv::Mat& disparity, int filterSize);

cv::Mat shiftPerspectiveWithDisparity(Camera& inputCam, Camera& outputCam, cv::Mat& disparity, cv::Mat& image);

std::vector<std::vector<std::array<int, 2>>> getGroups(std::vector<Camera>& cameras, std::string groupType);

cv::Mat Points3DToDepthMap(std::vector<cv::Point3d>& points, Camera camera, cv::Size resolution);

std::vector<cv::Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, cv::Size resolution);

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pairs);

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair, int cameraNum);

int getAbsDiff(cv::Mat& mat1, cv::Mat& mat2);

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

double calculateAverageError(cv::Mat &image);

cv::Mat depth2Normals(cv::Mat& depth, cv::Mat& mask, Camera cam);

cv::Mat getOrthogonalityFromCamera(cv::Mat& disparity, cv::Mat& mask, Camera perspective, Camera stereo, Camera normals);

cv::Mat getPixelNormals(Camera& camera, cv::Mat& image);

cv::Mat matrixDot(cv::Mat& mat1, cv::Mat& mat2);

cv::Mat blurWithMask(cv::Mat& image, cv::Mat& mask, int filterSize);

cv::Mat getBlurredSlope(cv::Mat image, int vertical, int blurKernelSize = 41); //VARIABLE
