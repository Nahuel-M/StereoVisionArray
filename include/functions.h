#pragma once
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#pragma warning (pop)

#include "stereoArraySGM.h"
#include "stereosgbmMultiCam.h"

#include "imageHandling.h"
#include "Camera.h"
#include "plotting.h"

extern std::string imageFolder;
extern std::vector<cv::Mat> images;
extern std::vector<cv::Mat> faceNormals;
extern std::vector<Camera> cameras;
extern cv::Mat mask;
extern cv::Mat disparityRef;
extern float scale;
extern cv::Mat K, D;

enum class pairType {
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

void getCameras(std::vector<Camera>& cameras, std::string positionFilePath = "", double f = 0.05, double sensorSize = 0.036, double pixelSize = 0);

void getCameras(std::vector<Camera>& cameras, float baseline);

void getImages(std::vector<cv::Mat>& images, std::string folderName, double scale = 1);

cv::Mat Points3DToDepthMap(std::vector<cv::Point3d>& points, Camera camera, cv::Size resolution);

std::vector<cv::Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, cv::Point2i principalPoint, cv::Mat mask);

std::vector<int> getCrossIDs(int centerCameraID);

int getAbsDiff(cv::Mat& mat1, cv::Mat& mat2);

double calculateAverageError(cv::Mat &image);

cv::Mat depth2Normals(const cv::Mat& depth, Camera cam, int normalDistance=1, int bilateralF1=0, int bilateralF2=0, int gaussianF=0);

cv::Mat getOrthogonalityFromCamera(const cv::Mat& depth, cv::Mat mask, cv::Mat& normals, Camera perspective, Camera orthogonality);

cv::Mat getPixelNormals(Camera& camera, cv::Mat& image);

cv::Mat getOcclusion(const cv::Mat& depth, const cv::Mat& camAngles, Camera perspective, Camera occlusionCaster);

cv::Mat matrixDot(cv::Mat& mat1, cv::Mat& mat2);

cv::Mat blurWithMask(const cv::Mat& image, const cv::Mat& mask, int filterSize);

cv::Mat getBlurredSlope(cv::Mat image, bool vertical, int blurKernelSize = 41); //VARIABLE

void getCameraIntrinsicParameters(std::string filePath, cv::Mat& K, cv::Mat& D);

void undistortImages(std::vector<cv::Mat>& images, cv::Mat& K, cv::Mat& D, bool verbose = false);

void exportOBJfromDisparity(cv::Mat depthImage, std::string fileName, Camera cam1, Camera cam2, float scale = 1.f);

void exportXYZfromDisparity(cv::Mat disparityImage, std::string fileName, Camera cam, float camDistance, cv::Point2i principalPoint, float scale = 1.f, cv::Mat mask = cv::Mat{});

cv::Mat getDiffWithAbsoluteReference(cv::Mat disparity, cv::Rect area, bool verbose = false);

double getAvgDiffWithAbsoluteReference(cv::Mat disparity, cv::Rect area, bool verbose = false, std::string savePath = "");

void blurImages(std::vector<cv::Mat>& images, float blurSigma);

void noiseResistantLocalBinaryPattern(std::vector<cv::Mat>& images, ushort threshold);

void localBinaryPattern(std::vector<cv::Mat>& images, std::vector<cv::Mat>& LBPs, int distance = 1);

void normalizeMats(std::vector<cv::Mat>& images);

void subImgsAndCamsAndSurfs(std::vector<int> ids, std::vector<cv::Mat>& outputMats, std::vector<Camera>& outputCams, std::vector<cv::Mat>& outputSurfs);

void subImgsAndCams(std::vector<int> ids, std::vector<cv::Mat>& outputMats, std::vector<Camera>& outputCams);

cv::Mat getVectorMatsAverage(std::vector<cv::Mat>& mats);

void makeArrayCollage(std::vector<cv::Mat> images, cv::Size arrayShape, float multiplier = 1, float scale = 1);

void testSGM(cv::Mat disparity, std::vector<int> camIDs, int minD, int numD, cv::Rect area, cv::Mat groundTruth = cv::Mat{});

void calibrateCamerasFromOrigin(cv::Rect area, StereoArraySGBM& sgbm);

cv::Vec3f getBallScreenParams(cv::Mat image, int minSize, int maxSize);

cv::Mat generateBallDepthMap(cv::Size imageSize, cv::Vec3f ballScreenParameters, Camera camParameters, double ballRadius);

std::vector<float> movingAverage(std::vector<float>& inputVector, int windowSize);

std::vector<float> centeredMovingAverage(std::vector<float>& inputVector, int windowSize);

std::vector<float> centeredMovingAverageAbsoluteDeviation(std::vector<float>& inputVector, int windowSize);

std::vector<std::pair<float, float>> getSortedOrthogonalityDifference(cv::Mat depth, cv::Mat groundTruth, cv::Mat_<float> orthogonality, cv::Mat_<uchar> mask);