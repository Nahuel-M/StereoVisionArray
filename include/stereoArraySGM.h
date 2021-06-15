#pragma once
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Camera.h"
#pragma warning (pop)

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

template <typename printableVector>
void printV(printableVector input);


//std::vector<float> calcDisparityCostForPlotting(std::vector<cv::Mat> images, std::vector<Camera> cameras, int camID,
//	int x, int y, int minD, int maxD);

std::vector<float> calcPixelArrayCost(std::vector<cv::Mat>& images, std::vector<Camera> cams, Camera centerCam,
	int minD, int maxD, cv::Point2i position, int camera = -1);

std::vector<float> calcPixelArrayIntensity(cv::Mat& image, Camera cam, Camera centerCam,
	int minD, int maxD, cv::Point2i position);

uchar getCamIntensities(PixType* imPointer,
	int x, int y, int minD, int maxD,
	cv::Point2f disparityStep, int width, int height, std::array<PixType, 100>& intensities);



struct StereoArraySGMParams
{
	StereoArraySGMParams(int _minDisparity, int _numDisparities,
		int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
		int _uniquenessRatio, int _speckleWindowSize, int _speckleRange)
	{
		minDisparity = _minDisparity;
		numDisparities = _numDisparities;
		P1 = _P1;
		P2 = _P2;
		disp12MaxDiff = _disp12MaxDiff;
		preFilterCap = _preFilterCap;
		uniquenessRatio = _uniquenessRatio;
		speckleWindowSize = _speckleWindowSize;
		speckleRange = _speckleRange;
	}

	int minDisparity;
	int numDisparities;
	int preFilterCap;
	int uniquenessRatio;
	int P1;
	int P2;
	int speckleWindowSize;
	int speckleRange;
	int disp12MaxDiff;
};

class StereoArraySGBM
{
public:
	StereoArraySGBM(int _minDisparity, int _numDisparities,
		int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
		int _uniquenessRatio, int _speckleWindowSize, int _speckleRange);

	void compute(std::vector<cv::Mat>& LBPs, std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Camera> cams,
		int centerCamID, cv::Rect area, cv::OutputArray disparr, double camDistance=0.05, 
		cv::Mat mask = cv::Mat{}, std::vector<int> usedCams = std::vector<int>{});

	void computeMinCost(std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Camera> cams,
		int centerCamID, cv::Rect area, cv::OutputArray disparr, double camDistance=0.05, std::vector<int> usedCams = std::vector<int>{});

	int getMinDisparity() const { return params.minDisparity; }
	void setMinDisparity(int minDisparity) { params.minDisparity = minDisparity; }

	int getNumDisparities() const { return params.numDisparities; }
	void setNumDisparities(int numDisparities) { params.numDisparities = numDisparities; }

	int getSpeckleWindowSize() const { return params.speckleWindowSize; }
	void setSpeckleWindowSize(int speckleWindowSize) { params.speckleWindowSize = speckleWindowSize; }

	int getSpeckleRange() const { return params.speckleRange; }
	void setSpeckleRange(int speckleRange) { params.speckleRange = speckleRange; }

	int getDisp12MaxDiff() const { return params.disp12MaxDiff; }
	void setDisp12MaxDiff(int disp12MaxDiff) { params.disp12MaxDiff = disp12MaxDiff; }

	int getPreFilterCap() const { return params.preFilterCap; }
	void setPreFilterCap(int preFilterCap) { params.preFilterCap = preFilterCap; }

	int getUniquenessRatio() const { return params.uniquenessRatio; }
	void setUniquenessRatio(int uniquenessRatio) { params.uniquenessRatio = uniquenessRatio; }

	int getP1() const { return params.P1; }
	void setP1(int P1) { params.P1 = P1; }

	int getP2() const { return params.P2; }
	void setP2(int P2) { params.P2 = P2; }

	StereoArraySGMParams params;
	cv::Mat buffer;

};
