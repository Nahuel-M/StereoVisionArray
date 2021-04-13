#pragma once
#include <iostream>
#include <iterator>
#include <vector>
#include <opencv2/core.hpp>

#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"
#include "plotting.h"

namespace plt = matplotlibcpp;

std::vector<cv::Mat> images;
std::vector<Camera> cameras;
cv::Mat mask;
cv::Mat disparityRef; 
cv::Mat K, D;

using namespace cv;

int main()
{

	std::vector<double> heat;
	std::vector<double> err;

	/// Images
	getImages(images, "Renders2", 1);
	//getCameraIntrinsicParameters("..\\CameraCalibration\\calibration\\CalibrationFile", K, D);
	//undistortImages(images, K, D);
	blurImages(images, 3);

	/// Cameras
	//double f = 0.05;
	//double sensor_size = 0.036;
	//getCameras(cameras, images.back().size(), 6e-3, 0, 1.55e-6);
	getCameras(cameras, images.back().size());

	/// Mask
	//std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images[12]);
	std::vector<cv::Point2i> maskPoints{ {1,1}, {images.back().rows-1,1}, {images.back().rows-1,images.back().cols-1}, {1,images.back().cols-1} };
	mask = drawMask(images.back(), maskPoints);


	std::vector<std::array<int, 2>> pairs;
	//Rect roi = Rect{ Point2i(337*4, 137*4), Point2i(914*4, 561*4) };
	Rect roi = Rect{ Point2i(500, 0), Point2i(3800, 2800) };

	int disp12MaxDiff = 3;
	int preFilterCap = 4;//PreFilterCap			4
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range
	int P1 = 5;
	int P2 = 192;
	StereoSGBMImpl2 sgbm = StereoSGBMImpl2(275, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);

	//Ptr<StereoSGBM> sgbm = StereoSGBM::create(250, 48, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);

	for(int i = 0; i < 15; i++)
	{
		Mat disparity = getCrossSGM(12, sgbm, false);
		//blurImages(images, 3);
		//showImage("disparity2", disparity - 4800, 70, false);
		std::cout << getAvgDiffWithAbsoluteReference(disparity, false) << std::endl;
	}

	//disparity = getCrossSGM(17, sgbm, true);



	//exportOBJfromDisparity(disparity, "disparityTest.obj", cameras[0], cameras[1], 0.5);


	//Mat deltaH = getBlurredSlope(images[12], 0);
	//Mat deltaV = getBlurredSlope(images[12], 1);
	//Mat nDH = deltaH / (deltaH + deltaV);
	//Mat nDV = 1 - nDH;

	//multiply(rightDisp, nDH, rightDisp, 1, rightDisp.type());
	//multiply(leftDisp, nDH, leftDisp, 1, rightDisp.type());
	//multiply(topDisp, nDV, topDisp, 1, rightDisp.type());
	//multiply(bottomDisp, nDV, bottomDisp, 1, rightDisp.type());

	//Mat combination = (rightDisp + topDisp + leftDisp + bottomDisp) / 2;
	//plotNoseBridge(combination, "Combination");
	//plotNoseBridge(disparityRef, "Reference");
	//showPlot();

	//showImage("nDH", nDH);
	//showImage("nDV", nDV);


}

