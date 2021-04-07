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
	getImages(images, "Photographs\\Series1Test", 1);
	getCameraIntrinsicParameters("..\\CameraCalibration\\calibration\\CalibrationFile", K, D);
	undistortImages(images, K, D);

	/// Cameras
	getCameras(cameras, images.back().size(), 6e-3, 0, 1.55e-6);

	/// Mask
	//std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images[12]);
	std::vector<cv::Point2i> maskPoints{ {1,1}, {images.back().rows-1,1}, {images.back().rows-1,images.back().cols-1}, {1,images.back().cols-1} };
	mask = drawMask(images.back(), maskPoints);


	std::vector<std::array<int, 2>> pairs = getCameraPairs(cameras, MID_RIGHT);
	Rect roi = Rect{ Point2i(437*4, 237*4), Point2i(914*4, 461*4) };
	cv::Mat rightDisp;

	int disp12MaxDiff = 3;
	int preFilterCap = 4;//PreFilterCap			4
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 100;	// Speckle window size	100
	int sRange = 5;	// Speckle range
	int P1 = 16;
	int P2 = 192;
	pairs = getCameraPairs(cameras, MID_RIGHT);
	Mat disparity;
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(250, 48, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
	Mat im1 = images[pairs[0][0]];
	Mat im2 = images[pairs[0][1]];
	//showImage("im1", im1(roi));
	//showImage("im2", im2(roi));
	Mat im1n, im2n;
	blur(im1, im1n, Size{ 6, 6 });
	blur(im2, im2n, Size{ 6, 6 });

	sgbm->compute(im2n(roi), im1n(roi), disparity);
	showImage("disparity2", disparity - 4150, 60, true);



	//exportOBJfromDisparity(disparity, "disparityTest.obj", cameras[0], cameras[1], 0.5);

	//cv::Mat topDisp = getDisparityFromPairSGM(pairs[0]);
	//showImage("topDisparity", topDisp, 5, false);

	//pairs = getCameraPairs(cameras, MID_LEFT);
	//cv::Mat leftDisp = getDisparityFromPairSGM(pairs[0]);
	//showImage("leftDisp", leftDisp, 5, false);

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

