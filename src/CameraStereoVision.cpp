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


	std::vector<std::array<int, 2>> pairs;
	Rect roi = Rect{ Point2i(337*4, 137*4), Point2i(914*4, 561*4) };

	int disp12MaxDiff = 3;
	int preFilterCap = 4;//PreFilterCap			4
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 100;	// Speckle window size	100
	int sRange = 5;	// Speckle range
	int P1 = 16;
	int P2 = 192;
	pairs = getCameraPairs(cameras, MID_RIGHT);
	StereoSGBMImpl2 sgbm2 = StereoSGBMImpl2(250, 48, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(250, 48, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
	std::vector<Mat> imageVector;
	std::vector<Point2i> directions;
	pairs = getCameraPairs(cameras, CROSS);
	imageVector.push_back(images[pairs[0][0]](roi));
	for (auto p : pairs)
	{
		imageVector.push_back(images[p[1]](roi));
		Point3d dir = (cameras[p[1]].pos3D - cameras[p[0]].pos3D);
		directions.push_back(Point2i{ (dir.x > 0) - (dir.x < 0), (dir.y > 0) - (dir.y < 0) });
	}
	Mat disparity;
	sgbm->compute(imageVector[0], imageVector[1], disparity);
	showImage("disparity", disparity - 4150, 60, false);
	sgbm2.computeMultiCam(imageVector, directions, disparity);
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

