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
std::vector<cv::Mat> faceNormals;
std::vector<Camera> cameras;
cv::Mat mask;
cv::Mat disparityRef; 
cv::Mat K, D;

using namespace cv;

int main()
{

	/// Images
	getImages(images, "Renders3", 1);
	//getCameraIntrinsicParameters("..\\CameraCalibration\\calibration\\CalibrationFile", K, D);
	//undistortImages(images, K, D);
	blurImages(images, 3);

	/// Cameras
	//getCameras(cameras, images.back().size(), 6e-3, 0, 1.55e-6);
	getCameras(cameras, images.back().size());

	/// Mask
	std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images[12]);
	//std::vector<cv::Point2i> maskPoints{ {1,1}, {images.back().rows-1,1}, {images.back().rows-1,images.back().cols-1}, {1,images.back().cols-1} };
	mask = drawMask(images.back(), maskPoints);

	//Rect roi = Rect{ Point2i(337*4, 137*4), Point2i(914*4, 561*4) };
	//Rect roi = Rect{ Point2i(500, 0), Point2i(3800, 2800) };
	Rect roi = Rect{ Point2i(878,421), Point2i(1918, 1716) };

	int disp12MaxDiff = 3;
	int preFilterCap = 4;//PreFilterCap			4
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range
	int P1 = 5;
	int P2 = 192;
	StereoSGBMImpl2 sgmInitial = StereoSGBMImpl2(220, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);
	Mat disparity = getCrossSGM(12, sgmInitial, false);
	//showImage("disparity", disparity - 3900, 70); //4800
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, false) << std::endl;

	Mat roiMask = mask(roi);
	Mat depth = disparity2Depth(disparity, cameras[12], cameras[13])(roi);
	Mat normals = depth2Normals(depth, roiMask, cameras[12]);
	for (int i = 0; i < cameras.size(); i++)
	{
		faceNormals.push_back(getOrthogonalityFromCamera(depth, roiMask, normals, cameras[12], cameras[i]));
	}

	Mat oneOverother = faceNormals[14] / (faceNormals[14] + faceNormals[10]);
	showImage("faceNormals[14]", faceNormals[14]);
	Mat oneOverother2 = faceNormals[10] / (faceNormals[10] + faceNormals[14]);
	showImage("faceNormals[10]", faceNormals[10], 1, true);

	StereoArraySGBM sgbm = StereoArraySGBM(220, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);
	Rect area{ Point2i{887,480}, Point2i{1923,1632} };
	sgbm.compute(images, faceNormals, area, Size{ 5,5 }, 12, disparity);
	showImage("disparity", disparity - 3900, 70); //4800
	Mat disparityHolder{ images[0].size(), CV_16SC1 };
	disparity.copyTo(disparityHolder(area));
	std::cout << getAvgDiffWithAbsoluteReference(disparityHolder, true) << std::endl;

	//exportOBJfromDisparity(disparity, "disparityTest.obj", cameras[0], cameras[1], 0.5);

	//plotNoseBridge(combination, "Combination");
	//plotNoseBridge(disparityRef, "Reference");
	//showPlot();



}

