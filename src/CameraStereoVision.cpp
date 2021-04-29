#pragma once
#include <iostream>
#include <iterator>
#include <vector>
#include <opencv2/core.hpp>

#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"


namespace plt = matplotlibcpp;

std::vector<cv::Mat> images;
std::vector<cv::Mat> faceNormals;
std::vector<Camera> cameras;
cv::Mat mask;
cv::Mat disparityRef; 
cv::Mat K, D;

using namespace cv;

void setup(std::string folderName)
{
	/// Images
	getImages(images, folderName, 1);
	//getCameraIntrinsicParameters("..\\CameraCalibration\\calibration\\CalibrationFile", K, D);
	//undistortImages(images, K, D);
	blurImages(images, 4);

	/// Cameras
	getCameras(cameras, images[0].size(), folderName, 6e-3, 0, 1.55e-6);
	//getCameras(cameras, images.back().size());

	/// Mask
	//std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images[12]);
	std::vector<cv::Point2i> maskPoints{ {1,1}, {images[0].rows - 1,1}, {images[0].rows - 1,images[0].cols - 1}, {1,images[0].cols - 1} };
	mask = drawMask(images.back(), maskPoints);
}

int main()
{
	setup("Photographs\\Series1RotatedUndistorted");



	//Rect area = Rect{ Point2i(500, 0), Point2i(3800, 2800) };	//Photograps/Series1
	Rect area = Rect{ Point2i(1600, 800), Point2i(3000, 2000) };	//Photograps/Series1
	//Rect area{ Point2i{887,421}, Point2i{1923,1716} };		// Renders3

	std::vector<Mat> subImages2;
	std::vector<Camera> subCameras2;

	//plt::figure(0);
	std::vector<int> camSel{1, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 20};
	//for (int c : camSel)
	//for (int c =0; c< cameras.size(); c++)
	//{
	//	subImgsAndCams({ c }, subImages2, subCameras2);
	//	std::vector<float> error = calcPixelArrayCost(subImages2, subCameras2, cameras[12], 240, 300, Point2i{ 1924,1510 },c);
	//	plt::plot(error, { {"label", std::to_string(c)} });

	//}		
	//plt::legend();
	//plt::show();
	/*plt::figure(1);
	subImgsAndCams(camSel, subImages2, subCameras2);
	std::vector<float> error = calcPixelArrayCost(images, cameras, cameras[12], 240, 300, Point2i{ 1924,1510 },12);
	plt::plot(error);
	plt::show();*/

	//std::cout << "Error: " << error.size() << std::endl;
	////printV(error);
	//plt::figure();
	//plt::plot(error);	
	//plt::show();
	//waitKey(0);

	Mat disparity;
	int disp12MaxDiff = 15;
	int preFilterCap = 15;//PreFilterCap		
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range
	int P1 = 5;
	int P2 = 192;
	//StereoSGBMImpl2 sgmInitial = StereoSGBMImpl2(216, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	// Renders3
	StereoSGBMImpl2 sgmInitial = StereoSGBMImpl2(252, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	// //Photograps/Series1
	//disparity = getCrossSGM(12, sgmInitial, false);
	//saveImage("DisparitySave", disparity);
	disparity = loadImage("DisparitySave");
	showImage("disparityCross", disparity(area) - 4300, 70, false, 0.5); //3900
	//getDiffWithAbsoluteReference(disparity(area), area, true);

	Mat roiMask = mask(area);
	//Mat depth = disparity2Depth(disparity, cameras[12], cameras[13])(area);
	//Mat normals = depth2Normals(depth, cameras[12]);
	//for (int i = 0; i < cameras.size(); i++)
	//{
	//	faceNormals.push_back(getOrthogonalityFromCamera(depth, roiMask, normals, cameras[12], cameras[i]));
	//}
	//normalizeMats(faceNormals);

	//StereoArraySGBM sgbm = StereoArraySGBM(216, 64, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange); // Renders3
	StereoArraySGBM sgbm = StereoArraySGBM(252, 48, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};
	std::vector<Mat> subImages;
	std::vector<Mat> subSurfsNorms;
	std::vector<Mat> subSurfsNormsAvgs;
	std::vector<Camera> subCameras;
	std::vector<Mat> difMats;


	subImgsAndCams({ 7, 11, 12, 13, 17 }, subImages, subCameras);
	testSGM(images[12](area), subImages, subCameras, 252, 48, area);

	subImgsAndCams({ 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24 }, subImages, subCameras);
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparity", disparity - 4300, 70, true, 0.5);

	subImgsAndCams({ 0, 4, 12, 20, 24 }, subImages, subCameras);
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparity", disparity - 4300, 70, true, 0.5);
	//testSGM(images[12](area), subImages, subCameras, 252, 48, area);
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, false, "righthalfCams.jpg") << std::endl;

	//subImgsAndCamsAndSurfs({ 2,3,4, 7,8,9, 12,13,14, 17,18,19, 22,23,24 }, subImages, subCameras, subSurfsNorms);
	//sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, false, "lefthalfCams.jpg") << std::endl;

	//subImgsAndCamsAndSurfs({ 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14 }, subImages, subCameras, subSurfsNorms);
	//sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, false, "tophalfCams.jpg") << std::endl;
	//subImgsAndCamsAndSurfs({ 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24 }, subImages, subCameras, subSurfsNorms);
	sgbm.compute(images, empty, cameras, cameras[12], area, disparity);
	showImage("disparity", disparity - 4300, 70, true, 0.5); //4800
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, true, "Series1_no_ortho.jpg") << std::endl;
	subImgsAndCamsAndSurfs({ 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24 }, subImages, subCameras, subSurfsNorms);
	sgbm.compute(subImages, faceNormals, subCameras, cameras[12], area, disparity);
	showImage("disparity", disparity - 4300, 70); //4800
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, true, "Series1_ortho.jpg") << std::endl;


	std::vector<int> cams{ 0, 5, 10, 15, 20 };
	subImgsAndCamsAndSurfs(cams, subImages, subCameras, subSurfsNorms);
	subSurfsNormsAvgs.push_back(getVectorMatsAverage(subSurfsNorms));
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparityOLC", disparity - 4300, 70, false); //4800
	difMats.push_back(getDiffWithAbsoluteReference(disparity, area));

	for (auto& c : cams) c += 1;
	subImgsAndCamsAndSurfs(cams, subImages, subCameras, subSurfsNorms);
	subSurfsNormsAvgs.push_back(getVectorMatsAverage(subSurfsNorms));
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparityOLC", disparity - 4300, 70, false); //4800
	difMats.push_back(getDiffWithAbsoluteReference(disparity, area));

	for (auto& c : cams) c += 1;
	subImgsAndCamsAndSurfs(cams, subImages, subCameras, subSurfsNorms);
	subSurfsNormsAvgs.push_back(getVectorMatsAverage(subSurfsNorms));
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparityOLC", disparity - 3900, 70, false); //4800
	difMats.push_back(getDiffWithAbsoluteReference(disparity, area));

	for (auto& c : cams) c += 1;
	subImgsAndCamsAndSurfs(cams, subImages, subCameras, subSurfsNorms);
	subSurfsNormsAvgs.push_back(getVectorMatsAverage(subSurfsNorms));
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparityOLC", disparity - 3900, 70, false); //4800
	difMats.push_back(getDiffWithAbsoluteReference(disparity, area));

	for (auto& c : cams) c += 1;
	subImgsAndCamsAndSurfs(cams, subImages, subCameras, subSurfsNorms);
	subSurfsNormsAvgs.push_back(getVectorMatsAverage(subSurfsNorms));
	sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	showImage("disparityOLC", disparity - 3900, 70, false); //4800
	difMats.push_back(getDiffWithAbsoluteReference(disparity, area));

	Mat lowestError{ difMats[0].size(), difMats[0].type(), Scalar{10000}};
	Mat lowestErrorMat{ difMats[0].size(), CV_8UC3, Scalar{0, 0, 0} };
	std::vector<Vec3b> Colors{ Vec3b{255,0,0}, Vec3b{170,0,100}, Vec3b{15, 0, 150}, Vec3b{100, 0, 170}, Vec3b{0,0,255} };
	for (Mat errorMat : difMats)
	{
		lowestError = cv::min(lowestError, errorMat);
	}
	Mat errorCollage;
	hconcat(difMats, errorCollage);
	showImage("difMat", errorCollage, 5, false, 1);

	Mat surfsCollage;
	hconcat(subSurfsNormsAvgs, surfsCollage);
	showImage("subsurfs", surfsCollage, 10, true, 0.25);

	//showImage("lowestError", lowestError, 5, true, 1);
	//showImage("lowestErrorMat", lowestErrorMat, true, 1);
	for (int matNum = 0; matNum < difMats.size(); matNum++)
	{
		Mat errorMat = difMats[matNum];			
		for (int y = 0; y < errorMat.rows; y++)
		{
			for (int x = 0; x < errorMat.cols; x++)
			{
				if (errorMat.at<uchar>(y, x) == lowestError.at<uchar>(y, x) && lowestError.at<uchar>(y,x) != 0 && lowestError.at<uchar>(y, x) != UCHAR_MAX)
					lowestErrorMat.at<Vec3b>(y, x) = Colors[matNum];
			}
		}
	}

	showImage("LowestErr", lowestErrorMat, 1, true, 1);

	//subImgsAndCamsAndSurfs({ 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24 }, subImages, subCameras, subSurfsNorms);
	//sgbm.compute(subImages, empty, subCameras, cameras[12], area, disparity);
	//showImage("disparity", disparity - 3900, 70); //4800
	//std::cout << getAvgDiffWithAbsoluteReference(disparity, area, true, "noOrtho.jpg") << std::endl;



	//exportOBJfromDisparity(disparity, "disparityTest.obj", cameras[0], cameras[1], 0.5);

	//plotNoseBridge(combination, "Combination");
	//plotNoseBridge(disparityRef, "Reference");
	//showPlot();



}

