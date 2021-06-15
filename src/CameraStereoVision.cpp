#pragma once
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <opencv2/core.hpp>
#pragma warning (pop)

#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"


namespace plt = matplotlibcpp;

std::string imageFolder;
std::vector<cv::Mat> images;
std::vector<cv::Mat> LBPs;
std::vector<cv::Mat> faceNormals;
std::vector<Camera> cameras;
std::vector<Camera> badCameras;
cv::Mat mask;
cv::Mat disparityRef; 
cv::Mat K, D;
cv::Mat groundTruth;
cv::Mat groundTruthNormals;
cv::Rect roi;
int P1, P2;

float scale = 1;

using namespace cv;

void setupPhysical(std::string folderName)
{
	imageFolder = folderName;

	/// Images
	getImages(images, folderName, scale);
	localBinaryPattern(images, LBPs, 4);

	/// Cameras
	getCameraIntrinsicParameters("..\\CameraCalibration\\calibration2\\CalibrationFile", K, D);
	getCameras(cameras, folderName);

	/// Ball model
	cv::Vec3f ballScreenParams = getBallScreenParams(images[12], 478, 482);
	roi = Rect{ Point2f(ballScreenParams[0] - ballScreenParams[2] * 1.1f, ballScreenParams[1] - ballScreenParams[2] * 1.1f),
		Point2f(ballScreenParams[0] + ballScreenParams[2] * 1.1f, ballScreenParams[1] + ballScreenParams[2] * 1.1f) };
	groundTruth = generateBallDepthMap(images[12].size(), ballScreenParams, cameras[12], 0.099);
	mask = groundTruth<0.99;
	groundTruthNormals = depth2Normals(groundTruth, cameras[12]);

	showImage("Mask", mask(roi), 1, false);
	showImage("Ideal raytraced ball depth", groundTruth(roi), 1, true, 0.7f);
	showImage("Ground truth normals", groundTruthNormals(roi), 1, false, 0.5);

	P1 = 10; P2 = 80;
}

void setupPhysicalFace(std::string folderName)
{
	imageFolder = folderName;

	/// Images
	getImages(images, folderName, scale);
	localBinaryPattern(images, LBPs, 4);

	/// Cameras
	getCameraIntrinsicParameters("..\\CameraCalibration\\calibration2\\CalibrationFile", K, D);
	getCameras(cameras, folderName);

	/// Ball model
	std::vector<Point2i> maskPoints{
		Point2i{2535,1365},
		Point2i{2468,1530},
		Point2i{2300,1640},
		Point2i{1997,1660},
		Point2i{1500,1500},
		Point2i{1560, 950},
		Point2i{2015, 920},
		Point2i{2400,1050},
		Point2i{2400,1050},
		Point2i{2530,1220}
	};
	mask = drawMask(images.back(), maskPoints);
	roi = cv::boundingRect(maskPoints);
	roi.x -= 20; roi.y -= 20; roi.width += 40; roi.height += 40;
	showImage("Mask", mask(roi), 1, false);
	showImage("12", images[12](roi), 1, false);
	//Mat masked;
	//images[12].copyTo(masked, mask);
	//showImage("masked", masked(roi), 1, true);
	P1 = 10; P2 = 80;
}


void setupRender(std::string folderName, float baseline)
{
	imageFolder = folderName;

	/// Images
	getImages(images, folderName, scale);
	localBinaryPattern(images, LBPs, 4);

	/// Cameras
	getCameras(cameras, baseline);

	/// Face model
	std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images[12]);
	roi = cv::boundingRect(maskPoints);
	roi.x -= 20; roi.y -= 20; roi.width += 40; roi.height += 40;
	mask = drawMask(images.back(), maskPoints);
	groundTruth = getIdealRef();
	resize(groundTruth, groundTruth, images[0].size(), 0, 0, INTER_LINEAR);
	groundTruth.convertTo(groundTruth, CV_32F);
	groundTruthNormals = depth2Normals(groundTruth, cameras[12],1, 0, 0, 5);

	showImage("Ground truth", groundTruth(roi), 1, false, 0.5);
	showImage("Mask", mask(roi), 1, false, 0.5);
	showImage("Ground truth normals", groundTruthNormals(roi), 1, false, 0.5);

	P1 = 5; P2 = 30;
}

void pitchPerformance(int distP, int multiplier)
{
	std::string strdp = std::to_string(distP);
	setupRender("Renders\\Renders"+strdp,float(distP)/1000.);
	//setupPhysical("Photographs\\CalibrationBol12p" + strdp + "Undistorted");

	Mat disparity;
	int minDisparity = 235 * distP*multiplier / 50;
	int numDisparities = 96;
	int disp12MaxDiff = 1500;
	int preFilterCap = 0; // Deprecated		
	int uniqRatio = 0;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range

	StereoArraySGBM sgbm = StereoArraySGBM(minDisparity, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};

	Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[12])(roi);
	showImage("Orthogonality", groundTruthOrthogonality, 1, false, 0.5);

	std::vector<int> c = { 12-multiplier*5,12-multiplier,12,12+multiplier,12+multiplier*5};
	sgbm.compute(LBPs, images, empty, cameras, 12, roi, disparity, float(distP*multiplier)/1000., mask, c);
	showImage("disparity", disparity - minDisparity * 16, 45, false, 0.5);
	Mat depth = disparity2Depth(disparity, cameras[12], cameras[12+multiplier]);

	std::vector<std::pair<float, float>> pairs = getSortedOrthogonalityDifference(depth, groundTruth(roi), groundTruthOrthogonality, mask(roi));
	std::vector<float> depthError, orthogonality;
	splitPairs(pairs, orthogonality, depthError);

	int window = (int)orthogonality.size() / 300;
	std::vector<float> mvAverage = centeredMovingAverage(depthError, window);
	float lastHalfAvg = 0;
	for (int n = mvAverage.size() / 2; n < mvAverage.size(); n++)
		lastHalfAvg += mvAverage[n] / (mvAverage.size() / 2);
	mvAverage += -lastHalfAvg;

	showDifference("diff", depth - lastHalfAvg, groundTruth(roi), 100, mask(roi));
	std::vector<float> deviation = centeredMovingAverageAbsoluteDeviation(depthError, window);
	std::string str2dp = std::to_string(distP * multiplier);
	saveVector("..\\..\\Python\\Plotting\\Data\\Pitch\\rPc5p" + str2dp + "Ort", orthogonality, window);
	saveVector("..\\..\\Python\\Plotting\\Data\\Pitch\\rPc5p" + str2dp + "Avg", mvAverage, window);
	saveVector("..\\..\\Python\\Plotting\\Data\\Pitch\\rPc5p" + str2dp + "Dev", deviation, window);

}

void camCountPerformance()
{
	//setupRender("Renders\\Renders50", 0.05);
	setupPhysical("Photographs\\CalibrationBol12p50Undistorted");

	Mat disparity;
	int minDisparity = 235;// 245 * distP / 50;
	int numDisparities = 96;
	int disp12MaxDiff = 1500;
	int preFilterCap = 0; // Deprecated		
	int uniqRatio = 0;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range

	StereoArraySGBM sgbm = StereoArraySGBM(minDisparity, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};

	Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[12])(roi);
	showImage("Orthogonality", groundTruthOrthogonality, 1, true, 0.5);
	std::vector<int> additiveCameraPos{ 12, 17, 11, 7, 13, 18, 8, 6, 16, 22, 14, 2, 10, 15, 23, 9, 1, 5, 21, 19, 3, 0, 20, 24, 4 };

	std::vector<int> c = { 12, 17, 11};
	for (int i = 3; i < additiveCameraPos.size(); i++)
	{
		c.push_back(additiveCameraPos[i]);
		Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[12])(roi);

		sgbm.compute(LBPs, images, empty, cameras, 12, roi, disparity, 0.05, mask, c);
		showImage("disparity", disparity - minDisparity * 16, 45, false, 0.5);
		Mat depth = disparity2Depth(disparity, cameras[12], cameras[13]);

		std::vector<std::pair<float, float>> pairs = getSortedOrthogonalityDifference(depth, groundTruth(roi), groundTruthOrthogonality, mask(roi));
		std::vector<float> depthError, orthogonality;
		splitPairs(pairs, orthogonality, depthError);

		int window = (int)orthogonality.size() / 500;
		std::vector<float> mvAverage = centeredMovingAverage(depthError, window);
		float lastHalfAvg = 0;
		for (int n = mvAverage.size() / 2; n < mvAverage.size(); n++)
			lastHalfAvg += mvAverage[n] / (mvAverage.size() / 2);
		mvAverage += -lastHalfAvg;

		showDifference("diff", depth - lastHalfAvg, groundTruth(roi), 200, mask(roi));
		std::vector<float> deviation = centeredMovingAverageAbsoluteDeviation(depthError, window);
		saveVector("..\\..\\Python\\Plotting\\Data\\Camcount\\pCc" + std::to_string(c.size()) + "Ort", orthogonality, 100);
		saveVector("..\\..\\Python\\Plotting\\Data\\Camcount\\pCc" + std::to_string(c.size()) + "Avg", mvAverage, 100);
		saveVector("..\\..\\Python\\Plotting\\Data\\Camcount\\pCc" + std::to_string(c.size()) + "Dev", deviation, 100);
	}
}

void orthogonalityPerformance()
{
	int distP = 50;
	//setupRender("Renders\\Renders50", float(distP)/1000.);
	setupPhysical("Photographs\\CalibrationBol12p50Undistorted");

	Mat disparity;
	int minDisparity = 235;// 245 * distP / 50;
	int numDisparities = 96;
	int disp12MaxDiff = 1500;
	int preFilterCap = 0; // Deprecated		
	int uniqRatio = 0;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range

	StereoArraySGBM sgbm = StereoArraySGBM(minDisparity, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};

	Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[12])(roi);
	showImage("Orthogonality", groundTruthOrthogonality, 1, true, 0.5);
	std::vector<int> additiveCameraPos{ 12, 13, 7, 11, 17, 18, 8, 6, 16, 22, 14, 2, 10, 15, 23, 9, 1, 5, 21, 19, 3, 0, 20, 24, 4 };
	std::vector<std::pair<float, float>> pairs;
	for (int i = 0; i < 9; i++)
	{
		std::vector<int> c = getCrossIDs(additiveCameraPos[i]);
		Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[additiveCameraPos[i]])(roi);
		showImage("Orthogonality", groundTruthOrthogonality, 1, false, 0.5);

		sgbm.compute(LBPs, images, empty, cameras, 12, roi, disparity, float(distP) / 1000., mask, c);
		showImage("disparity", disparity - minDisparity * 16, 45, false, 0.5);
		Mat depth = disparity2Depth(disparity, cameras[12], cameras[13]);
		std::vector<std::pair<float, float>> tempPairs = getSortedOrthogonalityDifference(depth, groundTruth(roi), groundTruthOrthogonality, mask(roi));
		pairs.insert(pairs.end(), tempPairs.begin(), tempPairs.end());
	}
	sort(pairs.begin(), pairs.end());
	std::vector<float> depthError, orthogonality;
	splitPairs(pairs, orthogonality, depthError);

	int window = (int)orthogonality.size()/500;
	std::vector<float> mvAverage = centeredMovingAverage(depthError, window);
	float lastHalfAvg = 0;
	for (int n = (int)mvAverage.size() / 2; n < mvAverage.size(); n++)
		lastHalfAvg += mvAverage[n] / (mvAverage.size() / 2);
	mvAverage += -lastHalfAvg;

	//showDifference("diff", depth, groundTruth(roi), 200, mask(roi));
	std::vector<float> deviation = centeredMovingAverageAbsoluteDeviation(depthError, window);
	std::string strdp = std::to_string(distP);
	saveVector("..\\..\\Python\\Plotting\\Data\\Orthogonality\\pOc5p"+ strdp + "Ort", orthogonality, window);
	saveVector("..\\..\\Python\\Plotting\\Data\\Orthogonality\\pOc5p" + strdp + "Avg", mvAverage, window);
	saveVector("..\\..\\Python\\Plotting\\Data\\Orthogonality\\pOc5p" + strdp + "Dev", deviation, window);
}

void fullPerformance()
{
	int distP = 50;
	//setupRender("Renders\\Renders50", float(distP)/1000.);
	setupPhysical("Photographs\\CalibrationBol12p50Undistorted");

	Mat disparity;
	int minDisparity = 235;// 245 * distP / 50;
	int numDisparities = 96;
	int disp12MaxDiff = 1500;
	int preFilterCap = 0; // Deprecated		
	int uniqRatio = 0;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 5;	// Speckle range

	StereoArraySGBM sgbm = StereoArraySGBM(minDisparity, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};

	Mat_<float> groundTruthOrthogonality = getOrthogonalityFromCamera(groundTruth, mask, groundTruthNormals, cameras[12], cameras[12])(roi);
	showImage("Orthogonality", groundTruthOrthogonality, 1, false, 0.5);

	sgbm.compute(LBPs, images, empty, cameras, 12, roi, disparity, float(distP) / 1000., mask);
	showImage("disparity", disparity - minDisparity * 16, 48, true, 0.5);
	Mat depth = disparity2Depth(disparity, cameras[12], cameras[13]);
	std::vector<std::pair<float, float>> pairs = getSortedOrthogonalityDifference(depth, groundTruth(roi), groundTruthOrthogonality, mask(roi));

	std::sort(pairs.begin(), pairs.end());
	std::vector<float> depthError, orthogonality;
	splitPairs(pairs, orthogonality, depthError);

	int window = (int)orthogonality.size() / 300;
	std::vector<float> mvAverage = centeredMovingAverage(depthError, window);
	float lastHalfAvg = 0;
	for (int n = (int)mvAverage.size() / 2; n < mvAverage.size(); n++)
		lastHalfAvg += mvAverage[n] / (mvAverage.size() / 2);
	mvAverage += -lastHalfAvg;
	showDifference("diff", depth - lastHalfAvg, groundTruth(roi), 150, mask(roi));
	std::vector<float> deviation = centeredMovingAverageAbsoluteDeviation(depthError, window);
	std::string strdp = std::to_string(distP);
	//saveVector("..\\..\\Python\\Plotting\\Data\\FullSetup\\pFc25p" + strdp + "Ort", orthogonality, window);
	//saveVector("..\\..\\Python\\Plotting\\Data\\FullSetup\\pFc25p" + strdp + "Avg", mvAverage, window);
	//saveVector("..\\..\\Python\\Plotting\\Data\\FullSetup\\pFc25p" + strdp + "Dev", deviation, window);
}

void physicalFacePerformance()
{
	//setupRender("Renders\\Renders50", float(distP)/1000.);
	setupPhysicalFace("Photographs\\Series12Undistorted");

	Mat disparity;
	int minDisparity = 265;// 245 * distP / 50;
	int numDisparities = 96;
	int disp12MaxDiff = 3;
	int preFilterCap = 0; // Deprecated		
	int uniqRatio = 0;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100		(Currently turned off in stereoArraySGM.cpp)
	int sRange = 5;	// Speckle range					(Currently turned off in stereoArraySGM.cpp)
	P1 = 4; P2 = 100;
	StereoArraySGBM sgbm = StereoArraySGBM(minDisparity, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange);	 //Photograps/Series1
	std::vector<Mat> empty{};

	sgbm.compute(LBPs, images, empty, cameras, 12, roi, disparity, 50. / 1000., mask);
	showImage("disparity "+std::to_string(P2), disparity - minDisparity * 16, 50, true, 0.5);
	Point2i roiPrincipalPoint = cameras[12].principalPoint - roi.tl();
	//Mat depth = disparity2Depth(disparity, cameras[12], cameras[13]);
	exportXYZfromDisparity(disparity, "HenkSGM.xyz", cameras[12], 0.05, roiPrincipalPoint, 1, mask(roi));
	waitKey(0);
}

int main()
{
	//pitchPerformance(5,1);
	//pitchPerformance(10,1);
	//pitchPerformance(20,1);
	//pitchPerformance(30,1);
	//pitchPerformance(40,1);
	//pitchPerformance(50,1);
	//pitchPerformance(30,2);
	//pitchPerformance(40,2);
	//pitchPerformance(50,2);
 
	//camCountPerformance();

	//orthogonalityPerformance();

	//physicalFacePerformance();

	Mat absDiff = imread("absDiffHenk.jpg", IMREAD_GRAYSCALE);
	Mat diff = imread("meanHenk.jpg", IMREAD_GRAYSCALE);
	//cv::subtract(diff, -128, diff, noArray(), CV_16S);
	diff.convertTo(diff, CV_32F, (6.23895 + 6.71251) / 256, -6.71251);
	absDiff.convertTo(absDiff, CV_32F, 6.71251 / 256.);
	//min = -6.71251 max = 6.23895
	Mat posDiff, negDiff;
	Mat mask = (diff > 0);
	Mat invMask = (diff < 0);
	absDiff.copyTo(posDiff, mask);
	absDiff.copyTo(negDiff, invMask);
	showImage("mask", mask, 1,false,1);
	showImage("imask", invMask, 1, false,1);
	showImage("pos", posDiff, 0.5, false, 1);
	showImage("neg", negDiff, 0.5, true, 1);
	showDifference("Difference", posDiff, negDiff, 0.5);
}
