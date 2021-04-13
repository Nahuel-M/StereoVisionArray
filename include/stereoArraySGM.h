#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


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

	void compute(std::vector<cv::Mat>& images, cv::Rect area, cv::Size arrayShape, int centerCamId, cv::OutputArray disparr);

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
