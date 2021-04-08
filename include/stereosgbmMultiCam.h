#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


struct StereoSGBMParams2
{
	StereoSGBMParams2()
	{
		minDisparity = numDisparities = 0;
		SADWindowSize = 0;
		P1 = P2 = 0;
		disp12MaxDiff = 0;
		preFilterCap = 0;
		uniquenessRatio = 0;
		speckleWindowSize = 0;
		speckleRange = 0;
		mode = cv::StereoSGBM::MODE_SGBM;
	}

	StereoSGBMParams2(int _minDisparity, int _numDisparities, int _SADWindowSize,
		int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
		int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
		int _mode)
	{
		minDisparity = _minDisparity;
		numDisparities = _numDisparities;
		SADWindowSize = _SADWindowSize;
		P1 = _P1;
		P2 = _P2;
		disp12MaxDiff = _disp12MaxDiff;
		preFilterCap = _preFilterCap;
		uniquenessRatio = _uniquenessRatio;
		speckleWindowSize = _speckleWindowSize;
		speckleRange = _speckleRange;
		mode = _mode;
	}

	inline bool isFullDP() const
	{
		return mode == cv::StereoSGBM::MODE_HH || mode == cv::StereoSGBM::MODE_HH4;
	}
	inline cv::Size calcSADWindowSize() const
	{
		const int dim = SADWindowSize > 0 ? SADWindowSize : 5;
		return cv::Size(dim, dim);
	}

	int minDisparity;
	int numDisparities;
	int SADWindowSize;
	int preFilterCap;
	int uniquenessRatio;
	int P1;
	int P2;
	int speckleWindowSize;
	int speckleRange;
	int disp12MaxDiff;
	int mode;
};

class StereoSGBMImpl2 : public cv::StereoSGBM
{
public:
	StereoSGBMImpl2();
	StereoSGBMImpl2(int _minDisparity, int _numDisparities, int _SADWindowSize,
		int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
		int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
		int _mode);

	//void create(int minDisparity, int numDisparities, int SADWindowSize,
	//	int P1, int P2, int disp12MaxDiff,
	//	int preFilterCap, int uniquenessRatio,
	//	int speckleWindowSize, int speckleRange,
	//	int mode);

	void compute(cv::InputArray leftarr, cv::InputArray rightarr, cv::OutputArray disparr);
	void computeMultiCam(std::vector<cv::Mat> images, std::vector<cv::Point2i> disparityDirection, cv::OutputArray disparr);

	int getMinDisparity() const { return params.minDisparity; }
	void setMinDisparity(int minDisparity) { params.minDisparity = minDisparity; }

	int getNumDisparities() const { return params.numDisparities; }
	void setNumDisparities(int numDisparities) { params.numDisparities = numDisparities; }

	int getBlockSize() const { return params.SADWindowSize; }
	void setBlockSize(int blockSize) { params.SADWindowSize = blockSize; }

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

	int getMode() const { return params.mode; }
	void setMode(int mode) { params.mode = mode; }

	StereoSGBMParams2 params;
	cv::Mat buffer;

	inline static const char* name_ = "StereoMatcher.SGBM";;
};
