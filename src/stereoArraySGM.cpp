
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <limits.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"
#pragma warning (pop)


#include "stereoArraySGM.h"
#include "imageHandling.h"
#include< intrin.h >

using namespace cv;



template <typename printableVector>
void printVector(printableVector input)
{
	for (auto const& i : input) {
		std::cout << (int)i << " ";
	}
	std::cout << std::endl;
}

// NR - the number of directions. the loop on x that computes Lr assumes that NR == 8.
// if you change NR, please, modify the loop as well.
enum { NR = 8, NR2 = NR / 2 };

static inline v_int16 vx_setseq_s16()
{
	return v_int16(0, 1, 2, 3, 4, 5, 6, 7);
}
// define some additional reduce operations:
static inline void min_pos(const v_int16& val, const v_int16& pos, short& min_val, short& min_pos)
{
	min_val = v_reduce_min(val); // Finds minimum of all values in the v_int16 input
	v_int16 v_mask = (vx_setall_s16(min_val) == val);	// Mask of only the min_val
	// Masks the position + the num of position. Sets all other values to the max short value and then finds the minimum
	// Looks like it will always find pos+index of pos where val is lowest
	min_pos = v_reduce_min(((pos + vx_setseq_s16()) & v_mask) | (vx_setall_s16(SHRT_MAX) & ~v_mask));
}

__forceinline CostType hammingDistance(uchar n1, uchar n2)
{
	//CostType disti = 0;
	//for (int i = 0; i < 8; i++) 
	//	disti += (n1 & (1 << i)) != (n2 & (1 << i));
	return (CostType)__popcnt((uint)n1^(uint)n2);
}

__forceinline float floatAt(float x, float y, int width, PixType* imagePointer)
{
	float floorX = floor(x);
	float floorY = floor(y);
	int intFloorX = (int)floorX;
	int intFloorY = (int)floorY;
	float xPercent = x - floorX;
	float yPercent = y - floorY;
	return
		float(imagePointer[intFloorY * width + intFloorX])			 * (1.f - xPercent)	* (1.f - yPercent) +
		float(imagePointer[intFloorY * width + intFloorX + 1])		 * (xPercent)		* (1.f - yPercent) +
		float(imagePointer[(intFloorY + 1) * width + intFloorX])	 * (1.f - xPercent)	* (yPercent) +
		float(imagePointer[(intFloorY + 1) * width + intFloorX + 1]) * (xPercent)		* (yPercent);

	//return imagePointer[int(x + 0.5) + int(y + 0.5) * width];
}

inline int xDer(int x, int yXwidth, int width, PixType* imagePointer)
{
	int xDerivative;
	xDerivative = imagePointer[yXwidth -  width + (x - 1)] - imagePointer[yXwidth - width + (x + 1)];
	xDerivative += (imagePointer[yXwidth + (x - 1)] - imagePointer[yXwidth + (x + 1)]) * 2;
	xDerivative += imagePointer[yXwidth + width + (x - 1)] - imagePointer[yXwidth + width + (x + 1)];
	return xDerivative;
}

inline int yDer(int x, int yXwidth, int width, PixType* imagePointer)
{
	int yDerivative;
	yDerivative = imagePointer[yXwidth - width + (x - 1)] - imagePointer[yXwidth + width + (x - 1)];
	yDerivative += (imagePointer[yXwidth - width + x] - imagePointer[yXwidth + width + x]) * 2;
	yDerivative += imagePointer[yXwidth - width + (x + 1)] - imagePointer[yXwidth + width + (x + 1)];
	return yDerivative;
}

inline bool calcDisparityCost(std::array<PixType*, 25> &imagePointers,
	int x, int y, int d, int centerCamId, std::vector<Point2f> &disparityStep,
	size_t camCount, int width, int height, CostType& cost,
	std::array<uchar, 25> &intensities, std::array<uchar, 25>& camCost)
{
	for (int c = 0; c < camCount; c++)
	{
		Point2i position = Point2i{ x , y } + Point2i{ d * disparityStep[c] };

		if (position.x < 1 || position.y < 1 || position.x >= width-1 || position.y >= height-1)
		{
			return false;
		}
		intensities[c] = imagePointers[c][position.x + position.y * width];
	}
	cost = 0;
	for (int c = 0; c < camCount; c++)
	{
		//intensityDiff = max(abs(intensities[c] - centerIntensity)-3,0.f);
		//cost += intensityDiff;// +abs(intensities[c] - avgIntensity);
		//if (c - 5 > 0)
		//	cost += max(abs(intensities[c] - intensities[c - 5])-3,0.f);
		//if (c + 5 < 25)
		//	cost += max(abs(intensities[c] - intensities[c + 5]) - 3, 0.f);
		//if (c % 5 > 0)
		//	cost += max(abs(intensities[c] - intensities[c - 1]) - 3, 0.f);
		//if (c % 5 < 4)
		//	cost += max(abs(intensities[c] - intensities[c + 1]) - 3, 0.f);
		camCost[c] = 0;
		bool u = (c - 5 > 0), d = (c + 5 < 25), l = (c % 5 > 0), r = (c % 5 < 4);
		if (u)
			camCost[c] += hammingDistance(intensities[c], intensities[c - 5]);
		if (d)
			camCost[c] += hammingDistance(intensities[c], intensities[c + 5]);
		if (l)
			camCost[c] += hammingDistance(intensities[c], intensities[c - 1]);
		if (r)
			camCost[c] += hammingDistance(intensities[c], intensities[c + 1]);
		camCost[c] = camCost[c] * (uchar)4 / ((uchar)u + (uchar)d + (uchar)l + (uchar)r);

		//cost += camCost[c];
	}
	//cost = cost * 4 / camCount;
	//return true;
	std::nth_element(camCost.begin(), camCost.begin() + camCount/2, camCost.end());
	uchar median = camCost[camCount/2];
	//std::cout << (int)median << std::endl;
	CostType correctCount = 0;
	for (int c = 0; c < camCount; c++)
	{
		if (abs(camCost[c] - median) < 3)
		{
			cost += camCost[c];
			correctCount += 1;
		}
	}
	//std::cout << correctCount << std::endl;
	cost = cost * 4 / correctCount;
	return true;
}

void calcPixelwiseArrayCost(std::vector<cv::Mat>& images, 
	std::vector<Point2f> disparityStep, cv::Rect area,
	int minD, int maxD, CostType* costOrigin)
{
	int centerCamId = -1;
	size_t camCount = images.size();
	int D = maxD - minD;
	int width = images[0].cols;
	int height = images[0].rows;
	std::array<PixType*, 25> imagePointers{};
	for (int c = 0; c < camCount; c++)
	{
		if (disparityStep[c] == Point2f{ 0,0 }) centerCamId = c;
		/// Get the origin of each image
		imagePointers[c] = images[c].ptr<PixType>(0);
	}


	std::array<uchar, 25> intensities{ 0 };
	std::array<uchar, 25> camCost{ 0 };
	std::array<int, 25> pintensities{ 0 };

	CostType cost = 0;
	Point2i position;

	/// Iterate over every pixel and every disparity
	costOrigin = costOrigin - minD;
	for (int y = area.tl().y; y < area.br().y; y++)
	{
		CostType* rowOrigin = costOrigin + (y - area.tl().y) * area.width * D;
		for (int x = area.tl().x; x < area.br().x; x++)
		{
			CostType* pixelOrigin =  rowOrigin + (x-area.tl().x) * D;
			for (int d = minD; d < maxD; d++)
			{
				if (!calcDisparityCost(imagePointers, x, y, d, centerCamId, disparityStep, camCount, width, height, cost,
					intensities, camCost))
				{
					for (; d < maxD; d++)
						pixelOrigin[d] = SHRT_MAX;
					goto next_pixel;
				}
				pixelOrigin[d] = cost;
			}
		next_pixel:;
		}

	}
}

inline void calcDisparityCostPerEdge(
	const std::vector<std::vector<PixType>>& LBPs,
	std::vector<std::vector<PixType>>& intensities,
	int centerCamIntensity,
	std::vector<std::vector<CostType>>& costs,
	std::array<uchar, 25>& dispLength,
	std::pair<int,int>& edge, int edgeID, int D)
{
	uchar minD = min(dispLength[edge.first], dispLength[edge.second]);

	int disp = 0;
	//std::cout << "minD " << (int)minD << std::endl;
	for (; disp < minD; disp++)
	{
		if (abs((int)intensities[edge.first][disp] - centerCamIntensity) > 10)	
		{ // Quick implementation of intensity matching. This avoids matches with the black background.
			costs[disp][edgeID] = SHRT_MAX / 25;
			continue;
		}
		costs[disp][edgeID] = hammingDistance(LBPs[edge.first][disp], LBPs[edge.second][disp]);
	}
	for (; disp < D; disp++)
	{
		costs[disp][edgeID] = SHRT_MAX/25;
	}
}

inline uchar getCamValues(PixType* imPointer,
	int x, int y, int minD, int maxD,
	Point2f disparityStep, int width, int height, std::vector<PixType>& intensities)
{
	Point2i origin = Point2i{ x , y };
	Point2i position{};
	for (int d = minD; d < maxD; d++) {
		position = origin + Point2i( d * disparityStep );
		if (position.x < 1 || position.y < 1 || position.x >= width - 1 || position.y >= height - 1)
			return d-minD;
		intensities[d-minD] = imPointer[position.x + position.y * width];
	}
	return maxD - minD;
}

inline CostType medianMean(std::vector<CostType>& v)
{
	size_t n = (v.size()+1)/2;
	nth_element(v.begin(), v.begin() + n, v.end());
	//return(v[n]);
	int avg = 0;
	for (int i = 0; i < n; i++)
		avg += v[i];
	avg = avg * 2 / (int)n;
	return (CostType)avg;
}

inline void addPairInDirection(int d, int c, std::vector<int>& usedCams, std::vector<int>& pairs)
{
	for (int i = 1; i <= 4; i++)
	{
		int pos = std::find(usedCams.begin(), usedCams.end(), c + d * i) - usedCams.begin();
		if (pos < usedCams.size())
		{
			pairs.push_back(pos);
			break;
		}
		if ((c + d * i) % 5 == 0 || (c + d * i) % 5 == 4)
			break;
	}
}

std::vector<int> getPairedCams(int c, std::vector<int>& usedCams)
{
	std::vector<int> pairs;
	addPairInDirection(-5, c, usedCams, pairs);
	addPairInDirection(5, c, usedCams, pairs);
	addPairInDirection(-1, c, usedCams, pairs);
	addPairInDirection(1, c, usedCams, pairs);
	std::cout << c << std::endl;
	printVector(pairs);
	return pairs;
}

std::vector<std::pair<int, int>> edgesFromPairs(std::vector<std::vector<int>>& pairedCams)
{
	std::vector<std::pair<int, int>> edges;
	for (int pc = 0; pc < pairedCams.size(); pc++)
	{
		for (int c = 0; c < pairedCams[pc].size(); c++) {
			std::pair<int, int> pair{ min(pc,pairedCams[pc][c]),max(pc,pairedCams[pc][c]) };
			if (std::find(edges.begin(), edges.end(), pair) == edges.end())
			{
				edges.push_back(pair);
				std::cout << edges.back().first << ", " << edges.back().second << std::endl;
			}
		}
	}
	return edges;
}

void calcPixelwiseArrayCostFlip(std::vector<cv::Mat>& LBPs, std::vector<cv::Mat>& images,
	std::vector<Point2f> disparityStep, cv::Rect area, cv::Mat_<uchar> mask,
	std::vector<int> usedCams, int minD, int maxD, CostType* costOrigin)
{
	if (usedCams.size() == 0) {	/// If no cameras are passed to this function, it uses all cameras
		for (int i = 0; i < disparityStep.size(); i++)
			usedCams.push_back(i);
	}
	int centerCamId=-1;
	size_t camCount = images.size();
	int D = maxD - minD;
	int width = images[0].cols;
	int height = images[0].rows;
	std::array<PixType*, 25> LBPointers{};
	std::array<PixType*, 25> imagePointers{};
	std::vector<std::vector<int>> pairedCams;
	for (int c = 0; c < images.size(); c++)
	{
		if (disparityStep[c] == Point2f{ 0,0 }) 
			centerCamId = c;
		/// Get the origin of each image
		LBPointers[c] = LBPs[c].ptr<PixType>(0);
		imagePointers[c] = images[c].ptr<PixType>(0);
	}
	for (int c : usedCams)
	{
		pairedCams.push_back(getPairedCams(c, usedCams));
	}
	std::vector<std::pair<int,int>> edges = edgesFromPairs(pairedCams);
	if (centerCamId < 0 || centerCamId >= 25)
	{
		throw("No center camera found?");
		return;
	}
	std::cout << "Center camera: " << centerCamId << std::endl;
	/// Iterate over every pixel and every disparity
	std::vector<std::vector<PixType>> LBPvals(usedCams.size(), std::vector<PixType>(D, 0) );
	std::vector<std::vector<PixType>> intensities(usedCams.size(), std::vector<PixType>(D, 0) );
	std::vector<std::vector<CostType>> costs(D, std::vector<CostType>(edges.size(), 0));
	std::array<uchar, 25> dispLength = { 0 };
	CostType cost = 0;
	Point2i position;
	for (int y = area.tl().y; y < area.br().y; y++)
	{
		CostType* rowOrigin = costOrigin + (y - area.tl().y) * area.width * D;
		for (int x = area.tl().x; x < area.br().x; x++)
		{
			CostType* pixelOrigin = rowOrigin + (x - area.tl().x) * D;
			if (mask(y, x) == 0)
			{
				for (int d = 0; d < D; d++)
				{
					pixelOrigin[d] = 0;
				}
				continue;
			}
			int centerCamIntensity = imagePointers[centerCamId][x + y * width];
			//if (usingMask && maskPointer[x + y * width] == 0) continue;
			for (int c = 0; c < usedCams.size(); c++) {	/// Calculate LBP values for each camera
				dispLength[c] = getCamValues(LBPointers[usedCams[c]], x, y, minD, maxD, disparityStep[usedCams[c]], width, height, LBPvals[c] );
				getCamValues(imagePointers[usedCams[c]], x, y, minD, maxD, disparityStep[usedCams[c]], width, height, intensities[c] );
			}				
			for (int e = 0; e < edges.size(); e++) {	/// Calculate matching cost with neighbours for each camera
				calcDisparityCostPerEdge(LBPvals, intensities, centerCamIntensity, costs, dispLength, edges[e], e, D);
			}
			for (int d = 0; d < D; d++)
			{											/// For each disparity, take the mean value of the 50% lowest costs
				pixelOrigin[d] = medianMean(costs[d]);
			}
		}
	}
}
//
//std::vector<float> calcDisparityCostForPlotting(std::vector<cv::Mat> images, std::vector<Camera> cameras, int camID, int x, int y, int minD, int maxD)
//{
//	size_t camCount = cameras.size();
//	int centerCamId = -1;
//	std::vector<Point2f> disparityStep;
//	std::array<PixType*, 25> imagePointers{};
//	Point2f centerPos{ (float)-cameras[12].pos3D.x, (float)cameras[12].pos3D.y };
//	float dist = 5e-2;
//	int D = maxD - minD;
//	std::array<int, 25> pintensities{ 0 };
//
//	for (size_t c = 0; c < cameras.size(); c++)
//	{
//		Point2f direction = (Point2f{ (float)-cameras[c].pos3D.x, (float)cameras[c].pos3D.y } - centerPos) / dist;
//		if (direction == Point2f{ 0,0 }) centerCamId = c;
//		disparityStep.push_back(Point2f(direction));
//		imagePointers[c] = images[c].ptr<PixType>(0);
//	}
//
//	std::vector<std::vector<PixType>> intensities(cameras.size(), std::vector<PixType>(D, 0));
//	std::vector<std::vector<CostType>> costs(D, std::vector<CostType>(cameras.size(), 0));
//
//	std::array<uchar, 25> dispLength = { 0 };
//	for (int c = 0; c < cameras.size(); c++)
//	{
//		dispLength[c] = getCamIntensities(imagePointers[c], x, y, minD, maxD, disparityStep[c], images[0].cols, images[0].rows, intensities[c]);
//	}
//	calcDisparityCostPerCamera(intensities, costs, dispLength, camID, camID, D);
//
//	//printVector(costs[camID]);
//	std::vector<float> returnCosts;
//	for (int d = 0; d < D; d++)
//	{
//		returnCosts.push_back(costs[d][camID]);
//	}
//	return returnCosts;
//}


void calcPixelwiseArrayCostWithSurfaceNorms(std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfsPars, 
	std::vector<Point2f> disparityStep, cv::Rect area,
	int minD, int maxD, CostType* costOrigin)
{
	size_t camCount = images.size();
	int D = maxD - minD;
	int width = images[0].cols;
	int height = images[0].rows;
	std::array<PixType*, 25> imagePointers;
	std::array<float*, 25> surfNormPts;
	for (int c = 0; c < camCount; c++)
	{
		/// Get the origin of each image
		imagePointers[c] = images[c].ptr<PixType>(0);
		surfNormPts[c] = surfsPars[c].ptr<float>(0);
	}
	costOrigin = costOrigin - minD;
	/// Iterate over every pixel and every disparity
	for (int y = area.tl().y; y < area.br().y; y++)
	{
		CostType* rowOrigin = costOrigin + (y - area.tl().y) * area.width * D;
		for (int x = area.tl().x; x < area.br().x; x++)
		{
			CostType* pixelOrigin = rowOrigin + (x - area.tl().x) * D;
			for (int d = minD; d < maxD; d++)
			{
				std::array<float, 25> intensities{ 0 };
				std::array<float, 25> weight{ 0 };
				float averageIntensity = 0;
				float weightedIntensitySum = 0;

				/// Iterate over all cameras to find intensity values
				for (int c = 0; c < camCount; c++)
				{
					Point2i position = Point2i{ x , y } + Point2i(d * disparityStep[c] + Point2f{ 0.5,0.5 });
					if (position.x < 0 || position.y < 0 || position.x >= width || position.y >= height)
					{
						for (; d < minD; d++)
							pixelOrigin[d] = SHRT_MAX;
						goto next_pixel;
					}
					weight[c] = surfNormPts[c][(y - area.tl().y) * area.width + x - area.tl().x];
					//if (weight[c] == 0)
					//	weight[c] = 1. / camCount;
					intensities[c] = (float)imagePointers[c][position.y * width + position.x];
					averageIntensity += intensities[c] * weight[c];
					weightedIntensitySum += weight[c];
					//std::cout << (float)intensities[validIntensities] << std::endl;
				}
				averageIntensity /= weightedIntensitySum;

				float cost = 0;
				for (int c = 0; c < camCount; c++)
				{
					cost += std::abs(intensities[c] - averageIntensity) * weight[c];
				}
				cost /= weightedIntensitySum;
				pixelOrigin[d] = (CostType)cost;
				//std::cout << averageIntensity << std::endl;
			}
		next_pixel:;
		}

	}
}

std::vector<float> calcPixelArrayIntensity(Mat& image, Camera cam, Camera centerCam,
	int minD, int maxD, Point2i position)
{
	Point2f centerPos{ (float)-centerCam.pos3D.x, (float)centerCam.pos3D.y };
	double dist = 5e-2;

	Point2f direction = (Point2f{ (float)-cam.pos3D.x, (float)cam.pos3D.y } - centerPos) / dist;

	int D = maxD - minD;
	int width = image.cols;
	int height = image.rows;
	Mat plotIm = image.clone();
	std::vector<float> intensities;
	for (int d = minD; d < maxD; d++)
	{
		/// Iterate over all cameras to find intensity values
		Point2i positionR = position + Point2i(d * direction);

		if (positionR.x < 0 || positionR.y < 0 || positionR.x >= width || positionR.y >= height)
		{
			for (; d < maxD; d++) {
				intensities.push_back(0);
			}
			return intensities;
		}
		intensities.push_back((float)image.at<uchar>(positionR.y, positionR.x));
		plotIm(Rect{ positionR, Size{1,1} }) = 255;
	}
	showImage("plotIm" + std::to_string(direction.x) + "_" + std::to_string(direction.y),
		plotIm(Rect{ position + Point2i(minD * direction) - Point2i{maxD - minD,maxD - minD}, Size{(maxD-minD)*2,(maxD - minD) * 2} }), 1, false, 3);
	return intensities;
}
/// Calculate x derivative for disparities for a single pixel

std::vector<float> calcPixelArrayCost(std::vector<cv::Mat>& images, std::vector<Camera> cams, Camera centerCam,
	int minD, int maxD, Point2i position, int camera)
{
	Point2d centerPos{ -centerCam.pos3D.x, centerCam.pos3D.y };
	double dist = 5e-2;

	std::vector<Point2i> directions;
	std::cout << "Directions: ";
	for (auto& c : cams)
	{
		Point2d direction = (Point2d{ -c.pos3D.x, c.pos3D.y } - centerPos) / dist;
		directions.push_back(Point2i(direction));
		std::cout << direction << ", ";
	}
	std::cout << std::endl;

	int camCount = (int)cams.size();
	int D = maxD - minD;
	int width = images[0].cols;
	int height = images[0].rows;
	Mat plotIm = images[0].clone();
	std::vector<float> pixelCost;
	for (int d = minD; d < maxD; d++)
	{
		std::vector<float> intensities;
		/// Iterate over all cameras to find intensity values
		float averageIntensity = 0;
		int c = 0;
		for (; c < camCount; c++)
		{
			Point2i positionR;
			positionR = position + d * directions[c];


			if (positionR.x < 0 || positionR.y < 0 || positionR.x >= width || positionR.y >= height)
			{
				intensities.push_back(0);
				std::cout << "broken" << std::endl;
				break;
			}

			intensities.push_back((float)images[c].at<uchar>(positionR.y, positionR.x));
			averageIntensity += intensities[c];
			//std::cout << intensities[c] << std::endl;
			plotIm(Rect{ positionR, Size{1,1} }) = 255;
		}

		averageIntensity /= c;

		float cost = 0;
		for (int c = 0; c < camCount; c++)
		{
			cost += std::abs(intensities[c] - intensities[camera]);
		}
		pixelCost.push_back(cost);
	}
	//showImage("plotIm", plotIm(Rect{ position + minD * directions[0] - Point2i{150,150}, Size{300,300} }), 1, true, 3);
	return pixelCost;
}

class BufferSGBM
{
private:
	size_t Da;
	size_t Dlra;
	size_t costWidth;
	size_t height;
	size_t width;
	size_t hsumRows;
	uchar dirs;
	uchar dirs2;
	static const size_t TAB_OFS = 256 * 4;

public:
	CostType* Cbuf;
	CostType* Sbuf;
	CostType* disp2cost;
	DispType* disp2ptr;
	PixType* tempBuf;
	std::vector<CostType*> Lr;
	std::vector<CostType*> minLr;
	PixType* clipTab;

private:
	utils::BufferArea area;

public:
	BufferSGBM(
		size_t Da_,
		size_t Dlra_,
		size_t cn,
		size_t width_,
		size_t height_,
		const StereoArraySGMParams& params)
		: width(width_), height(height_),
		Da(Da_),
		Dlra(Dlra_),
		Cbuf(NULL),
		Sbuf(NULL),
		disp2cost(NULL),
		disp2ptr(NULL),
		tempBuf(NULL),
		Lr(2, (CostType*)NULL),
		minLr(2, (CostType*)NULL),
		clipTab(NULL)
	{
		const size_t TAB_SIZE = 256 + TAB_OFS * 2;
		costWidth = width * Da;
		hsumRows = 3;
		dirs = NR;
		dirs2 = NR2;
		// for each possible stereo match (img1(x,y) <=> img2(x-d,y))
		// we keep pixel difference cost (C) and the summary cost over NR directions (S).
		// we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
		area.allocate(Cbuf, costWidth * height, CV_SIMD_WIDTH); // summary cost over different (nDirs) directions
		area.allocate(Sbuf, costWidth * height, CV_SIMD_WIDTH);
		area.allocate(disp2cost, width, CV_SIMD_WIDTH);
		area.allocate(disp2ptr, width, CV_SIMD_WIDTH);
		area.allocate(tempBuf, width * (6 * cn + 3), CV_SIMD_WIDTH);
		// the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
		// for 8-way dynamic programming we need the current row and
		// the previous row, i.e. 2 rows in total
		for (size_t i = 0; i < 2; ++i)
		{
			// 2D: [ NR ][ w1 * NR2 ][ NR ] * [ Dlra ]
			area.allocate(Lr[i], calcLrCount() * Dlra, CV_SIMD_WIDTH);
			// 1D: [ NR ][ w1 * NR2 ][ NR ]
			area.allocate(minLr[i], calcLrCount(), CV_SIMD_WIDTH);
		}
		area.commit();
	}
	inline void initCBuf(CostType val) const
	{
		for (size_t i = 0; i < costWidth * height; ++i)
			Cbuf[i] = val;
	}
	inline void clearLr(const Range& range = Range::all()) const
	{
		for (uchar i = 0; i < 2; ++i)
		{
			if (range == Range::all())
			{
				memset(Lr[i], 0, calcLrCount() * Dlra * sizeof(CostType));
				memset(minLr[i], 0, calcLrCount() * sizeof(CostType));
			}
			else
			{
				memset(getLr(i, range.start), 0, range.size() * sizeof(CostType) * Dlra);
				memset(getMinLr(i, range.start), 0, range.size() * sizeof(CostType));
			}
		}
	}
	inline size_t calcLrCount() const
	{
		return width * dirs2 + 2 * dirs;
	}
	inline void swapLr()
	{
		std::swap(Lr[0], Lr[1]);
		std::swap(minLr[0], minLr[1]);
	}
	inline CostType* getCBuf(int row) const
	{
		CV_Assert(row >= 0);
		return Cbuf + row * costWidth;
	}
	inline CostType* getSBuf(int row) const
	{
		CV_Assert(row >= 0);
		return Sbuf + row * costWidth;
	}
	inline void clearSBuf(int row, const Range& range = Range::all()) const
	{
		if (range == Range::all())
			memset(getSBuf(row), 0, costWidth * sizeof(CostType));
		else
			memset(getSBuf(row) + range.start * Da, 0, range.size() * Da * sizeof(CostType));
	}
	inline void clearSBuf()
	{
		for (size_t i = 0; i < costWidth * height; ++i)
			Sbuf[i] = 0;
	}

	// shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
	// and will occasionally use negative indices with the arrays
	// we need to shift Lr[k] pointers by 1, to give the space for d=-1.
	inline CostType* getLr(uchar id, int idx, uchar shift = 0) const
	{
		CV_Assert(id < 2);
		const size_t fixed_offset = dirs * Dlra;
		return Lr[id] + fixed_offset + (idx * (int)dirs2 + (int)shift) * (int)Dlra;
	}
	inline CostType* getMinLr(uchar id, int idx, uchar shift = 0) const
	{
		CV_Assert(id < 2);
		const size_t fixed_offset = dirs;
		return minLr[id] + fixed_offset + (idx * dirs2 + shift);
	}
};


/*
	computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
	that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
	minD <= d < maxD.
	disp2full is the reverse disparity map, that is:
	disp2full(x+roi.x,y+roi.y)=d means that img2(x+roi.x, y+roi.y) ~ img1(x+roi.x+d, y+roi.y)
	note that disp1buf will have the same size as the roi and
	disp2full will have the same size as img1 (or img2).
	On exit disp2buf is not the final disparity, it is an intermediate result that becomes
	final after all the tiles are processed.
	the disparity in disp1buf is written with sub-pixel accuracy
	(4 fractional bits, see StereoSGBM::DISP_SCALE),
	using quadratic interpolation, while the disparity in disp2buf
	is written as is, without interpolation.
	disp2cost also has the same size as img1 (or img2).
	It contains the minimum current cost, used to find the best disparity, corresponding to the minimal cost.
	*/
static void computeDisparityArraySGM(std::vector<cv::Mat>& LBPs, std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Point2f> directions, cv::Rect area,
	Mat& disp1, std::vector<int>& usedCams, const StereoArraySGMParams& params, cv::Mat mask)
{

	const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
	const int DISP_SCALE = (1 << DISP_SHIFT);
	const CostType MAX_COST = SHRT_MAX;

#pragma region LoadParameters
	int minD = params.minDisparity;// Minimum disparity
	int maxD = minD + params.numDisparities;// Maximum disparity
	int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
	int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
	int P1 = params.P1, P2 = params.P2;

	int width = area.width, height = area.height;
	const int D = params.numDisparities; // Number of disparities
	int Da = (int)alignSize(D, v_int16::nlanes);	// Aligned number of disparities
	int minX1 = 0; /* Min pos to stay within disparity in im1 */
	int maxX1 = width; //std::min(minD, 0); /* Max pos to stay within disparity in im1 */
	int minY1 = 0;
	int maxY1 = height; /* Max pos to stay within disparity in im1 */
	int width1 = maxX1 - minX1; // Width in im1 that can actually be used for disparity
	int height1 = maxY1 - minY1;
	int Dlra = Da + v_int16::nlanes;//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
	int INVALID_DISP = 0, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int npasses = 2;
#pragma endregion
	std::cout << "Calculating disparity with an array of " << directions.size() << " cameras..." << std::endl;
	std::cout << "Calculating disparities from " << minD << " to " << maxD << std::endl;
	// Return false when iteration is not possible
	if (minX1 >= maxX1)
	{
		disp1 = Scalar::all(INVALID_DISP_SCALED);
		return;
	}
	// Create a buffer memory to store all operation results
	BufferSGBM mem(Da, Dlra, images[0].channels(), width, height, params);
	mem.initCBuf((CostType)P2); // add P2 to every C(x,y). it saves a few operations in the inner loops
	mem.clearSBuf();
	CostType* const C = mem.getCBuf(0);



	if (surfaceParallelity.size() == images.size())
	{
		std::cout << "Calculating pixelwise cost without weights" << std::endl;
		calcPixelwiseArrayCostWithSurfaceNorms(LBPs, surfaceParallelity, directions, area, minD, maxD, C);
	}
	else {
		std::cout << "Calculating pixelwise cost without weights" << std::endl;
		calcPixelwiseArrayCostFlip(LBPs, images, directions, area, mask, usedCams, minD, maxD, C);
	}
	std::cout << "Pixelwise cost calculated" << std::endl;
	Mat_<uchar> areaMask = mask(area);

	for (int pass = 1; pass <= npasses; pass++)
	{
		int x1, y1, x2, y2, dx, dy;

		if (pass == 1)	// Forward pass
		{
			y1 = 0; y2 = area.height; dy = 1;
			x1 = 0; x2 = area.width; dx = 1;

		}
		else			// Backwards pass
		{
			y1 = area.height - 1; y2 =  -1; dy = -1;
			x1 = area.width - 1; x2 = -1; dx = -1;
		}

		uchar lrID = 0;
		mem.clearLr();

		for (int y = y1; y != y2; y += dy)	// Iterate through all rows
		{
			int x, d;
			DispType* disp1ptr = disp1.ptr<DispType>(y);
			CostType* const C = mem.getCBuf(y);
			CostType* const S = mem.getSBuf(y);

			for (x = x1; x != x2; x += dx)
			{
				if (areaMask(y, x) == 0)
				{
					continue;
				}

				int delta0 = P2 + *mem.getMinLr(lrID, x - dx);
				int delta1 = P2 + *mem.getMinLr(1 - lrID, x - 1, 1);
				int delta2 = P2 + *mem.getMinLr(1 - lrID, x, 2);
				int delta3 = P2 + *mem.getMinLr(1 - lrID, x + 1, 3);

				CostType* Lr_p0 = mem.getLr(lrID, x - dx);
				CostType* Lr_p1 = mem.getLr(1 - lrID, x - 1, 1);
				CostType* Lr_p2 = mem.getLr(1 - lrID, x, 2);
				CostType* Lr_p3 = mem.getLr(1 - lrID, x + 1, 3);

				Lr_p0[-1] = Lr_p0[D] = MAX_COST;
				Lr_p1[-1] = Lr_p1[D] = MAX_COST;
				Lr_p2[-1] = Lr_p2[D] = MAX_COST;
				Lr_p3[-1] = Lr_p3[D] = MAX_COST;

				CostType* Lr_p = mem.getLr(lrID, x);
				const CostType* Cp = C + x * Da;
				CostType* Sp = S + x * Da;

				CostType* minL = mem.getMinLr(lrID, x);
				d = 0;
				v_int16 _P1 = vx_setall_s16((short)P1);

				v_int16 _delta0 = vx_setall_s16((short)delta0);
				v_int16 _delta1 = vx_setall_s16((short)delta1);
				v_int16 _delta2 = vx_setall_s16((short)delta2);
				v_int16 _delta3 = vx_setall_s16((short)delta3);
				v_int16 _minL0 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL1 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL2 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL3 = vx_setall_s16((short)MAX_COST);

				for (; d <= D - v_int16::nlanes; d += v_int16::nlanes)
				{
					v_int16 Cpd = vx_load_aligned(Cp + d);
					v_int16 Spd = vx_load_aligned(Sp + d);
					v_int16 L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p0 + d), vx_load(Lr_p0 + d - 1) + _P1), vx_load(Lr_p0 + d + 1) + _P1), _delta0) - _delta0 + Cpd;
					v_store_aligned(Lr_p + d, L);
					_minL0 = v_min(_minL0, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p1 + d), vx_load(Lr_p1 + d - 1) + _P1), vx_load(Lr_p1 + d + 1) + _P1), _delta1) - _delta1 + Cpd;
					v_store_aligned(Lr_p + d + Dlra, L);
					_minL1 = v_min(_minL1, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p2 + d), vx_load(Lr_p2 + d - 1) + _P1), vx_load(Lr_p2 + d + 1) + _P1), _delta2) - _delta2 + Cpd;
					v_store_aligned(Lr_p + d + Dlra * 2, L);
					_minL2 = v_min(_minL2, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p3 + d), vx_load(Lr_p3 + d - 1) + _P1), vx_load(Lr_p3 + d + 1) + _P1), _delta3) - _delta3 + Cpd;
					v_store_aligned(Lr_p + d + Dlra * 3, L);
					_minL3 = v_min(_minL3, L);
					Spd += L;

					v_store_aligned(Sp + d, Spd);
				}
				// Get minimum for L0-L3
				v_int16 t0, t1, t2, t3;
				v_zip(_minL0, _minL2, t0, t2);
				v_zip(_minL1, _minL3, t1, t3);
				v_zip(v_min(t0, t2), v_min(t1, t3), t0, t1);
				t0 = v_min(t0, t1);
				t0 = v_min(t0, v_rotate_right<4>(t0));

				v_store_low(minL, t0);

				for (; d < D; d++)
				{
					int Cpd = Cp[d], L;
					int Spd = Sp[d];

					L = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) - delta0;
					Lr_p[d] = (CostType)L;
					minL[0] = std::min(minL[0], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1))) - delta1;
					Lr_p[d + Dlra] = (CostType)L;
					minL[1] = std::min(minL[1], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2))) - delta2;
					Lr_p[d + Dlra * 2] = (CostType)L;
					minL[2] = std::min(minL[2], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3))) - delta3;
					Lr_p[d + Dlra * 3] = (CostType)L;
					minL[3] = std::min(minL[3], (CostType)L);
					Spd += L;

					Sp[d] = saturate_cast<CostType>(Spd);
				}
			}

			if (pass == npasses)
			{
				x = 0;
				#pragma region Set_disparity_maps_to INVALID_DISP SCALED
				v_int16 v_inv_dist = vx_setall_s16((DispType)INVALID_DISP_SCALED);
				v_int16 v_max_cost = vx_setall_s16(MAX_COST);
				for (; x <= area.width - v_int16::nlanes; x += v_int16::nlanes)
				{
					v_store(disp1ptr + x, v_inv_dist);
					v_store(mem.disp2ptr + x, v_inv_dist);
					v_store(mem.disp2cost + x, v_max_cost);
				}

				for (; x < area.width; x++)
				{
					disp1ptr[x] = mem.disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
					mem.disp2cost[x] = MAX_COST;
				}
				#pragma endregion
				for (x = area.width - 1; x >= 0; x--)
				{
					CostType* Sp = S + x * Da;
					CostType minS = MAX_COST;
					short bestDisp = -1;

					d = 0;
					#pragma region vx speedup
					v_int16 _minS = vx_setall_s16(MAX_COST), _bestDisp = vx_setall_s16(-1);
					for (; d <= D - v_int16::nlanes; d += v_int16::nlanes)
					{
						v_int16 L0 = vx_load_aligned(Sp + d);
						_bestDisp = v_select(_minS > L0, vx_setall_s16((short)d), _bestDisp);
						_minS = v_min(L0, _minS);
					}
					min_pos(_minS, _bestDisp, minS, bestDisp);
					#pragma endregion
					for (; d < D; d++)
					{
						int Sval = Sp[d];
						if (Sval < minS)
						{
							minS = (CostType)Sval;
							bestDisp = (short)d;
						}
					}

					for (d = 0; d < D; d++)
					{
						if (Sp[d] * (100 - uniquenessRatio) < minS * 100 && std::abs(bestDisp - d) > 1)
							break;
					}
					if (d < D)
						continue;
					d = bestDisp;
					//int _x2 = x + minX1 - d - minD;
					//if (mem.disp2cost[_x2] > minS)
					//{
					//	mem.disp2cost[_x2] = (CostType)minS;
					//	mem.disp2ptr[_x2] = (DispType)(d + minD);
					//}
					if (d < 0) std::cout << "PANIC! d<0" << std::endl;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
						d = d * DISP_SCALE + ((Sp[d - 1] - Sp[d + 1]) * DISP_SCALE + denom2) / (denom2 * 2);
					}
					else
						d *= DISP_SCALE;
					disp1ptr[x] = (DispType)(d + minD * DISP_SCALE);
				}

				//for (x = 0; x < area.width; x++)
				//{
				//	// we round the computed disparity both towards -inf and +inf and check
				//	// if either of the corresponding disparities in disp2 is consistent.
				//	// This is to give the computed disparity a chance to look valid if it is.
				//	int d1 = disp1ptr[x];
				//	if (d1 == INVALID_DISP_SCALED)
				//		continue;
				//	int _d = d1 >> DISP_SHIFT;
				//	int d_ = (d1 + DISP_SCALE - 1) >> DISP_SHIFT;
				//	int _x = x - _d, x_ = x - d_;
				//	if (0 <= _x && _x < width && mem.disp2ptr[_x] >= minD && std::abs(mem.disp2ptr[_x] - _d) > disp12MaxDiff &&
				//		0 <= x_ && x_ < width && mem.disp2ptr[x_] >= minD && std::abs(mem.disp2ptr[x_] - d_) > disp12MaxDiff)
				//		disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
				//}
			}

			lrID = 1 - lrID; // now shift the cyclic buffers
		}
	}
}

static void computeMinCostArraySGM(std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Point2f> directions, cv::Rect area,
	Mat& disp1, std::vector<int>& usedCams, const StereoArraySGMParams& params)
{
	std::cout << "Calculating minimum cost with an array of " << directions.size() << " cameras." << std::endl;
	const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
	const int DISP_SCALE = (1 << DISP_SHIFT);
	const CostType MAX_COST = SHRT_MAX;

#pragma region LoadParameters
	int minD = params.minDisparity;// Minimum disparity
	int maxD = minD + params.numDisparities;// Maximum disparity
	int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
	int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
	int P1 = params.P1, P2 = params.P2;

	int width = area.width, height = area.height;
	const int D = params.numDisparities; // Number of disparities
	int Da = (int)alignSize(D, v_int16::nlanes);	// Aligned number of disparities
	int minX1 = 0; /* Min pos to stay within disparity in im1 */
	int maxX1 = width; //std::min(minD, 0); /* Max pos to stay within disparity in im1 */
	int minY1 = 0;
	int maxY1 = height; /* Max pos to stay within disparity in im1 */
	int width1 = maxX1 - minX1; // Width in im1 that can actually be used for disparity
	int height1 = maxY1 - minY1;
	int Dlra = Da + v_int16::nlanes;//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
	int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int npasses = 2;
#pragma endregion

	// Return false when iteration is not possible
	if (minX1 >= maxX1)
	{
		disp1 = Scalar::all(INVALID_DISP_SCALED);
		return;
	}
	// Create a buffer memory to store all operation results
	BufferSGBM mem(Da, Dlra, images[0].channels(), width, height, params);
	mem.initCBuf((CostType)P2); // add P2 to every C(x,y). it saves a few operations in the inner loops
	mem.clearSBuf();
	CostType* const C = mem.getCBuf(0);
	if (surfaceParallelity.size() == images.size())
	{
		std::cout << "Calculating pixelwise cost without weights" << std::endl;
		calcPixelwiseArrayCostWithSurfaceNorms(images, surfaceParallelity, directions, area, minD, maxD, C);
	}
	else {
		std::cout << "Calculating pixelwise cost without weights" << std::endl;
		//calcPixelwiseArrayCostFlip(images, directions, area, usedCams, minD, maxD, C);
	}
	for (int pass = 1; pass <= npasses; pass++)
	{
		int x1, y1, x2, y2, dx, dy;

		if (pass == 1)	// Forward pass
		{
			y1 = 0; y2 = area.height; dy = 1;
			x1 = 0; x2 = area.width; dx = 1;

		}
		else			// Backwards pass
		{
			y1 = area.height - 1; y2 = -1; dy = -1;
			x1 = area.width - 1; x2 = -1; dx = -1;
		}

		uchar lrID = 0;
		mem.clearLr();

		for (int y = y1; y != y2; y += dy)	// Iterate through all rows
		{
			int x, d;
			DispType* disp1ptr = disp1.ptr<DispType>(y);
			CostType* const C = mem.getCBuf(y);
			CostType* const S = mem.getSBuf(y);

			for (x = x1; x != x2; x += dx)
			{
				int delta0 = P2 + *mem.getMinLr(lrID, x - dx);
				int delta1 = P2 + *mem.getMinLr(1 - lrID, x - 1, 1);
				int delta2 = P2 + *mem.getMinLr(1 - lrID, x, 2);
				int delta3 = P2 + *mem.getMinLr(1 - lrID, x + 1, 3);

				CostType* Lr_p0 = mem.getLr(lrID, x - dx);
				CostType* Lr_p1 = mem.getLr(1 - lrID, x - 1, 1);
				CostType* Lr_p2 = mem.getLr(1 - lrID, x, 2);
				CostType* Lr_p3 = mem.getLr(1 - lrID, x + 1, 3);

				Lr_p0[-1] = Lr_p0[D] = MAX_COST;
				Lr_p1[-1] = Lr_p1[D] = MAX_COST;
				Lr_p2[-1] = Lr_p2[D] = MAX_COST;
				Lr_p3[-1] = Lr_p3[D] = MAX_COST;

				CostType* Lr_p = mem.getLr(lrID, x);
				const CostType* Cp = C + x * Da;
				CostType* Sp = S + x * Da;

				CostType* minL = mem.getMinLr(lrID, x);
				d = 0;
				v_int16 _P1 = vx_setall_s16((short)P1);

				v_int16 _delta0 = vx_setall_s16((short)delta0);
				v_int16 _delta1 = vx_setall_s16((short)delta1);
				v_int16 _delta2 = vx_setall_s16((short)delta2);
				v_int16 _delta3 = vx_setall_s16((short)delta3);
				v_int16 _minL0 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL1 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL2 = vx_setall_s16((short)MAX_COST);
				v_int16 _minL3 = vx_setall_s16((short)MAX_COST);

				for (; d <= D - v_int16::nlanes; d += v_int16::nlanes)
				{
					v_int16 Cpd = vx_load_aligned(Cp + d);
					v_int16 Spd = vx_load_aligned(Sp + d);
					v_int16 L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p0 + d), vx_load(Lr_p0 + d - 1) + _P1), vx_load(Lr_p0 + d + 1) + _P1), _delta0) - _delta0 + Cpd;
					v_store_aligned(Lr_p + d, L);
					_minL0 = v_min(_minL0, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p1 + d), vx_load(Lr_p1 + d - 1) + _P1), vx_load(Lr_p1 + d + 1) + _P1), _delta1) - _delta1 + Cpd;
					v_store_aligned(Lr_p + d + Dlra, L);
					_minL1 = v_min(_minL1, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p2 + d), vx_load(Lr_p2 + d - 1) + _P1), vx_load(Lr_p2 + d + 1) + _P1), _delta2) - _delta2 + Cpd;
					v_store_aligned(Lr_p + d + Dlra * 2, L);
					_minL2 = v_min(_minL2, L);
					Spd += L;

					L = v_min(v_min(v_min(vx_load_aligned(Lr_p3 + d), vx_load(Lr_p3 + d - 1) + _P1), vx_load(Lr_p3 + d + 1) + _P1), _delta3) - _delta3 + Cpd;
					v_store_aligned(Lr_p + d + Dlra * 3, L);
					_minL3 = v_min(_minL3, L);
					Spd += L;

					v_store_aligned(Sp + d, Spd);
				}
				// Get minimum for L0-L3
				v_int16 t0, t1, t2, t3;
				v_zip(_minL0, _minL2, t0, t2);
				v_zip(_minL1, _minL3, t1, t3);
				v_zip(v_min(t0, t2), v_min(t1, t3), t0, t1);
				t0 = v_min(t0, t1);
				t0 = v_min(t0, v_rotate_right<4>(t0));

				v_store_low(minL, t0);

				for (; d < D; d++)
				{
					int Cpd = Cp[d], L;
					int Spd = Sp[d];

					L = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) - delta0;
					Lr_p[d] = (CostType)L;
					minL[0] = std::min(minL[0], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1))) - delta1;
					Lr_p[d + Dlra] = (CostType)L;
					minL[1] = std::min(minL[1], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2))) - delta2;
					Lr_p[d + Dlra * 2] = (CostType)L;
					minL[2] = std::min(minL[2], (CostType)L);
					Spd += L;

					L = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3))) - delta3;
					Lr_p[d + Dlra * 3] = (CostType)L;
					minL[3] = std::min(minL[3], (CostType)L);
					Spd += L;

					Sp[d] = saturate_cast<CostType>(Spd);
				}
			}

			if (pass == npasses)
			{
				x = 0;

				v_int16 v_inv_dist = vx_setall_s16((DispType)INVALID_DISP_SCALED);
				v_int16 v_max_cost = vx_setall_s16(MAX_COST);
#pragma region Set_disparity_maps_to INVALID_DISP SCALED
				for (; x <= area.width - v_int16::nlanes; x += v_int16::nlanes)
				{
					v_store(disp1ptr + x, v_inv_dist);
					v_store(mem.disp2ptr + x, v_inv_dist);
					v_store(mem.disp2cost + x, v_max_cost);
				}

				for (; x < area.width; x++)
				{
					disp1ptr[x] = mem.disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
					mem.disp2cost[x] = MAX_COST;
				}
#pragma endregion
				for (x = area.width - 1; x >= 0; x--)
				{
					CostType* Sp = S + x * Da;
					CostType minS = MAX_COST;
					short bestDisp = -1;

					d = 0;
					v_int16 _minS = vx_setall_s16(MAX_COST), _bestDisp = vx_setall_s16(-1);
					for (; d <= D - v_int16::nlanes; d += v_int16::nlanes)
					{
						v_int16 L0 = vx_load_aligned(Sp + d);
						_bestDisp = v_select(_minS > L0, vx_setall_s16((short)d), _bestDisp);
						_minS = v_min(L0, _minS);
					}
					min_pos(_minS, _bestDisp, minS, bestDisp);

					for (; d < D; d++)
					{
						int Sval = Sp[d];
						if (Sval < minS)
						{
							minS = (CostType)Sval;
							bestDisp = (short)d;
						}
					}

					for (d = 0; d < D; d++)
					{
						if (Sp[d] * (100 - uniquenessRatio) < minS * 100 && std::abs(bestDisp - d) > 1)
							break;
					}
					if (d < D)
						continue;
					d = bestDisp;
					int _x2 = x + minX1 - d - minD;
					if (mem.disp2cost[_x2] > minS)
					{
						mem.disp2cost[_x2] = (CostType)minS;
						mem.disp2ptr[_x2] = (DispType)(d + minD);
					}

					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
						d = d * DISP_SCALE + ((Sp[d - 1] - Sp[d + 1]) * DISP_SCALE + denom2) / (denom2 * 2);
					}
					else
						d *= DISP_SCALE;
					disp1ptr[x] = (DispType)(minS);
				}

			}

			lrID = 1 - lrID; // now shift the cyclic buffers
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

StereoArraySGBM::StereoArraySGBM(int _minDisparity, int _numDisparities,
	int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
	int _uniquenessRatio, int _speckleWindowSize, int _speckleRange)
	: params(_minDisparity, _numDisparities,
		_P1, _P2, _disp12MaxDiff, _preFilterCap,
		_uniquenessRatio, _speckleWindowSize, _speckleRange){}

void StereoArraySGBM::compute(std::vector<cv::Mat>& LBPs, std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Camera> cams,
	int centerCamID, cv::Rect area, cv::OutputArray disparr, double camDistance, cv::Mat mask, std::vector<int> usedCams){
	CV_INSTRUMENT_REGION();
	disparr.create(area.size(), CV_16S);
	Mat disp = disparr.getMat();
	CV_Assert((area & cv::Rect(0, 0, images[0].cols, images[0].rows)) == area); // Check if defined area is in image
	CV_Assert(images.size() == cams.size()); // Check if the right amount of images is present

	double dist = camDistance;
	Point2f centerPos{ (float)-cams[centerCamID].pos3D.x, (float)cams[centerCamID].pos3D.y };
	std::vector<Point2f> directions;
	std::cout << "Disparity directions: ";
	for (auto &c : cams)
	{
		Point2d direction = (Point2f{ (float)-c.pos3D.x, (float)c.pos3D.y } - centerPos)/dist;
		directions.push_back(Point2f(direction));
		std::cout << directions.back() << ", ";
	}
	std::cout << std::endl;
	computeDisparityArraySGM(LBPs, images, surfaceParallelity, directions, area, disp, usedCams, params, mask);

	cv::medianBlur(disp, disp, 3);

	//if (params.speckleWindowSize > 0)
	//	filterSpeckles(disp, (params.minDisparity - 1) * StereoMatcher::DISP_SCALE, params.speckleWindowSize,
	//		StereoMatcher::DISP_SCALE * params.speckleRange, buffer);
}

void StereoArraySGBM::computeMinCost(std::vector<cv::Mat>& images, std::vector<cv::Mat>& surfaceParallelity, std::vector<Camera> cams,
	int centerCamID, cv::Rect area, cv::OutputArray disparr, double camDistance, std::vector<int> usedCams)
{
	CV_INSTRUMENT_REGION();
	disparr.create(area.size(), CV_16S);
	Mat disp = disparr.getMat();
	CV_Assert((area & cv::Rect(0, 0, images[0].cols, images[0].rows)) == area); // Check if defined area is in image
	CV_Assert(images.size() == cams.size()); // Check if the right amount of images is present

	//double dist = std::max(abs(cams[0].pos3D.x - cams[1].pos3D.x), abs(cams[0].pos3D.y - cams[1].pos3D.y));
	double dist = camDistance;
	Point2f centerPos{ (float)-cams[centerCamID].pos3D.x, (float)cams[centerCamID].pos3D.y };
	std::vector<Point2f> directions;
	std::cout << "Disparity directions: ";
	for (auto& c : cams)
	{
		Point2d direction = (Point2f{ (float)-c.pos3D.x, (float)c.pos3D.y } - centerPos) / dist;
		directions.push_back(Point2f(direction));
		std::cout << directions.back() << ", ";
	}
	std::cout << std::endl;
	computeMinCostArraySGM(images, surfaceParallelity, directions, area, disp, usedCams, params);

	cv::medianBlur(disp, disp, 3);

	if (params.speckleWindowSize > 0)
		filterSpeckles(disp, (params.minDisparity - 1) * StereoMatcher::DISP_SCALE, params.speckleWindowSize,
			StereoMatcher::DISP_SCALE * params.speckleRange, buffer);
}