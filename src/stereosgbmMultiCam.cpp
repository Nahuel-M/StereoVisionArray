
#include <limits.h>
#include <iostream>
#include <vector>
#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"
#include "stereosgbmMultiCam.h"
#include "imageHandling.h"


using namespace cv;

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

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

static const int DEFAULT_RIGHT_BORDER = -1;

/*
	For each pixel row1[x], max(maxD, 0) <= minX <= x < maxX <= width - max(0, -minD),
	and for each disparity minD<=d<maxD the function
	computes the cost (cost[(x-minX)*(maxD - minD) + (d - minD)]), depending on the difference between
	row1[x] and row2[x-d]. The subpixel algorithm from
	"Depth Discontinuities by Pixel-to-Pixel Stereo" by Stan Birchfield and C. Tomasi
	is used, hence the suffix BT.
	the temporary buffer should contain width2*2 elements
	*/
static void calcPixelCostBT2(const Mat& img1, const Mat& img2,
	int minD, int maxD, CostType* costOrigin, int rowStep, int colStep, int matchCount, int direction,
	PixType* buffer, const PixType* tab)
{
	#pragma region LoadParameters
	int x, c, width = img1.cols, cn = img1.channels();
	//int minX1 = std::max(maxD, 0), maxX1 = width - std::max(maxD, 0);
	int minX1 = maxD, maxX1 = width - maxD;
	int D = (int)alignSize(maxD - minD, v_int16::nlanes);
	int width1 = maxX1 - minX1;	// Actual width to be iterated over in img1
	int height = img1.rows;
	#pragma endregion
	costOrigin -= minD + maxD * colStep + maxD * rowStep;
	for (int y = maxD; y < height - maxD; y++)
	{
		CostType* costY = costOrigin + y * rowStep;
		const PixType* row1 = img1.ptr<PixType>(y), * row2 = img2.ptr<PixType>(y);
		PixType* prow1 = buffer + width * 2;
		PixType* prow2 = prow1 + width * 2;

		for (c = 0; c < cn * 2; c++)
		{
			prow1[width * c] = prow1[width * c + width - 1] =
				prow2[width * c] = prow2[width * c + width - 1] = tab[0];
		}

		int n = y > 0 ? -(int)img1.step : 0, s = y < img1.rows - 1 ? (int)img1.step : 0;

		for (x = 0; x < width; x++)
		{
			prow1[x] = tab[(row1[x + 1] - row1[x - 1]) * 2 + row1[x + n + 1] - row1[x + n - 1] + row1[x + s + 1] - row1[x + s - 1]];
			prow2[width - 1 - x] = tab[(row2[x + 1] - row2[x - 1]) * 2 + row2[x + n + 1] - row2[x + n - 1] + row2[x + s + 1] - row2[x + s - 1]];

			prow1[x + width] = row1[x];
			prow2[width - 1 - x + width] = row2[x];
		}

		for (c = 0; c < 2; c++, prow1 += width, prow2 += width)
		{
			int diff_scale = c * 2;
			// precompute
			//   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
			//   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
			//   to process values from [minX2, maxX2) we should check memory location (width - 1 - maxX2, width - 1 - minX2]
			//   so iterate through [width - maxX2, width - minX2)
			for (x = 1; x < width - 1; x++)
			{
				int v = prow2[x];
				int vl = x > 0 ? (v + prow2[x - 1]) / 2 : v;
				int vr = x < width - 1 ? (v + prow2[x + 1]) / 2 : v;
				int v0 = std::min(vl, vr); v0 = std::min(v0, v);
				int v1 = std::max(vl, vr); v1 = std::max(v1, v);
				buffer[x] = (PixType)v0;
				buffer[x + width] = (PixType)v1;
			}

			for (x = minX1; x < maxX1; x++)
			{
				CostType* costYX = costY + x * colStep;
				int u = prow1[x];
				int ul = x > 0 ? (u + prow1[x - 1]) / 2 : u;
				int ur = x < width - 1 ? (u + prow1[x + 1]) / 2 : u;
				int u0 = std::min(ul, ur); u0 = std::min(u0, u);
				int u1 = std::max(ul, ur); u1 = std::max(u1, u);

				for (int d = minD; d < maxD; d++)
				{
					int directedD = d * direction;
					int v = prow2[width - x - 1 + directedD];
					int v0 = buffer[width - x - 1 + directedD];
					int v1 = buffer[width - x - 1 + directedD + width];
					int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
					int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

					costYX[d] =
						(CostType)(costYX[d] +
						(min(c0, c1) / matchCount >> diff_scale));
				}
			}
		}
	}
}


class BufferSGBM
{
private:
	size_t width1;
	size_t height1;
	size_t Da;
	size_t Dlra;
	size_t costWidth;
	size_t costHeight;
	size_t height;
	size_t width;
	size_t hsumRows;
	bool fullDP;
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
	BufferSGBM(size_t width1_, size_t height1_,
		size_t Da_,
		size_t Dlra_,
		size_t cn,
		size_t width_,
		size_t height_,
		const StereoSGBMParams2& params)
		: width(width_), height(height_), width1(width1_), height1(height1_),
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
		fullDP = params.isFullDP();
		costWidth = width1 * Da;
		costHeight = height1 * Da;
		hsumRows = 3;
		dirs = NR;
		dirs2 = NR2;
		// for each possible stereo match (img1(x,y) <=> img2(x-d,y))
		// we keep pixel difference cost (C) and the summary cost over NR directions (S).
		// we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
		area.allocate(Cbuf, costWidth * height1, CV_SIMD_WIDTH); // summary cost over different (nDirs) directions
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
		area.allocate(clipTab, TAB_SIZE, CV_SIMD_WIDTH);
		area.commit();

		// init clipTab
		const int ftzero = std::max(params.preFilterCap, 15) | 1;
		for (int i = 0; i < (int)TAB_SIZE; i++) {
			clipTab[i] = (PixType)(std::min(std::max(i - (int)TAB_OFS, -ftzero), ftzero) + ftzero);
		}
	}
	inline const PixType* getClipTab() const
	{
		return clipTab + TAB_OFS;
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
		return width1 * dirs2 + 2 * dirs;
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


static void calcPixelCostBT2MultiCam(Mat& img1, Mat& img2, Mat& img3,
	int minD, int maxD, CostType* costOrigin, int rowStep, int colStep, int matchCount, PixType* buffer, const PixType* tab)
{
	#pragma region LoadParameters
	int x, c, width = img1.cols, cn = img1.channels();
	//int minX1 = std::max(maxD, 0), maxX1 = width - std::max(maxD, 0);
	int minX1 = maxD, maxX1 = width - maxD;
	int D = (int)alignSize(maxD - minD, v_int16::nlanes);
	int width1 = maxX1 - minX1;	// Actual width to be iterated over in img1
	int height = img1.rows;
	#pragma endregion
	costOrigin -= minD    +    maxD * colStep    +    maxD * rowStep;
	for (int y = maxD; y < height-maxD; y++)
	{
		CostType* costY = costOrigin + y * rowStep;
		//CostType* cost = costOrigin + y * rowStep;
		const PixType* row1 = img1.ptr<PixType>(y), * row2 = img2.ptr<PixType>(y), * row3 = img3.ptr<PixType>(y);
		PixType* prow1 = buffer + width * 4;
		PixType* prow2 = prow1 + width * 2;
		PixType* prow3 = prow2 + width * 2;
		for (c = 0; c < 2; c++)
		{
			prow1[width * c] = prow1[width * c + width - 1] =
				prow2[width * c] = prow2[width * c + width - 1] =
				prow3[width * c] = prow3[width * c + width - 1] = tab[0];
		}

		int n = y > 0 ? -(int)img1.step : 0, s = y < img1.rows - 1 ? (int)img1.step : 0;

		for (x = 0; x < width; x++)
		{
			prow1[x] = tab[(row1[x + 1] - row1[x - 1]) * 2 + row1[x + n + 1] - row1[x + n - 1] + row1[x + s + 1] - row1[x + s - 1]];
			prow2[width - 1 - x] = tab[(row2[x + 1] - row2[x - 1]) * 2 + row2[x + n + 1] - row2[x + n - 1] + row2[x + s + 1] - row2[x + s - 1]];
			prow3[width - 1 - x] = tab[(row3[x + 1] - row3[x - 1]) * 2 + row3[x + n + 1] - row3[x + n - 1] + row3[x + s + 1] - row3[x + s - 1]];

			prow1[x + width] = row1[x];
			prow2[width - 1 - x + width] = row2[x];
			prow3[width - 1 - x + width] = row3[x];
		}

		//memset(cost, 0, width1 * D * sizeof(cost[0]));

		for (c = 0; c < cn * 2; c++, prow1 += width, prow2 += width)
		{
			int diff_scale = c * 2;
			for (x = 1; x < width-1; x++)
			{
				int v = prow2[x];
				int vl = x > 0 ? (v + prow2[x - 1]) / 2 : v;
				int vr = x < width - 1 ? (v + prow2[x + 1]) / 2 : v;
				int v0 = std::min(vl, vr); v0 = std::min(v0, v);
				int v1 = std::max(vl, vr); v1 = std::max(v1, v);
				buffer[x] = (PixType)v0;
				buffer[x + width] = (PixType)v1;

				int v_2 = prow3[x];
				int vl_2 = x > 0 ? (v_2 + prow3[x - 1]) / 2 : v_2;
				int vr_2 = x < width - 1 ? (v_2 + prow3[x + 1]) / 2 : v_2;
				int v0_2 = std::min(vl_2, vr_2); v0_2 = std::min(v0_2, v_2);
				int v1_2 = std::max(vl_2, vr_2); v1_2 = std::max(v1_2, v_2);
				buffer[x + 2 * width] = (PixType)v0_2;
				buffer[x + 3 * width] = (PixType)v1_2;
			}

			for (x = minX1; x < maxX1; x++)
			{
				CostType* costYX = costY + x * colStep;
				int u = prow1[x];
				int ul = x > 0 ? (u + prow1[x - 1]) / 2 : u;
				int ur = x < width - 1 ? (u + prow1[x + 1]) / 2 : u;
				int u0 = std::min(ul, ur); u0 = std::min(u0, u);
				int u1 = std::max(ul, ur); u1 = std::max(u1, u);

				for (int d = minD; d < maxD; d++)
				{
					int v = prow2[width - x - 1 + d];
					int v0 = buffer[width - x - 1 + d];
					int v1 = buffer[width - x - 1 + d + width];
					int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
					int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

					int v0_2 = buffer[width - x - 1 - d + width * 2];
					int v1_2 = buffer[width - x - 1 - d + width * 3];
					int v_2 = prow3[width - x - 1 + d];
					int c0_2 = std::max(0, u - v1_2); c0_2 = std::max(c0_2, v0_2 - u);
					int c1_2 = std::max(0, v_2 - u1); c1_2 = std::max(c1_2, u0 - v_2);

					costYX[d] =
						(CostType)(costYX[d] +
						((min(c0, c1) + min(c0_2, c1_2))/matchCount >> diff_scale));
				}
			}
		}
	}
}


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
static void computeDisparitySGBM2(std::vector<Mat> images, std::vector<cv::Point2i> disparityDirection,
	Mat& disp1, const StereoSGBMParams2& params)
{
	Mat img1 = images[0];
	//showImage("img1", img1);
	Mat img2 = images[1];
	const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
	const int DISP_SCALE = (1 << DISP_SHIFT);
	const CostType MAX_COST = SHRT_MAX;

	#pragma region LoadParameters
	int minD = params.minDisparity;// Minimum disparity
	int maxD = minD + params.numDisparities;// Maximum disparity
	int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
	int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
	int P1 = params.P1, P2 = params.P2;
	int width = disp1.cols, height = disp1.rows;
	const int D = params.numDisparities; // Number of disparities
	int Da = (int)alignSize(D, v_int16::nlanes);	// Aligned number of disparities
	int minX1 = std::max(maxD, 0); /* Min pos to stay within disparity in im1 */
	int maxX1 = width - std::max(maxD, 0); //std::min(minD, 0); /* Max pos to stay within disparity in im1 */
	int minY1 = std::max(maxD, 0);
	int maxY1 = height - std::max(maxD, 0); /* Max pos to stay within disparity in im1 */
	int width1 = maxX1 - minX1; // Width in im1 that can actually be used for disparity
	int height1 = maxY1 - minY1;
	std::cout << "D: " << D << ", Da: " << Da << std::endl;
	int Dlra = Da + v_int16::nlanes;//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
	int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int npasses = params.isFullDP() ? 2 : 1;
	#pragma endregion

	// Return false when iteration is not possible
	if (minX1 >= maxX1)
	{
		disp1 = Scalar::all(INVALID_DISP_SCALED);
		return;
	}
	// Create a buffer memory to store all operation results
	BufferSGBM mem(width1, height1, Da, Dlra, img1.channels(), width, height, params);
	mem.initCBuf((CostType)P2); // add P2 to every C(x,y). it saves a few operations in the inner loops

	std::vector<Mat> hImages; std::vector<int> hDir;
	std::vector<Mat> vImages; std::vector<int> vDir;

	std::cout << disparityDirection << std::endl;
	hImages.push_back(images[0]);
	vImages.push_back(images[0].t());
	for (size_t i = 0; i < disparityDirection.size(); i++)
	{
		if (disparityDirection[i].x != 0) {
			hImages.push_back(images[i+1]);
			hDir.push_back(disparityDirection[i].x);
		}
		else
		{
			vImages.push_back(images[i+1].t());
			vDir.push_back(-disparityDirection[i].y);
		}
	}


	CostType* C = mem.getCBuf(0);
	memset(C, 0, width1 * height * D * sizeof(C[0]));
	int matchCount = images.size() - 1;
	P1 *= matchCount;
	P2 *= matchCount;
	matchCount = 1;
	int rowStep = Da * width1;
	int colStep = Da;
	if (hImages.size() == 3) 
	{
		calcPixelCostBT2MultiCam(hImages[0], hImages[1], hImages[2], minD, maxD, C, rowStep, colStep, matchCount, mem.tempBuf, mem.getClipTab());
	}
	else if (hImages.size() == 2)
	{
		calcPixelCostBT2(hImages[0], hImages[1], minD, maxD, C, rowStep, colStep, matchCount, hDir[0], mem.tempBuf, mem.getClipTab());
	}
	rowStep = Da;
	colStep = Da * width1;
	if (vImages.size() == 3)
	{
		calcPixelCostBT2MultiCam(vImages[0], vImages[2], vImages[1], minD, maxD, C, rowStep, colStep, matchCount, mem.tempBuf, mem.getClipTab());
	}
	else if (vImages.size() == 2)
	{
		calcPixelCostBT2(vImages[0], vImages[1], minD, maxD, C, rowStep, colStep, matchCount, vDir[0], mem.tempBuf, mem.getClipTab());
	}

	for (int pass = 1; pass <= npasses; pass++)
	{
		int x1, y1, x2, y2, dx, dy;

		if (pass == 1)	// Forward pass
		{
			y1 = maxD; y2 = height-maxD; dy = 1;
			x1 = 0; x2 = width1; dx = 1;

		}
		else			// Backwards pass
		{
			y1 = height - 1-maxD; y2 = -1+maxD; dy = -1;
			x1 = width1 - 1; x2 = -1; dx = -1;
		}

		uchar lrID = 0;
		mem.clearLr();

		for (int y = y1; y != y2; y += dy)	// Iterate through all rows
		{
			int x, d;
			DispType* disp1ptr = disp1.ptr<DispType>(y);
			CostType* const C = mem.getCBuf(y-maxD);
			CostType* const S = mem.getSBuf(y);

			// compute C on the first pass, and reuse it on the second pass, if any.

			/*
				[formula 13 in the paper]
				compute L_r(p, d) = C(p, d) +
				min(L_r(p-r, d),
				L_r(p-r, d-1) + P1,
				L_r(p-r, d+1) + P1,
				min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
				where p = (x,y), r is one of the directions.
				we process all the directions at once:
				0: r=(-dx, 0)
				1: r=(-1, -dy)
				2: r=(0, -dy)
				3: r=(1, -dy)   !!!Note that only directions 0 to 3 are processed
				4: r=(-2, -dy)
				5: r=(-1, -dy*2)
				6: r=(1, -dy*2)
				7: r=(2, -dy)
				*/

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
				for (; x <= width - v_int16::nlanes; x += v_int16::nlanes)
				{
					v_store(disp1ptr + x, v_inv_dist);
					v_store(mem.disp2ptr + x, v_inv_dist);
					v_store(mem.disp2cost + x, v_max_cost);
				}

				for (; x < width; x++)
				{
					disp1ptr[x] = mem.disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
					mem.disp2cost[x] = MAX_COST;
				}
				#pragma endregion
				for (x = width1 - 1; x >= 0; x--)
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
					disp1ptr[x + minX1] = (DispType)(d + minD * DISP_SCALE);
				}

				for (x = minX1; x < maxX1; x++)
				{
					// we round the computed disparity both towards -inf and +inf and check
					// if either of the corresponding disparities in disp2 is consistent.
					// This is to give the computed disparity a chance to look valid if it is.
					int d1 = disp1ptr[x];
					if (d1 == INVALID_DISP_SCALED)
						continue;
					int _d = d1 >> DISP_SHIFT;
					int d_ = (d1 + DISP_SCALE - 1) >> DISP_SHIFT;
					int _x = x - _d, x_ = x - d_;
					if (0 <= _x && _x < width && mem.disp2ptr[_x] >= minD && std::abs(mem.disp2ptr[_x] - _d) > disp12MaxDiff &&
						0 <= x_ && x_ < width && mem.disp2ptr[x_] >= minD && std::abs(mem.disp2ptr[x_] - d_) > disp12MaxDiff)
						disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
				}
			}

			lrID = 1 - lrID; // now shift the cyclic buffers
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

StereoSGBMImpl2::StereoSGBMImpl2()
	: params()
{
	// nothing
}

StereoSGBMImpl2::StereoSGBMImpl2(int _minDisparity, int _numDisparities, int _SADWindowSize,
	int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
	int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
	int _mode)
	: params(_minDisparity, _numDisparities, _SADWindowSize,
		_P1, _P2, _disp12MaxDiff, _preFilterCap,
		_uniquenessRatio, _speckleWindowSize, _speckleRange,
		_mode)
{
	// nothing
}

void StereoSGBMImpl2::computeMultiCam(std::vector<Mat> images, std::vector<cv::Point2i> disparityDirection, OutputArray disparr)
{
	CV_INSTRUMENT_REGION();
	disparr.create(images[0].size(), CV_16S);
	Mat disp = disparr.getMat();

	computeDisparitySGBM2(images, disparityDirection, disp, params);

	cv::medianBlur(disp, disp, 3);

	if (params.speckleWindowSize > 0)
		filterSpeckles(disp, (params.minDisparity - 1) * StereoMatcher::DISP_SCALE, params.speckleWindowSize,
			StereoMatcher::DISP_SCALE * params.speckleRange, buffer);
}

void StereoSGBMImpl2::compute(cv::InputArray leftarr, cv::InputArray rightarr, OutputArray disparr)
{
	CV_INSTRUMENT_REGION();

	//Mat left = leftarr.getMat();
	//Mat right = rightarr.getMat();
	//CV_Assert(left.size() == right.size() && left.type() == right.type() &&
	//	left.depth() == CV_8U);
	std::cout << "Before" << std::endl;
	//StereoSGBM::compute(leftarr, rightarr, disparr);
	std::cout << "Function out of use in this format. Use: " << std::endl << 
		"compute(std::vector<Mat> & images, std::vector<cv::Point2i> disparityDirection, OutputArray disparr)" << std::endl;
	//disparr.create(images[0].size(), CV_16S);
	//Mat disp = disparr.getMat();

	//computeDisparitySGBM(images, disparityDirection, disp, params);

	//cv::medianBlur(disp, disp, 3);

	//if (params.speckleWindowSize > 0)
	//	filterSpeckles(disp, (params.minDisparity - 1) * StereoMatcher::DISP_SCALE, params.speckleWindowSize,
	//		StereoMatcher::DISP_SCALE * params.speckleRange, buffer);
}

Rect getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
{
	int maxD = minDisparity + numberOfDisparities - 1;

	int xmin = std::max(roi1.x, roi2.x + maxD);
	int xmax = std::min(roi1.x + roi1.width, roi2.x + roi2.width);
	int ymin = std::max(roi1.y, roi2.y);
	int ymax = std::min(roi1.y + roi1.height, roi2.y + roi2.height) ;

	Rect r(xmin, ymin, xmax - xmin, ymax - ymin);

	return r.width > 0 && r.height > 0 ? r : Rect();
}

typedef cv::Point_<short> Point2s;

template <typename T>
void filterSpecklesImpl(cv::Mat & img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat & _buf)
{
	using namespace cv;

	int width = img.cols, height = img.rows, npixels = width * height;
	size_t bufSize = npixels * (int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
	if (!_buf.isContinuous() || _buf.empty() || _buf.cols * _buf.rows * _buf.elemSize() < bufSize)
		_buf.reserveBuffer(bufSize);

	uchar* buf = _buf.ptr();
	int i, j, dstep = (int)(img.step / sizeof(T));
	int* labels = (int*)buf;
	buf += npixels * sizeof(labels[0]);
	Point2s* wbuf = (Point2s*)buf;
	buf += npixels * sizeof(wbuf[0]);
	uchar* rtype = (uchar*)buf;
	int curlabel = 0;

	// clear out label assignments
	memset(labels, 0, npixels * sizeof(labels[0]));

	for (i = 0; i < height; i++)
	{
		T* ds = img.ptr<T>(i);
		int* ls = labels + width * i;

		for (j = 0; j < width; j++)
		{
			if (ds[j] != newVal)   // not a bad disparity
			{
				if (ls[j])     // has a label, check for bad label
				{
					if (rtype[ls[j]]) // small region, zero out disparity
						ds[j] = (T)newVal;
				}
				// no label, assign and propagate
				else
				{
					Point2s* ws = wbuf; // initialize wavefront
					Point2s p((short)j, (short)i);  // current pixel
					curlabel++; // next label
					int count = 0;  // current region size
					ls[j] = curlabel;

					// wavefront propagation
					while (ws >= wbuf) // wavefront not empty
					{
						count++;
						// put neighbors onto wavefront
						T* dpp = &img.at<T>(p.y, p.x);
						T dp = *dpp;
						int* lpp = labels + width * p.y + p.x;

						if (p.y < height - 1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff)
						{
							lpp[+width] = curlabel;
							*ws++ = Point2s(p.x, p.y + 1);
						}

						if (p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff)
						{
							lpp[-width] = curlabel;
							*ws++ = Point2s(p.x, p.y - 1);
						}

						if (p.x < width - 1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff)
						{
							lpp[+1] = curlabel;
							*ws++ = Point2s(p.x + 1, p.y);
						}

						if (p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff)
						{
							lpp[-1] = curlabel;
							*ws++ = Point2s(p.x - 1, p.y);
						}

						// pop most recent and propagate
						// NB: could try least recent, maybe better convergence
						p = *--ws;
					}

					// assign label type
					if (count <= maxSpeckleSize)   // speckle region
					{
						rtype[ls[j]] = 1;   // small region label
						ds[j] = (T)newVal;
					}
					else
						rtype[ls[j]] = 0;   // large region label
				}
			}
		}
	}
}


void cv::filterSpeckles(InputOutputArray _img, double _newval, int maxSpeckleSize,
	double _maxDiff, InputOutputArray __buf)
{
	CV_INSTRUMENT_REGION();

	Mat img = _img.getMat();
	int type = img.type();
	Mat temp, & _buf = __buf.needed() ? __buf.getMatRef() : temp;
	CV_Assert(type == CV_8UC1 || type == CV_16SC1);

	int newVal = cvRound(_newval), maxDiff = cvRound(_maxDiff);

	CV_IPP_RUN_FAST(ipp_filterSpeckles(img, maxSpeckleSize, newVal, maxDiff, _buf));

	if (type == CV_8UC1)
		filterSpecklesImpl<uchar>(img, newVal, maxSpeckleSize, maxDiff, _buf);
	else
		filterSpecklesImpl<short>(img, newVal, maxSpeckleSize, maxDiff, _buf);
}

void cv::validateDisparity(InputOutputArray _disp, InputArray _cost, int minDisparity,
	int numberOfDisparities, int disp12MaxDiff)
{
	CV_INSTRUMENT_REGION();

	Mat disp = _disp.getMat(), cost = _cost.getMat();
	int cols = disp.cols, rows = disp.rows;
	int minD = minDisparity, maxD = minDisparity + numberOfDisparities;
	int x, minX1 = std::max(maxD, 0), maxX1 = cols + std::min(minD, 0);
	AutoBuffer<int> _disp2buf(cols * 2);
	int* disp2buf = _disp2buf.data();
	int* disp2cost = disp2buf + cols;
	const int DISP_SHIFT = 4, DISP_SCALE = 1 << DISP_SHIFT;
	int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int costType = cost.type();

	disp12MaxDiff *= DISP_SCALE;

	CV_Assert(numberOfDisparities > 0 && disp.type() == CV_16S &&
		(costType == CV_16S || costType == CV_32S) &&
		disp.size() == cost.size());

	for (int y = 0; y < rows; y++)
	{
		short* dptr = disp.ptr<short>(y);

		for (x = 0; x < cols; x++)
		{
			disp2buf[x] = INVALID_DISP_SCALED;
			disp2cost[x] = INT_MAX;
		}

		if (costType == CV_16S)
		{
			const short* cptr = cost.ptr<short>(y);

			for (x = minX1; x < maxX1; x++)
			{
				int d = dptr[x], c = cptr[x];

				if (d == INVALID_DISP_SCALED)
					continue;

				int x2 = x - ((d + DISP_SCALE / 2) >> DISP_SHIFT);

				if (disp2cost[x2] > c)
				{
					disp2cost[x2] = c;
					disp2buf[x2] = d;
				}
			}
		}
		else
		{
			const int* cptr = cost.ptr<int>(y);

			for (x = minX1; x < maxX1; x++)
			{
				int d = dptr[x], c = cptr[x];

				if (d == INVALID_DISP_SCALED)
					continue;

				int x2 = x - ((d + DISP_SCALE / 2) >> DISP_SHIFT);

				if (disp2cost[x2] > c)
				{
					disp2cost[x2] = c;
					disp2buf[x2] = d;
				}
			}
		}

		for (x = minX1; x < maxX1; x++)
		{
			// we round the computed disparity both towards -inf and +inf and check
			// if either of the corresponding disparities in disp2 is consistent.
			// This is to give the computed disparity a chance to look valid if it is.
			int d = dptr[x];
			if (d == INVALID_DISP_SCALED)
				continue;
			int d0 = d >> DISP_SHIFT;
			int d1 = (d + DISP_SCALE - 1) >> DISP_SHIFT;
			int x0 = x - d0, x1 = x - d1;
			if ((0 <= x0 && x0 < cols && disp2buf[x0] > INVALID_DISP_SCALED && std::abs(disp2buf[x0] - d) > disp12MaxDiff) &&
				(0 <= x1 && x1 < cols && disp2buf[x1] > INVALID_DISP_SCALED && std::abs(disp2buf[x1] - d) > disp12MaxDiff))
				dptr[x] = (short)INVALID_DISP_SCALED;
		}
	}
}
