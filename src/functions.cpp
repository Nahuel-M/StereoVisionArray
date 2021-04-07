#include <fstream>

#include <iostream>
#include <filesystem>

#include "functions.h"
#include "Camera.h"
#include "dlibFaceSelect.h"

using namespace cv;

cv::Mat depth2Disparity(cv::Mat& depth, Camera camera1, Camera camera2)
{
	double camDistance = norm(camera1.pos3D - camera2.pos3D);
	double preMult = camDistance * camera1.f / camera1.pixelSize;
	Mat result;
	divide(preMult * 16, depth, result, 2);
	return result;
}

cv::Mat disparity2Depth(cv::Mat &disparity, Camera camera1, Camera camera2)
{
	Mat preMult;
	double camDistance = norm(camera1.pos3D - camera2.pos3D);
	multiply(disparity, camera1.pixelSize, preMult, 1, CV_64F);
	Mat depth = (camDistance * 16 * camera1.f / preMult);	//INEFFICIENT
	depth.setTo(0, depth < -100000 | depth > 100000);
	return depth;
}

//Mat getDisparityFromPair(std::vector<cv::Mat> &images, std::vector<Camera> &cameras, cv::Mat &mask, std::array<int, 2> pair)
//{
//	int kernelSize = 20;
//	Mat depth = Mat{ images[12].size(), CV_64FC1 };
//	Mat disparity = Mat{ images[12].size(), CV_8UC1 };
//	Size resolution = images[0].size();
//	Point2i halfRes = resolution / 2;
//	double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
//	waitKey(0);
//	for (int y = kernelSize; y < (resolution.height - kernelSize); y++) {
//		for (int x = kernelSize; x < resolution.width - kernelSize; x++) {
//			if (mask.at<uint8_t>(Point(x, y)) == 0) continue;
//			Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
//
//			Point3d vec = cameras[pair[0]].inv_project(Point2i{ x, y }-halfRes);
//			Point3d p1 = cameras[pair[0]].pos3D + (vec * 0.5);
//			Point3d p2 = cameras[pair[0]].pos3D + vec;
//			Point2i pixel1 = cameras[pair[1]].project(p1) + halfRes;
//			Point2i pixel2 = cameras[pair[1]].project(p2) + halfRes;
//			if (pixel1.x<kernelSize || pixel1.y < kernelSize || pixel1.x > resolution.width - kernelSize || pixel1.y > resolution.height - kernelSize) {
//				continue;
//			}
//			if (pixel2.x<kernelSize || pixel2.y < kernelSize || pixel2.x > resolution.width - kernelSize || pixel2.y > resolution.height - kernelSize) {
//				continue;
//			}
//			std::vector<Point2i> pixels = bresenham(pixel1, pixel2);
//			std::vector<double> error;
//			for (auto p : pixels) {
//				Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
//				Mat selection = images[pair[1]](selector);
//				Mat result{ CV_32FC1 };
//
//				error.push_back(getAbsDiff(selection, kernel));
//			}
//
//			int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
//
//			Point2i pixel = pixels[maxIndex];
//			//std::cout << norm(pixel - Point2i{ x, y }) << std::endl;
//			disparity.at<unsigned char>(Point(x, y)) = (int)norm(pixel - Point2i{ x, y });
//		}
//	}
//	//imshow("Disp", disparity);
//	return disparity;
//}

//Mat getDisparityFromPair2(std::vector<cv::Mat>& images, std::vector<Camera>& cameras, cv::Mat& mask, std::array<int, 2> pair)
//{
//	int kernelSize = 20;
//	Mat depth = Mat{ images[12].size(), CV_64FC1 };
//	Mat disparity = Mat{ images[12].size(), CV_8UC1 };
//	Size resolution = images[0].size();
//	Point2i halfRes = resolution / 2;
//	double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
//
//	double preMultX = (cameras[pair[0]].pos3D.x - cameras[pair[1]].pos3D.x) / camDistance;
//	double preMultY = (cameras[pair[0]].pos3D.y - cameras[pair[1]].pos3D.y) / camDistance;
//	int size[3] = { resolution.width, resolution.height, 155 };
//	cv::Mat errorStorage = cv::Mat(3, size, CV_16U, cv::Scalar(0));
//
//	for (int y = kernelSize; y < (resolution.height - kernelSize); y++) {
//		for (int x = kernelSize; x < resolution.width - kernelSize; x++) {
//			if (mask.at<uint8_t>(Point(x, y)) == 0) continue;
//			Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
//
//			Point2i pixel1 = { x + int(100 * preMultX), y + int(100 * preMultY) };
//			Point2i pixel2 = { x + int(255 * preMultX), y + int(255 * preMultY) };
//
//			if (pixel1.x<kernelSize || pixel1.y < kernelSize || pixel1.x > resolution.width - kernelSize || pixel1.y > resolution.height - kernelSize) {
//				continue;
//			}
//			if (pixel2.x<kernelSize || pixel2.y < kernelSize || pixel2.x > resolution.width - kernelSize || pixel2.y > resolution.height - kernelSize) {
//				continue;
//			}
//
//			std::vector<Point2i> pixels = bresenham(pixel1, pixel2);
//
//			std::vector<int> error;
//			for (int p = 0; p < pixels.size(); p++) {
//				Rect selector = Rect{ pixels[p] - Point(kernelSize, kernelSize), pixels[p] + Point(kernelSize, kernelSize) };
//				Mat selection = images[pair[1]](selector);
//				int err = getAbsDiff(selection, kernel);
//				error.push_back(err);
//				errorStorage.at<unsigned short>(x, y, p) = err;
//			}
//
//			int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
//
//			Point2i pixel = pixels[maxIndex];
//			//std::cout << norm(pixel - Point2i{ x, y }) << std::endl;
//			disparity.at<unsigned char>(Point(x, y)) = (int)norm(pixel - Point2i{ x, y });
//		}
//	}
//
//	//imshow("Disp", disparity);
//	Mat globDisparity{ disparity.size(), disparity.type() };
//
//	for (int y = 1; y < (resolution.height)-1; y++) {
//		std::cout << " Y: " << y << std::endl;
//		for (int x = 1; x < resolution.width-1; x++) {
//			Range range[] = { Range(y,y+1), Range(x,x+1), Range::all() };
//			Mat data = errorStorage(range);
//			unsigned short disp =
//				disparity.at<unsigned char>(y - 1, x) +
//				disparity.at<unsigned char>(y + 1, x) +
//				disparity.at<unsigned char>(y, x - 1) +
//				disparity.at<unsigned char>(y, x + 1);
//			disp = disp / 4;
//			for (auto e = 0; e < data.size[0]; e++) {
//				data.at<unsigned char>(e) = data.at<unsigned char>(e) + (abs(e + 100 - disp) > 0) * 100 + (abs(e + 100 - disp) > 1) * 500;
//			}
//			double min;
//			Point minLoc;            
//			try
//			{
//				minMaxLoc(data, &min, nullptr, &minLoc, nullptr);
//			}
//			catch (cv::Exception & e)
//			{
//				std::cout << e.what() << std::endl;
//			}
//			try 
//			{
//				std::cout << y << std::endl;
//				globDisparity.at<unsigned char>(y, x) = minLoc.x + 100;
//			}
//			catch(cv::Exception & e)
//			{
//				std::cout << e.what() << std::endl;
//			}
//		}
//	}
//	showImage("Disp", disparity);
//	showImage("Glob", globDisparity);
//	return globDisparity;
//}

//cv::Mat getDisparityFromPairSGM(cv::Mat& image1, cv::Mat& image2, cv::Mat& mask, Camera cam1, Camera cam2, int P1, int P2)
//{
//	Point3d distance = cam1.pos3D - cam2.pos3D;
//	Ptr<StereoSGBM> sgbm;
//	Mat* im1pointer = &image1;
//	Mat* im2pointer = &image2;
//	Mat im1rot, im2rot;
//
//	int disp12MaxDiff = 12;
//	int preFilterCap = 4;
//	int uniquenessRatio = 1;
//	int speckleWindowSize = 100;
//	int speckleRange = 5;
//
//
//	if (distance.x > 0) {
//		sgbm = StereoSGBM::create(-240, 144, 1, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_HH);
//	}
//	else if (distance.x < 0) {
//		sgbm = StereoSGBM::create(96, 144, 1, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_HH);
//	}
//	else if (distance.y > 0) {
//		sgbm = StereoSGBM::create(-240, 144, 1, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_HH);
//		rotate(image1, im1rot, ROTATE_90_COUNTERCLOCKWISE);
//		rotate(image2, im2rot, ROTATE_90_COUNTERCLOCKWISE);
//		im1pointer = &im1rot;
//		im2pointer = &im2rot;
//	}
//	else if (distance.y < 0) {
//		sgbm = StereoSGBM::create(96, 144, 1, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_HH);
//		rotate(image1, im1rot, ROTATE_90_COUNTERCLOCKWISE);
//		rotate(image2, im2rot, ROTATE_90_COUNTERCLOCKWISE);
//		im1pointer = &im1rot;
//		im2pointer = &im2rot;
//	}
//	cv::Mat disparity;
//	sgbm->setMode(StereoSGBM::MODE_SGBM);
//	sgbm->compute(*im1pointer, *im2pointer, disparity);
//	if (distance.y != 0) {
//		rotate(disparity, disparity, ROTATE_90_CLOCKWISE);
//	}
//	disparity = abs(disparity);
//	std::cout << disparity.type() << std::endl;
//	disparity.convertTo(disparity, 2);
//	disparity.setTo(0, mask == 0);
//
//	return disparity;
//}

cv::Mat getDisparityFromPairSGM(std::array<int,2> pair, int P1, int P2)
{

	Point3d distance = cameras[pair[0]].pos3D - cameras[pair[1]].pos3D;
	Ptr<StereoSGBM> sgbm;
	Mat* im1pointer = &images[pair[0]];
	Mat* im2pointer = &images[pair[1]];
	Mat im1rot, im2rot;

	int disp12MaxDiff = 12;
	int preFilterCap = 4;//PreFilterCap			4
	int uniqRatio = 2;	// Uniqueness ratio		
	int sWinSize = 200;	// Speckle window size	100
	int sRange = 10;	// Speckle range
	int numDisp = 48;	// Number of disparities
	int minDisp = 250;	// Minimum disparity
	

	if (distance.x > 0) {
		sgbm = StereoSGBM::create(-minDisp - numDisp, numDisp, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
	}
	else if (distance.x < 0) {
		sgbm = StereoSGBM::create(minDisp, numDisp, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
	}
	else if (distance.y < 0) {
		sgbm = StereoSGBM::create(-minDisp - numDisp, numDisp, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
		rotate(images[pair[0]], im1rot, ROTATE_90_COUNTERCLOCKWISE);
		rotate(images[pair[1]], im2rot, ROTATE_90_COUNTERCLOCKWISE);
		im1pointer = &im1rot;
		im2pointer = &im2rot;
	}
	else if (distance.y > 0) {
		sgbm = StereoSGBM::create(minDisp, numDisp, 1, P1, P2, disp12MaxDiff, preFilterCap, uniqRatio, sWinSize, sRange, 1);
		rotate(images[pair[0]], im1rot, ROTATE_90_COUNTERCLOCKWISE);
		rotate(images[pair[1]], im2rot, ROTATE_90_COUNTERCLOCKWISE);
		im1pointer = &im1rot;
		im2pointer = &im2rot;
	}
	cv::Mat disparity;
	//sgbm->setMode(StereoSGBM::MODE_SGBM);
	sgbm->compute(*im1pointer, *im2pointer, disparity);
	if (distance.y != 0) {
		rotate(disparity, disparity, ROTATE_90_CLOCKWISE);
	}
	disparity = abs(disparity);
	disparity.convertTo(disparity, 2);
	disparity.setTo(0, mask == 0);

	return disparity;
}

void getCameras(std::vector<Camera>& cameras, cv::Size resolution, double f, double sensorSize, double pixelSize)
{
	std::cout << "Image Resolution: " << resolution << std::endl;
	if (pixelSize == 0) {
		pixelSize = sensorSize / resolution.width;
	}
	std::cout << "Pixel Size: " << pixelSize << std::endl;


	for (int y = 0; y < 5; y++)
	{
		for (int x = 0; x < 5; x++)
		{
			cameras.push_back(Camera(f, Point3d{ 0.1 - x * 0.05, 0.1 - y * 0.05, -0.75 }, pixelSize));	//VARIABLE
		}
	}
}

void getImages(std::vector<cv::Mat>& images, std::string folderName, double scale)
{
	std::vector<std::string> files = getImagesPathsFromFolder(folderName);
	for (int i = 0; i < files.size(); i++) {
		images.push_back(imread(files[i], IMREAD_GRAYSCALE));
		resize(images.back(), images.back(), Size(), scale, scale);
	}
}

cv::Mat improveWithDisparity(cv::Mat& disparity, cv::Mat centerImage, std::vector<cv::Mat> &images, std::vector<std::array<Camera, 2>> &cameras, int windowSize)
{
	Mat mask = getFaceMask(centerImage);
	Mat improvedDisparity{ disparity.size(), disparity.type() };
	int kernelSize = (windowSize - 1) / 2;
	for (int c = 0; c < cameras.size(); c++) {
		std::array<Camera,2> cam = cameras[c];

		Mat shifted = shiftPerspectiveWithDisparity(cam[0], cam[1], disparity, images[c]);
		Point2d distance = Point2d{ cam[0].pos3D.x - cam[1].pos3D.x, cam[0].pos3D.y - cam[1].pos3D.y };
		distance.x = distance.x / norm(distance.x) && (distance.x>0.001);
		distance.y = distance.y / norm(distance.y) && (distance.y > 0.001);
		std::cout << distance << std::endl;
		for (int y = 0; y < centerImage.rows; y++) {
			for (int x = 0; x < centerImage.cols; x++) {
				if (mask.at<uint8_t>(Point(x, y)) == 0) continue; 
				Mat window = centerImage(Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
				std::vector<double> error;
				for (int p = 0; p <= 10; p++) {
					Point2i newP = Point2i{ x, y } + (Point2i) distance * (p - 5);
					Mat compWindow = shifted(Rect{ newP - Point2i{kernelSize, kernelSize}, newP + Point2i{kernelSize, kernelSize} });
					error.push_back(getAbsDiff(compWindow, window));
				}
				int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
				improvedDisparity.at<unsigned char>(y, x) = disparity.at<unsigned char>(y, x) + (maxIndex - 5)*(distance.x+distance.y);
			}
		}

	}
	return improvedDisparity;
}

cv::Mat iterDisparityImproveSGM(cv::Mat& disparity, Mat& mask, cv::Mat& centerIm, cv::Mat& offCenterIm, Camera centerCam, Camera offCenterCam)
{
	Mat improvedDisparity{ disparity.size(), disparity.type() };
	Mat center;
	centerIm.copyTo(center, mask);

	Mat shifted = shiftPerspectiveWithDisparity(centerCam, offCenterCam, disparity, offCenterIm);
	showImage("shifted", shifted);
	Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
		-16, 32, 1,
		2, 16, 12,
		4, 1,
		10, 2,
		cv::StereoSGBM::MODE_HH
	);

	Mat addDisp;
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
	sgbm->compute(center, shifted, addDisp);

	add(disparity, addDisp, improvedDisparity, mask, disparity.type());
	std::cout << disparity.type() << addDisp.type() << improvedDisparity.type() << std::endl;
	std::cout << "After " << addDisp.at<signed short>(718, 613) << std::endl;

	return improvedDisparity;
}

cv::Mat shiftPerspectiveWithDisparity(Camera& inputCam, Camera& outputCam, cv::Mat& disparity, cv::Mat& image)
{
	Mat shiftedImage = Mat{ image.size() , image.type() };
	double camDistance = norm(inputCam.pos3D - outputCam.pos3D);

	double preMultX = (inputCam.pos3D.x - outputCam.pos3D.x) / norm(inputCam.pos3D - outputCam.pos3D) / 16;
	double preMultY = (inputCam.pos3D.y - outputCam.pos3D.y) / norm(inputCam.pos3D - outputCam.pos3D) / 16;
	for (int y = 0; y < shiftedImage.rows; y++) {
		for (int x = 0; x < shiftedImage.cols; x++) {
			double disp = disparity.at<ushort>(y, x);
			if (disp == 0) {
				continue;
			}
			int shiftedX = disp * preMultX + x;
			int shiftedY = disp * preMultY + y;
			if (shiftedY >= disparity.rows || shiftedY < 0 || shiftedX >= disparity.cols || shiftedX < 0)
				continue;
			shiftedImage.at<unsigned char>(y, x) = image.at<unsigned char>(shiftedY, shiftedX);
		}
	}
	return shiftedImage;
}

cv::Mat shiftPerspective(Camera inputCam, Camera outputCam, cv::Mat &depth)
{
	Mat shiftedDepthMap = Mat{ depth.size() , depth.type() };
	double preMultX = (inputCam.pos3D.x - outputCam.pos3D.x) * inputCam.f / inputCam.pixelSize;
	double preMultY = (inputCam.pos3D.y - outputCam.pos3D.y) * inputCam.f / inputCam.pixelSize;
	int direction[] = { outputCam.pos3D.x - inputCam.pos3D.x, outputCam.pos3D.y - inputCam.pos3D.y };
	//std::cout << "direction0 " << direction[0] << std::endl;
	for (int x = 0; x < depth.cols; x++) {
		for (int y = 0; y < depth.rows; y++) {
			double d = depth.at<ushort>(y,x);
			if (d < 0.5) 
				continue;
			int shiftedX = int(preMultX / d) + x;
			int shiftedY = int(preMultY / d) + y;
			if (shiftedY >= depth.rows || shiftedY < 0 || shiftedX >= depth.cols || shiftedX < 0) 
				continue;
			shiftedDepthMap.at<double>(Point(shiftedX, shiftedY)) = d;
		}
	}

	//cv::imshow("Map", shiftedDepthMap);
	//cv::imshow("Original", depthMap);
	//cv::waitKey(0);
	return shiftedDepthMap;
}

cv::Mat shiftDisparityPerspective(Camera inputCam, Camera outputCam, cv::Mat& disparity)
{

	Mat shiftedDepthMap = Mat{ disparity.size() , disparity.type() };
	/// Hardcoded cam distance difference of 0.05m. Disparity counts with 1/16 pixel VARIABLE
	double camXDiff = (inputCam.pos3D.x - outputCam.pos3D.x) / (0.05 * 16);
	double camYDiff = (inputCam.pos3D.y - outputCam.pos3D.y) / (0.05 * 16);

	for (int x = 0; x < disparity.cols; x++) {
		for (int y = 0; y < disparity.rows; y++) {
			double d = disparity.at<ushort>(y, x);
			if (d == 0) continue;
			int shiftedX = camXDiff * d + x;
			int shiftedY = camYDiff * d + y;
			//std::cout << shiftedX << ", " << shiftedY << std::endl;
			if (shiftedY >= disparity.rows || shiftedY < 0 || shiftedX >= disparity.cols || shiftedX < 0)
				continue;
			shiftedDepthMap.at<ushort>(Point(shiftedX, shiftedY)) = d;
		}
	}

	return shiftedDepthMap;
}

void fillHoles(cv::Mat& disparity, int filterSize) 
{
	Mat binaryMask = (disparity != 0);
	Mat blurWeights{ disparity.size(), CV_16U };
	GaussianBlur(binaryMask, blurWeights, Size(filterSize, filterSize), 0, 0);

	Mat blurredDisparity;
	GaussianBlur(disparity, blurredDisparity, Size(filterSize, filterSize), 0, 0);

	Mat weightedBlur;
	divide(blurredDisparity, blurWeights, weightedBlur, 256, CV_16U);

	weightedBlur.copyTo(disparity, disparity==0);
}

std::vector< std::vector<std::array<int, 2>> > getGroups(std::vector<Camera> &cameras, std::string groupType)
{
	std::vector< std::vector<std::array<int, 2>> > groups;
	if (groupType == "CHESS") {
		for (int i = 0; i < 25; i += 2) {
			groups.push_back(getCameraPairs(cameras, CROSS, i));
		}
	}
	return groups;
}

cv::Mat Points3DToDepthMap(std::vector<Point3d>& points, Camera camera, cv::Size resolution)
{
	Mat depthMap = Mat{ resolution, CV_64FC1 };
	std::cout << depthMap.size() << std::endl;
	Point2i halfRes = resolution / 2;
	for (auto p : points) 
	{
		Point2i pixel = camera.project(p) + halfRes;
		if (pixel.x >= 0 && pixel.x < resolution.width && pixel.y >= 0 && pixel.y < resolution.height) {
			//std::cout << p << ", " << pixel << std::endl;
			depthMap.at<double>(pixel) = p.z - camera.pos3D.z;
		}
	}
	return depthMap;
}

std::vector<Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, cv::Size resolution)
{
	Point2i halfRes = resolution / 2;
	std::vector<Point3d> Points;
	for (int u = 0; u < depthMap.cols; u++) {
		for (int v = 0; v < depthMap.rows; v++) {
			double depth = depthMap.at<double>(Point(u, v));
			if (depth > 0.1)
				Points.push_back( camera.pos3D + camera.inv_project(Point(u,v)-halfRes) * depth );
		}
	}
	return Points;
}

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair) {
	std::vector<std::array<int, 2>> pairs;
	if (pair == TO_CENTER) {
		for (int i = 0; i < cameras.size(); i++) {
			if (i == 12) continue;
			pairs.push_back({ 12, i });
		}
	}
	else if (pair == TO_CENTER_SMALL) {
			pairs.push_back({ 12, 6 });
			pairs.push_back({ 12, 7 });
			pairs.push_back({ 12, 8 });
			pairs.push_back({ 12, 11 });
			pairs.push_back({ 12, 13 });
			pairs.push_back({ 12, 16 });
			pairs.push_back({ 12, 17 });
			pairs.push_back({ 12, 18 });
	}
	else if (pair == MID_LEFT) {
		pairs.push_back({ 12, 11 });
	}
	else if (pair == MID_RIGHT) {
		pairs.push_back({ 12, 13 });
	}
	else if (pair == LEFT_LEFTER) {
		pairs.push_back({ 11, 10 });
	}
	else if (pair == RIGHT_RIGHTER) {
		pairs.push_back({ 14, 13 });
	}
	else if (pair == MID_TOP) {
		pairs.push_back({ 12, 7 });
	}
	else if (pair == MID_BOTTOM) {
		pairs.push_back({ 12, 17 });
	}
	else if (pair == LINE_HORIZONTAL) {
		for (int i = 10; i < 15; i++) {
			if (i == 12) continue;
			pairs.push_back({ 12, i });
		}
	}
	else if (pair == LINE_VERTICAL) {
		for (int i = 2; i < 25; i+=5) {
			if (i == 12) continue;
			pairs.push_back({ 12, i });
		}
	}
	else if (pair == CROSS) {
			pairs.push_back({ 12, 11 });
			pairs.push_back({ 12, 13 });
			pairs.push_back({ 12, 7 });
			pairs.push_back({ 12, 17 });
	}
	else if (pair == JUMP_CROSS) {
		pairs.push_back({ 12, 10 });
		pairs.push_back({ 12, 14 });
		pairs.push_back({ 12, 2 });
		pairs.push_back({ 12, 24 });
	}
	return pairs;
}

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair, const int cameraNum) {
	std::vector<std::array<int, 2>> pairs;
	if (pair == CROSS) {
		if(cameraNum-5>0)
			pairs.push_back({ cameraNum, cameraNum - 5 });
		if(cameraNum+5<25)
			pairs.push_back({ cameraNum, cameraNum + 5 });
		if(cameraNum%5>0)
			pairs.push_back({ cameraNum, cameraNum - 1 });
		if (cameraNum % 5 < 4)
			pairs.push_back({ cameraNum, cameraNum + 1 });
		
	}
	return pairs;
}

int getAbsDiff(cv::Mat& mat1, cv::Mat& mat2)
{
	return sum(abs(mat1-mat2))[0];
}

// natural compare by Christian Ammer
bool compareNat(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return compareNat(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (compareNat(anew, bnew));
}

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath)
{
	namespace fs = std::filesystem;
	std::vector<std::string> filePaths;
	for (auto& p : fs::directory_iterator(folderPath))
	{
		filePaths.push_back(p.path().u8string());
		//std::cout << p.path().u8string() << std::endl;
	}
	std::sort(filePaths.begin(), filePaths.end(), compareNat);
	return filePaths;
}

double calculateAverageError(cv::Mat &image)
{
	std::string folder = "Images";
	std::vector<std::string> files = getImagesPathsFromFolder(folder);
	Mat mask = getFaceMask(image);
	return cv::mean(image, mask)[0];
}

cv::Mat depth2Normals(cv::Mat& depth, Mat& mask, Camera cam)
{
	Mat averagedDepth;
	averagedDepth = blurWithMask(depth, mask, 79);
	//showImage("Depth", depth);
	//showImage("Mask", mask);

	double orthogonalDiff = cam.pixelSize / cam.f * 2;

	cv::Mat normals(depth.size(), CV_32FC3);

	for (int x = 3; x < depth.rows-3; ++x)
	{
		for (int y = 3; y < depth.cols-3; ++y)
		{
			if (mask.at<uchar>(x, y) == 0) continue;
			float dzdy = ((float)averagedDepth.at<double>(x + 1, y) - (float)averagedDepth.at<double>(x - 1, y));
			float dzdx = ((float)averagedDepth.at<double>(x, y + 1) - (float)averagedDepth.at<double>(x, y - 1));

			Vec3f d(-dzdx, -dzdy, orthogonalDiff * averagedDepth.at<double>(x, y));
			Vec3f n = normalize(d);

			normals.at<Vec3f>(x, y) = n;
		}
	}

	//showImage("normals", abs(normals));
	return normals;
}

// UNOPERATIONAL
cv::Mat disparity2Normals(cv::Mat& disparity, Mat& mask, Camera cam)
{
	Mat avgDisparity;
	avgDisparity = blurWithMask(disparity, mask, 49);
	float camDist = 0.05;	/// VARIABLE
	double preMult = 16 * camDist * cam.f / cam.pixelSize;

	cv::Mat normals(disparity.size(), CV_32FC3);

	for (int x = 3; x < disparity.rows - 3; ++x)
	{
		for (int y = 3; y < disparity.cols - 3; ++y)
		{
			if (mask.at<uchar>(x, y) == 0) continue;
			float dzdx = (preMult / avgDisparity.at<ushort>(x + 1, y) - preMult / avgDisparity.at<ushort>(x - 1, y));
			float dzdy = (preMult / avgDisparity.at<ushort>(x, y + 1) - preMult / avgDisparity.at<ushort>(x, y - 1));

			Vec3f d(-dzdx, -dzdy, 2 * camDist * avgDisparity.at<ushort>(x, y));
			Vec3f n = normalize(d);

			normals.at<Vec3f>(x, y) = n;
		}
	}

	//showImage("normals", abs(normals));
	return normals;
}

cv::Mat getOrthogonalityFromCamera(cv::Mat& disparity, cv::Mat& mask, Camera perspective, Camera stereo, Camera orthogonality)
{
	Mat depth = disparity2Depth(disparity, perspective, stereo);
	showImage("depth", depth*4-2);
	Mat normals = depth2Normals(depth, mask, perspective);
	showImage("normals", normals);
	Vec3f camDiff3D = (Vec3f)(Vec3d)perspective.pos3D - (Vec3f)(Vec3d)orthogonality.pos3D; //INEFFICIENT
	Mat camAngle{disparity.size(), CV_32FC3};
	double preMult = perspective.pixelSize / perspective.f;

	for (int u = 0; u < disparity.cols; u++) {
		for (int v = 0; v < disparity.rows; v++) {
			if (disparity.at<ushort>(v, u) == 0) continue;
			double d = depth.at<double>(v, u);
			Vec3f pos3D(
				(u - (disparity.cols / 2)) * d * preMult,
				(v - (disparity.rows / 2)) * d * preMult,
				d
			);

			pos3D = pos3D + camDiff3D;
			camAngle.at<Vec3f>(v, u) = normalize(pos3D);
		}
	}
	//showImage("camAngle", abs(camAngle));
	//showImage("Normals", normals);

	Mat normalOrthogonality = matrixDot(camAngle, normals);
	//showImage("normalOrthogonality", normalOrthogonality);
	return normalOrthogonality;
}

cv::Mat getPixelNormals(Camera& camera, Mat& image)
{
	cv::Mat normals(image.size(), CV_32FC3);
	for (int u = 0; u < image.cols; u++) {
		for (int v = 0; v < image.rows; v++) {
			Vec3f pixelNormal(
				(u - image.cols / 2) *	camera.pixelSize, 
				(v - image.rows / 2) *	camera.pixelSize, 
				camera.f
			);
			normals.at<Vec3f>(v, u) = normalize(pixelNormal);
		}
	}

	return normals;
}

cv::Mat matrixDot(cv::Mat& mat1, cv::Mat& mat2)
{
	Mat dotted{ Size{mat1.cols, mat1.rows}, CV_32F };
	if (mat1.size() != mat2.size()) {
		std::cout << "Mat sizes don't match for matrixDot function" << std::endl;
		return Mat{};
	}

	for (int u = 0; u < mat1.cols; u++) {
		for (int v = 0; v < mat1.rows; v++) {
			//std::cout << mat1.at<Vec3f>(v, u).dot(mat2.at<Vec3f>(v, u)) << std::endl;
			dotted.at<float>(v, u) = mat1.at<Vec3f>(v, u).dot(mat2.at<Vec3f>(v, u));
		}
	}
	//showImage("Dot", dotted);
	return dotted;
}

cv::Mat blurWithMask(cv::Mat& image, cv::Mat& mask, int filterSize)
{
	Mat blurMask;
	Mat floatMask;
	mask.convertTo(floatMask, CV_32F);
	GaussianBlur(floatMask, blurMask, Size(filterSize, filterSize), 0, 0);

	Mat blurredImage;
	GaussianBlur(image, blurredImage, Size(filterSize, filterSize), 0, 0);

	Mat weightedBlur;
	divide(blurredImage, blurMask, weightedBlur, 256, image.type());

	Mat maskedBlur;
	weightedBlur.copyTo(maskedBlur, mask);

	return maskedBlur;
}

cv::Mat getBlurredSlope(cv::Mat image, int vertical, int blurKernelSize)
{
	char kdata[] = { -1, 0, 1 };
	Mat kernel;
	if (vertical == 0) 
	{
		kernel = Mat(1, 3, CV_8S, kdata);
	}
	else
	{
		kernel = Mat(3, 1, CV_8S, kdata);
	}
	Mat delta;
	filter2D(image, delta, CV_32F, kernel);
	delta = abs(delta);
	//blur(delta, delta, Size(50, 50));
	GaussianBlur(delta, delta, Size(blurKernelSize, blurKernelSize), 0);
	return delta;
}

void getCameraIntrinsicParameters(std::string filePath, cv::Mat& K, cv::Mat& D)
{
	/// Read intrinsic camera calibration info from file
	FileStorage fs(filePath, FileStorage::READ);
	fs["K"] >> K;
	fs["D"] >> D;
	fs.release();
}

void undistortImages(std::vector<cv::Mat>& images, cv::Mat& K, cv::Mat& D, bool verbose)
{
	Mat map1, map2;
	cv::initUndistortRectifyMap(K, D, Mat(), K, images[0].size(), images[0].type(), map1, map2);
	for (int i = 0; i < images.size(); i++) {
		if(verbose){ showImage("Before", images[i]); }
		remap(images[i], images[i], map1, map2, INTER_LINEAR);
		if (verbose) { showImage("After", images[i]); }
	}

}

void exportOBJfromDisparity(cv::Mat disparityImage, std::string fileName, Camera cam1, Camera cam2, float scale) 
{
	float preMult = norm(cam1.pos3D-cam2.pos3D) * cam1.f * 16 / cam1.pixelSize;
	std::cout << "Premult: " << norm(cam1.pos3D - cam2.pos3D) << " * " << cam1.f << " = " << preMult << std::endl;
	if (scale != 1.f) {
		resize(disparityImage, disparityImage, Size{}, scale, scale);
	}
	Mat_<float> Z;
	divide(preMult, disparityImage, Z, CV_32F);
	std::ofstream outputFile(fileName);
	for (int u = 0; u < Z.rows; u++)
	{
		for (int v = 0; v < Z.cols; v++)
		{
			//std::cout << Z(u, v) << ", " << disparityImage.at<short>(u, v) << std::endl;
			outputFile << "v " << Z(u, v) * cam1.pixelSize * (v-Z.cols/2) / cam1.f << " " << Z(u, v) * cam1.pixelSize * (u-Z.rows/2) / cam1.f << " " << Z(u, v) << std::endl;
		}
	}
	outputFile.close();
}