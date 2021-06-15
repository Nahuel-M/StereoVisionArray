#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#pragma warning (pop)

#include "functions.h"
#include "Camera.h"
#include "dlibFaceSelect.h"
#include "bresenham.h"

using namespace cv;

cv::Mat depth2Disparity(cv::Mat& depth, Camera camera1, Camera camera2)
{
	double camDistance = 0.05; //VARIABLE
	std::cout << camera1.f << std::endl;
	double preMult = camDistance * camera1.f / camera1.pixelSize;
	Mat result;
	divide(preMult * 16, depth, result, CV_16S);
	return result;
}

cv::Mat disparity2Depth(cv::Mat &disparity, Camera camera1, Camera camera2)
{
	Mat preMult;
	//double camDistance = 0.05; //VARIABLE
	double camDistance = norm(camera1.pos3D - camera2.pos3D);
	cv::multiply(disparity, camera1.pixelSize, preMult, 1, CV_32F);
	Mat depth = (camDistance * 16 * camera1.f / preMult);	//INEFFICIENT
	depth.setTo(0, depth < -100000 | depth > 100000);
	return depth;
}

void getCameras(std::vector<Camera>& cameras, std::string positionFilePath, double f, double sensorSize, double pixelSize)
{
	if (images.size() < 1) {
		throw("Load images first to get image size");
	}
	cameras.clear();
	Size resolution = images[0].size();
	Point2i principalPoint; double fx, fy;
	std::vector<Point3d> camPositions;

	FileStorage camPosFile(positionFilePath + "\\cameraPosition.xyz", FileStorage::READ);
	if (positionFilePath != "" && camPosFile.isOpened()) {
			
			camPosFile["camera_positions"] >> camPositions;		// Load camera positions from file
			if (!camPosFile["camera_pixel_size"].empty()) {
				camPosFile["camera_pixel_size"] >> pixelSize; pixelSize = pixelSize / 1000 / scale;
			} else if (!camPosFile["camera_sensor_width"].empty()) {
				camPosFile["camera_sensor_width"] >> sensorSize;// Load sensor width from file
				pixelSize = sensorSize / resolution.width;		// Calculate pixel size from sensor width
			} else
				throw("No sensor width or pixel size provided. Camera information cannot be calculated");
			camPosFile.release();
	}
	else
	{
		std::cout << "Camera positions file could not be found. Resorting to theoretical positions." << std::endl;
		for (int y = 0; y < 5; y++)
			for (int x = 0; x < 5; x++)
				camPositions.push_back(Point3d{ 0.1 - x * 0.05, 0.1 - y * 0.05, -0.84 });
	}

	if (!K.empty())
	{
		f = (K.at<double>(0, 0) + K.at<double>(1, 1)) / 2 * pixelSize;
		fx = K.at<double>(0, 0) * pixelSize;
		fy = K.at<double>(1, 1) * pixelSize;
		principalPoint = Point2i{ (int)K.at<double>(0,2), (int)K.at<double>(1,2) };
	}
	else
	{
		fx = f;
		fy = f;
		principalPoint = Point2i{ resolution.width/2, resolution.height/2 };
	}
	if (pixelSize == 0) 
		pixelSize = sensorSize / resolution.width;

	for (Point3d& cam : camPositions)
			cameras.push_back(Camera(f, fx, fy, cam, pixelSize, principalPoint));
	std::cout << "Image Resolution: " << resolution << std::endl;
	std::cout << "Pixel Size: " << pixelSize << std::endl;
	std::cout << "Focal length: " << f << std::endl;
}

void getCameras(std::vector<Camera>& cameras, float baseline)
{
	if (images.size() < 1) {
		throw("Load images first to get image size");
	}
	cameras.clear();
	Size resolution = images[0].size();
	double f = 0.05, sensorSize = 0.036;
	double pixelSize = sensorSize / resolution.width;
	std::vector<Point3d> camPositions;

	float b = baseline;
	for (int y = 0; y < 5; y++)
		for (int x = 0; x < 5; x++)
			camPositions.push_back(Point3d{ 2*b - x * b, 2*b - y * b, -0.84 });

	double fx = f;
	double fy = f;
	Point2i principalPoint = Point2i{ resolution.width / 2, resolution.height / 2 };

	for (Point3d& cam : camPositions)
		cameras.push_back(Camera(f, fx, fy, cam, pixelSize, principalPoint));
	std::cout << "Image Resolution: " << resolution << std::endl;
	std::cout << "Pixel Size: " << pixelSize << std::endl;
	std::cout << "Focal length: " << f << std::endl;
}

void getImages(std::vector<cv::Mat>& images, std::string folderName, double scale)
{
	images.clear();
	std::vector<std::string> files = getImagesPathsFromFolder(folderName);
	for (int i = 0; i < files.size(); i++) {
		images.push_back(imread(files[i], IMREAD_GRAYSCALE | IMREAD_ANYDEPTH));
		//cv::Vec3f screenBall = getBallScreenParams(images.back(), 478, 485);
		//circle(images.back(), Point2i(screenBall[0], screenBall[1]), screenBall[2]+2000, Scalar(0, 0, 0), 3950, 8, 0);
		//showImage("back", images.back());
		if(scale!=1)
			resize(images.back(), images.back(), Size(), scale, scale);
	}
}

cv::Mat Points3DToDepthMap(std::vector<Point3d>& points, Camera camera, cv::Size resolution)
{
	Mat depthMap = Mat{ resolution, CV_64FC1 };
	std::cout << depthMap.size() << std::endl;
	Point2i halfRes = resolution / 2;
	for (auto &p : points) 
	{
		Point2i pixel = camera.project(p) + halfRes;
		if (pixel.x >= 0 && pixel.x < resolution.width && pixel.y >= 0 && pixel.y < resolution.height) {
			//std::cout << p << ", " << pixel << std::endl;
			depthMap.at<double>(pixel) = p.z - camera.pos3D.z;
		}
	}
	return depthMap;
}

std::vector<Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, Point2i principalPoint, Mat mask)
{
	std::vector<Point3d> Points;
	for (int v = 0; v < depthMap.rows; v++) {
		for (int u = 0; u < depthMap.cols; u++) {
			double depth = depthMap.at<double>(v,u);
			if (mask.at<uchar>(v,u)>0)
				Points.push_back( camera.pos3D + camera.inv_project(Point(u,v)- principalPoint) * depth );
		}
	}
	return Points;
}

std::vector<int> getCrossIDs(int centerCameraID) {
	std::vector<int> cams;
		if (centerCameraID - 5>0)
			cams.push_back(centerCameraID - 5 );
		if (centerCameraID + 5<25)
			cams.push_back(centerCameraID + 5 );
		cams.push_back(centerCameraID);
		if (centerCameraID % 5>0)
			cams.push_back(centerCameraID - 1 );
		if (centerCameraID % 5 < 4)
			cams.push_back(centerCameraID + 1 );

	return cams;
}

int getAbsDiff(cv::Mat& mat1, cv::Mat& mat2)
{
	return (int)sum(abs(mat1-mat2))[0];
}

double calculateAverageError(cv::Mat &image)
{;
	std::vector<std::string> files = getImagesPathsFromFolder(imageFolder);
	Mat centerFace = imread(files[12], IMREAD_GRAYSCALE);
	Mat mask = getFaceMask(centerFace);
	resize(mask, mask, image.size());
	return cv::mean(image, mask)[0];
}

cv::Mat depth2Normals(const cv::Mat& depth, Camera cam, int normalDistance, int bilateralF1, int bilateralF2, int gaussianF)
{
	Mat depth32F;
	Mat blurredDepth;
	depth.convertTo(depth32F, CV_32F);
	blurredDepth = depth32F;
	if (bilateralF2 != 0)
	{
		/// If bilatteral fitering is requested, overwrite the blurredDepth variable
		bilateralFilter(depth32F, blurredDepth, 0, bilateralF1, bilateralF2);	
		//bilateralFilter(depth32F, blurredDepth, 0, 4, 15);
	}

	float orthogonalDiff = float(2 * cam.pixelSize / cam.f);
	cv::Mat normals(depth.size(), CV_32FC3);
	int d = normalDistance;
	for (int y = d; y < depth.rows-d; ++y)
	{
		for (int x = d; x < depth.cols-d; ++x)
		{
			float dzdx = (blurredDepth.at<float>(y, x + d) - blurredDepth.at<float>(y, x - d)) / 2
				+ (blurredDepth.at<float>(y-d, x + d) - blurredDepth.at<float>(y-d, x - d)) / 4
				+ (blurredDepth.at<float>(y+d, x + d) - blurredDepth.at<float>(y+d, x - d)) / 4;
			float dzdy = (blurredDepth.at<float>(y - d, x) - blurredDepth.at<float>(y + d, x)) / 2
				+ (blurredDepth.at<float>(y - d, x+d) - blurredDepth.at<float>(y + d, x+d)) / 4
				+ (blurredDepth.at<float>(y - d, x-d) - blurredDepth.at<float>(y + d, x-d)) / 4;

			Vec3f d(-dzdx, -dzdy, orthogonalDiff * blurredDepth.at<float>(y, x));
			Vec3f n = normalize(d);
			normals.at<Vec3f>(y, x) = n;
		}
	}
	if (gaussianF != 0)
	{
		GaussianBlur(normals, normals, Size{ gaussianF,gaussianF }, 0, 0);
		//GaussianBlur(blurredDepth, blurredDepth, Size{ 79,79 },0,0); // VARIABLE
	}
	//showImage("Normals", (normals + 1) / 2, 1, true, 0.5);
	return normals;
}

cv::Mat getOrthogonalityFromCamera(const cv::Mat& depth, cv::Mat mask, cv::Mat& normals, Camera perspective, Camera orthogonality)
{
	Mat blurredDepth = depth;
	//blur(depth, blurredDepth, Size{ 3,3 });
	Vec3f camDiff3D = (Vec3f)(Vec3d)perspective.pos3D - (Vec3f)(Vec3d)orthogonality.pos3D;
	Mat camAngle{ depth.size(), CV_32FC3, Scalar{0,0,0} };
	float preMultX = float(perspective.pixelSize / perspective.fx);
	float preMultY = float(perspective.pixelSize / perspective.fy);
	//showImage("mask", mask);
	for (int v = 0; v < depth.rows; v++) {
		for (int u = 0; u < depth.cols; u++) {
			if (mask.at<uchar>(v, u) == 0) {
				continue;
			}
			float d = blurredDepth.at<float>(v, u);
			Vec3f pos3D(
				(u - perspective.principalPoint.x) * d * preMultX,
				-(v - perspective.principalPoint.y) * d * preMultY,
				d
			);
			pos3D = pos3D + camDiff3D;
			camAngle.at<Vec3f>(v, u) = normalize(pos3D);
		}
	}
	//showImage("depth", depth);
	//showImage("camAngle", abs(camAngle));
	//showImage("Normals", abs(normals));

	Mat normalOrthogonality = matrixDot(camAngle, normals);
	//Mat occlusion = getOcclusion(blurredDepth, camAngle, perspective, orthogonality);
	//saveImage("RenderOcclusions/C_" + std::to_string(orthogonality.pos3D.x) + "_" + std::to_string(orthogonality.pos3D.y), occlusion);
	//Mat occlusion = loadImage("RenderOcclusions/C_" + std::to_string(orthogonality.pos3D.x) + "_" + std::to_string(orthogonality.pos3D.y));
	//uchar data[] = { 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0 };
	//Mat kernel{ Size{5,5}, CV_8U, data};
	//showImage("occlusion", occlusion, 1, false);
	//dilate(occlusion, occlusion, kernel);
	//showImage("occlusion2", occlusion, 1, false);
	//erode(occlusion, occlusion, kernel);
	//erode(occlusion, occlusion, kernel);
	//showImage("occlusion3", occlusion);
	//multiply(normalOrthogonality, occlusion, normalOrthogonality, 1. / 255., normalOrthogonality.type());
	//cv::threshold(normalOrthogonality, normalOrthogonality, 0, FLT_MAX, cv::THRESH_TOZERO);
	normalOrthogonality.setTo(0, normalOrthogonality < 0);
	//showImage("normalOrthogonality", normalOrthogonality, 1, false);
	//showImage("occlusion", occlusion);
	return normalOrthogonality;
}

cv::Mat getPixelNormals(Camera& camera, Mat& image)
{
	cv::Mat normals(image.size(), CV_32FC3);
	float f = (float)camera.f;
	float pixelSize = (float)camera.pixelSize;
	for (int u = 0; u < image.cols; u++) {
		for (int v = 0; v < image.rows; v++) {
			Vec3f pixelNormal(
				(u - image.cols / 2) *	pixelSize, 
				(v - image.rows / 2) *	pixelSize, 
				f
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
			dotted.at<float>(v, u) = mat1.at<Vec3f>(v, u).dot(mat2.at<Vec3f>(v, u));
		}
	}
	//showImage("Dot", dotted);
	return dotted;
}

void plot2Dpoints(cv::Mat& mat, std::vector<cv::Point2i>& points2D)
{
	Mat_<uchar> plot{ mat.size() };
	for (auto &p : points2D) {
		if (p.x - 10 < 0 || p.x + 10 > mat.cols || p.y - 10 < 0 || p.y + 10 > mat.rows) continue;
		//std::cout << p << std::endl;
		plot(Rect{ Point2i(p), Size{10,10} }) = 255;
	}
	cv::resize(plot, plot, cv::Size(), 0.25, 0.25);
	cv::namedWindow("Plot", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("Plot", plot);
}

cv::Mat getOcclusion(const cv::Mat& depth, const cv::Mat& camAngles, Camera perspective, Camera occlusionCaster)
{
	Mat occlusion{ depth.size(), CV_8U , Scalar{255} };
	Point3d relPos = occlusionCaster.pos3D - perspective.pos3D;
	float orthogonalDiff = (float)(perspective.pixelSize / perspective.f);
	float invOrthDiff = 1 / orthogonalDiff;
	int halfRows = depth.rows / 2, halfCols = depth.cols / 2;
	for (int r = 0; r < depth.rows; r++)
	{
		for (int c = 0; c < depth.cols; c++)
		{
			double startDepth = depth.at<double>(r, c);
			Vec3f angle = camAngles.at<Vec3f>(r, c);
			double angle_times_orthogonalDiff = angle[2] * orthogonalDiff;
			double angle_times_relPosX = angle[2] * relPos.x, angle_times_relPosY = angle[2] * relPos.y;
			Point2i startPos2D{ c,r };
			Point2f endPos2D = Point2f{ angle[0] / angle[2] * invOrthDiff,-angle[1] / angle[2] * invOrthDiff } + Point2f{ float(depth.cols) / 2.f, float(depth.rows) / 2.f };
			Point2f delta2D = Point2f(startPos2D) - endPos2D;
			std::vector<Point2i> points = bresenhamDxDy(startPos2D, delta2D, depth.size());
			//std::cout << points.size() << std::endl;
			//Mat plot{ depth.size(), CV_8U , Scalar{255} };
			//plot2Dpoints(plot, points);
			//waitKey(0);
			double castDepth;
			int count = 0;
			for (Point2i p : points) {
				if (count > 250) break;
				if (relPos.x != 0 && relPos.y != 0)
				{
					castDepth =  -angle_times_relPosY / (angle[1] - angle_times_orthogonalDiff * -(p.y - halfRows));
					castDepth += -angle_times_relPosX / (angle[0] - angle_times_orthogonalDiff * (p.x - halfCols));
					castDepth /= 2;
				}
				else if (relPos.x != 0)
				{
					castDepth = -angle_times_relPosX / (angle[0] - angle_times_orthogonalDiff * (p.x - halfCols));
				}
				else
				{
					castDepth = -angle_times_relPosY / (angle[1] - angle_times_orthogonalDiff * -(p.y - halfRows));
				}

				if (castDepth < 0.7) break;
				if (castDepth-1e-7 > depth.at<double>(p))
				{
					//std::cout << startPos2D << ",  " << p << ", " << angle << ", " << castDepth << ", " << ", " << depth.at<double>(p) << std::endl;
					//std::cout << startPos2D << ",  " << p << ", " << castDepth << ", " << depth.at<double>(p) << std::endl;
					occlusion.at<uchar>(r, c) = 0;
					break;
				}
				count++;
			}
		}
	}
	//Mat showOff;
	//cv::multiply(occlusion, depth, showOff, 1./255., depth.type());
	//showImage("occlusion", showOff-0.6, 4);
	return occlusion;
}

cv::Mat blurWithMask(const cv::Mat& image, const cv::Mat& mask, int filterSize)
{
	Mat maskedImage;
	image.copyTo(maskedImage, mask);

	Mat floatMask;
	mask.convertTo(floatMask, CV_32F);

	Mat blurMask;
	GaussianBlur(floatMask, blurMask, Size(filterSize, filterSize), 0, 0);

	Mat blurredImage;
	GaussianBlur(maskedImage, blurredImage, Size(filterSize, filterSize), 0, 0);

	Mat weightedBlur;
	divide(blurredImage, blurMask, weightedBlur, 256, image.type());

	Mat maskedBlur;
	weightedBlur.copyTo(maskedBlur, mask);

	return maskedBlur;
}

cv::Mat getBlurredSlope(cv::Mat image, bool vertical, int blurKernelSize)
{
	char kdata[] = { -1, 0, 1 };
	Mat kernel;
	if (vertical) 
	{
		kernel = Mat(3, 1, CV_8S, kdata);
	}
	else
	{
		kernel = Mat(1, 3, CV_8S, kdata);
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
		//imwrite("Photographs\\Series1WhiteBalancedUndistorted\\" + std::to_string(i) + ".jpg", images[i]);
	}

}

void exportOBJfromDisparity(cv::Mat disparityImage, std::string fileName, Camera cam1, Camera cam2, float scale) 
{
	double preMult = norm(cam1.pos3D-cam2.pos3D) * cam1.f * 16 / cam1.pixelSize;
	//std::cout << "Premult: " << norm(cam1.pos3D - cam2.pos3D) << " * " << cam1.f << " = " << preMult << std::endl;
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

void exportXYZfromDisparity(cv::Mat disparityImage, std::string fileName, Camera cam, float camDistance, Point2i principalPoint, float scale, cv::Mat mask)
{
	double preMult = camDistance * cam.f * 16 / cam.pixelSize;
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
			if (mask.at<uchar>(u, v) > 0)
			{
				float xp = 1000 * Z(u, v) * cam.pixelSize * (v - principalPoint.x) / cam.fx;
				float yp = 1000 * Z(u, v) * cam.pixelSize * (u - principalPoint.y) / cam.fy;
				float zp = 1000 * Z(u, v);
				outputFile << -yp << " " << -xp << " " << -zp << std::endl;
				//outputFile << Z(u, v) * cam.pixelSize * (v - principalPoint.x) / cam.fx << " " << Z(u, v) * cam.pixelSize * (u - principalPoint.y) / cam.fy << " " << Z(u, v) << std::endl;

			}
		}
	}
	outputFile.close();
}


cv::Mat getDiffWithAbsoluteReference(Mat disparity, Rect area, bool verbose)
{
	Mat disparityHolder{ images[0].size(), CV_16SC1 };
	disparity.copyTo(disparityHolder(area));
	Mat ref = getIdealRef();
	Mat refDisparity = depth2Disparity(ref, cameras[0], cameras[1]);
	resize(disparityHolder, disparityHolder, refDisparity.size());
	disparityHolder = disparityHolder(Rect{ area.tl() * refDisparity.cols / images[0].cols, area.br() * refDisparity.cols / images[0].cols });
	refDisparity = refDisparity((Rect{ area.tl() * refDisparity.cols / images[0].cols, area.br() * refDisparity.cols / images[0].cols }));
	Mat difference;
	subtract(disparityHolder, refDisparity, difference, noArray(), disparityHolder.type());
	difference = abs(difference);
	difference.convertTo(difference, 0);
	if (verbose)
		showDifference("difference", disparityHolder, refDisparity, 800);
	return difference;
}

double getAvgDiffWithAbsoluteReference(Mat disparity, Rect area, bool verbose, std::string savePath)
{
	Mat difference = getDiffWithAbsoluteReference(disparity, area, verbose);

	if (savePath != "")
	{
		imwrite(savePath, difference*5);
	}
	return calculateAverageError(difference);

}

__forceinline uchar noiseResistantLocalBinaryWeight(ushort n1, ushort n2, ushort threshold)
{
	return ((n1 - n2) > threshold) + (abs(n1 - n2) <= threshold) * 2;
}

__forceinline uchar noiseResistantLocalBinaryCombine(uchar vals[8])
{
	//std::cout << (int)vals[0] << (int)vals[1] << (int)vals[2] << (int)vals[3] << (int)vals[4] << (int)vals[5] << (int)vals[6] << (int)vals[7] << std::endl;

	int counter=0;
denoise:;
	counter = 0;
	for (int i = 0; i < 8; i++)
	{
		int i_m = (i+7) % 8;
		int i_p = (i+1) % 8;
		if (vals[i] == 2) {
			if (vals[i_m] != 2) {
				vals[i] = vals[i_m];
			}
			else if (vals[i_p] != 2) {
				vals[i] = vals[i_p];
			}
		}
		counter += vals[i] != 2;
	}

	if (counter == 8)
	{
		goto combine;
	}
	else if (counter == 0)
	{
		return 0;
	}
	else goto denoise;
combine:;
	//std::cout << (int)vals[0] << (int)vals[1] << (int)vals[2] << (int)vals[3] << (int)vals[4] << (int)vals[5] << (int)vals[6] << (int)vals[7] << std::endl;
	return vals[0] + 
		(vals[1] << 1) + 
		(vals[2] << 2) + 
		(vals[3] << 3) + 
		(vals[4] << 4) + 
		(vals[5] << 5) + 
		(vals[6] << 6) + 
		(vals[7] << 7);
}

void noiseResistantLocalBinaryPattern(std::vector<cv::Mat>& images, ushort threshold)
{
	for (auto& im : images)
	{
		Mat_<uchar> hamming{ im.size() };
		ushort* imptr = im.ptr<ushort>(0);
		uchar* hamptr = hamming.ptr<uchar>(0);
		hamming = 0;
		uchar vals[8] = { 0 };
		ushort intensities[9] = { 0 };
		int d = 4;
		for (int r = d; r < im.rows - d; r++)
		{
			for (int c = d; c < im.cols - d; c++)
			{
				intensities[0] = imptr[(r - d) * im.cols + c]    ;
				intensities[1] = imptr[(r - d) * im.cols + c - d];
				intensities[2] = imptr[(r)	   * im.cols + c - d];
				intensities[3] = imptr[(r + d) * im.cols + c - d];
				intensities[4] = imptr[(r + d) * im.cols + c]    ;
				intensities[5] = imptr[(r + d) * im.cols + c + d];
				intensities[6] = imptr[(r)     * im.cols + c + d];
				intensities[7] = imptr[(r - d) * im.cols + c + d];
				intensities[8] = imptr[r	   * im.cols + c];

				vals[0] = noiseResistantLocalBinaryWeight(intensities[0], intensities[8], threshold);
				vals[1] = noiseResistantLocalBinaryWeight(intensities[1], intensities[8], threshold);
				vals[2] = noiseResistantLocalBinaryWeight(intensities[2], intensities[8], threshold);
				vals[3] = noiseResistantLocalBinaryWeight(intensities[3], intensities[8], threshold);
				vals[4] = noiseResistantLocalBinaryWeight(intensities[4], intensities[8], threshold);
				vals[5] = noiseResistantLocalBinaryWeight(intensities[5], intensities[8], threshold);
				vals[6] = noiseResistantLocalBinaryWeight(intensities[6], intensities[8], threshold);
				vals[7] = noiseResistantLocalBinaryWeight(intensities[7], intensities[8], threshold);
				//std::cout << intensities[0] << ", "
				//	<< intensities[1] << ", "
				//	<< intensities[2] << ", "
				//	<< intensities[3] << ", "
				//	<< intensities[4] << ", "
				//	<< intensities[5] << ", "
				//	<< intensities[6] << ", "
				//	<< intensities[7] << ", "
				//	<< intensities[8] << ", "
				//	<< std::endl;
				//std::cout << (int)vals[0] << ", "
				//	<< (int)vals[1] << ", "
				//	<< (int)vals[2] << ", "
				//	<< (int)vals[3] << ", "
				//	<< (int)vals[4] << ", "
				//	<< (int)vals[5] << ", "
				//	<< (int)vals[6] << ", "
				//	<< (int)vals[7] << ", "
				//	<< std::endl;
				hamptr[r * im.cols + c] = noiseResistantLocalBinaryCombine(vals);
				//waitKey(0);
			}
		}
		im = hamming;
		//showImage("im", im, 1, true, 0.2);
	}
}

void localBinaryThreadUchar(Mat& im, Mat& LBP, int distance)
{
	uchar* imptr = im.ptr<uchar>(0);
	uchar* hamptr = LBP.ptr<uchar>(0);
	uchar vals[8] = { 0 };
	int d = distance;
	int ddiag = int(cv::sqrt(distance * distance/2)+0.5);
	for (int r = d; r < im.rows - d; r++)
	{
		for (int c = d; c < im.cols - d; c++)
		{
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - d	  ) * im.cols + c]		  );
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - ddiag) * im.cols + c - ddiag]) << 1;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r		  ) * im.cols + c - d]	  ) << 2;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + ddiag) * im.cols + c - ddiag]) << 3;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + d	  ) * im.cols + c]		  ) << 4;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + ddiag) * im.cols + c + ddiag]) << 5;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r		  ) * im.cols + c + d]	  ) << 6;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - ddiag) * im.cols + c + ddiag]) << 7;
		}
	}
}

void localBinaryThreadUshort(Mat& im, Mat& LBP, int distance)
{
	ushort* imptr = im.ptr<ushort>(0);
	uchar* hamptr = LBP.ptr<uchar>(0);
	uchar vals[8] = { 0 };
	int d = distance;
	int ddiag = int(cv::sqrt(distance * distance / 2) + 0.5);
	for (int r = d; r < im.rows - d; r++)
	{
		for (int c = d; c < im.cols - d; c++)
		{
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - d) * im.cols + c]);
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - ddiag) * im.cols + c - ddiag]) << 1;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r)*im.cols + c - d]) << 2;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + ddiag) * im.cols + c - ddiag]) << 3;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + d) * im.cols + c]) << 4;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + ddiag) * im.cols + c + ddiag]) << 5;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r)*im.cols + c + d]) << 6;
			hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - ddiag) * im.cols + c + ddiag]) << 7;
		}
	}
}

void localBinaryPattern(std::vector<cv::Mat>& images, std::vector<cv::Mat>& LBPs, int distance)
{
	LBPs.clear();
	std::vector<std::thread> threads;
	for (int i = 0; i < images.size(); i++)
	{
		LBPs.push_back(Mat(images[i].size(), CV_8U, Scalar(0)));
	}
	if (images[0].type() == CV_8U) {
		for (int i = 0; i < images.size(); i++)
		{
			threads.push_back(std::thread(localBinaryThreadUchar, std::ref(images[i]), std::ref(LBPs[i]), distance));
		}
	}
	else if (images[0].type() == CV_16U)
	{
		for (int i = 0; i < images.size(); i++)
		{
			threads.push_back(std::thread(localBinaryThreadUshort, std::ref(images[i]), std::ref(LBPs[i]), distance));
		}
	}

	for (auto& thr : threads)
	{
		thr.join();
	}
	for (int i = 0; i < images.size(); i++)
	{
		if(images[i].type()!=CV_8U)
			images[i].convertTo(images[i], CV_8U, 1. / 256.);
	}
}

void blurImages(std::vector<cv::Mat>& images, float blurSigma)
{
	for (auto& im : images) 
	{
		Mat_<uchar> hamming{ im.size() };
		ushort* imptr = im.ptr<ushort>(0);
		uchar* hamptr = hamming.ptr<uchar>(0);
		hamming = 0;
		uchar vals[8] = { 0 };
		for (int r = 1; r < im.rows-1; r++)
		{
			for (int c = 1; c < im.cols-1; c++)
			{
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - 1) * im.cols + c]    );
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - 1) * im.cols + c - 1]) << 1 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r	)   * im.cols + c - 1])	<< 2 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + 1) * im.cols + c - 1]) << 3 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + 1) * im.cols + c]	  ) << 4 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r + 1) * im.cols + c + 1]) << 5 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r	)   * im.cols + c + 1]) << 6 ;
				hamptr[r * im.cols + c] += (imptr[r * im.cols + c] <= imptr[(r - 1) * im.cols + c + 1]) << 7 ;
			}
		}
		im = hamming;
		//showImage("im", im, 1, true, 2);
	}

		//Mat blurred;
		//Mat floatIm;
		//im.convertTo(floatIm, CV_32FC1);
		////Mat floatImSquared = floatIm.mul(floatIm);
		//GaussianBlur(floatIm, blurred, Size{ 0, 0 }, blurSigma);
		//GaussianBlur(floatIm, floatIm, Size{ 0, 0 }, blurSigma / 5.);
		//subtract(floatIm, blurred, floatIm, noArray());
		//floatIm = floatIm * 5 + 128;
		//floatIm.convertTo(im, CV_8UC1);

		//Mat blurred;
		//Mat blurredSquared;
		//Mat floatIm;
		//Mat stdDev;
		//im.convertTo(floatIm, CV_32FC1);
		//GaussianBlur(floatIm, blurred, Size{ 31,31 }, 0);
		//Mat floatImSquared = floatIm.mul(floatIm);
		//GaussianBlur(floatImSquared, blurredSquared, Size{ 31,31 }, 0);
		//cv::sqrt(blurredSquared - blurred.mul(blurred), stdDev);
		//subtract(floatIm, blurred, floatIm, noArray());
		//floatIm /= stdDev / 10;
		//floatIm += 128;
		//floatIm.convertTo(im, CV_8UC1);
		//showImage("im", im);
}

void normalizeMats(std::vector<cv::Mat>& images)
{
	Mat sumMat{ images[0].size(), images[0].type(), Scalar{0} };
	for (int i = 0; i < images.size(); i++)
	{
		sumMat += images[i];
	}
	for (int i = 0; i < images.size(); i++)
	{
		images[i] /= sumMat;
	}
}

void subImgsAndCamsAndSurfs(std::vector<int> ids, std::vector<Mat>& outputMats, std::vector<Camera>& outputCams, std::vector<cv::Mat>& outputSurfs)
{
	outputMats.clear();
	outputCams.clear();
	outputSurfs.clear();
	for (auto i : ids) {
		outputMats.push_back(images[i]);
		outputCams.push_back(cameras[i]);
		outputSurfs.push_back(faceNormals[i]);
	}
}

void subImgsAndCams(std::vector<int> ids, std::vector<Mat>& outputMats, std::vector<Camera>& outputCams)
{
	outputMats.clear();
	outputCams.clear();
	for (auto i : ids) {
		outputMats.push_back(images[i]);
		outputCams.push_back(cameras[i]);
	}
}

inline short median(std::vector<short>& v)
{
	size_t n = v.size() / 2;
	nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}

Mat getVectorMatsAverage(std::vector<cv::Mat>& mats)
{
	Mat sum = Mat{ mats[0].size(), mats[0].type(), Scalar{0} };
	std::vector<short> values;
	for (int r = 0; r < mats[0].rows; r++)
	{
		for (int c = 0; c < mats[0].cols; c++)
		{
			values.clear();
			for (int m = 0; m < mats.size(); m++)
			{
				values.push_back(mats[m].at<short>(r, c));
			}
			sum.at<short>(r, c) = median(values);
		}
	}
	//for (auto &m : mats)
	//{
	//	sum += m;
	//}
	//sum /= mats.size();
	return sum;
}

void makeArrayCollage(std::vector<cv::Mat> images, cv::Size arrayShape, float multiplier, float scale)
{
	Mat collage;
	std::vector<Mat> hConcats(arrayShape.height);
	for (int i = 0; i < arrayShape.height; i++)
	{
		std::vector<Mat> subVector;
		for (int j = 0; j < arrayShape.width; j++)
		{
			subVector.push_back(images[i * arrayShape.height + j]);
		}
		cv::hconcat(subVector, hConcats[i]);
	}
	cv::vconcat(hConcats, collage);
	showImage("Collage", collage, multiplier, false, scale);
}

std::vector<int> testSGMCamIDs;

int testSGMminD;
int testSGMnumD;
cv::Rect testSGMarea;
Mat testSGMgroundTruth;

namespace plt = matplotlibcpp;

static void pixelSGM(int event, int x, int y, int flags, void* param)
{
	if (event == cv::MouseEventTypes::EVENT_LBUTTONDOWN)
	{
		x *= 2;
		y *= 2;
		std::cout << x << ", " << y << std::endl;

		plt::figure(1);
		plt::clf();
	//	for (int id : testSGMCamIDs) {
	//		std::vector<float> error = calcDisparityCostForPlotting(images, cameras,
	//			id,x + testSGMarea.x, y + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> error2 = calcDisparityCostForPlotting(images, cameras,
	//			id, x + testSGMarea.x+1, y + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> error3 = calcDisparityCostForPlotting(images, cameras,
	//			id, x + testSGMarea.x-1, y + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> error4 = calcDisparityCostForPlotting(images, cameras,
	//			id, x + testSGMarea.x, y + 1 + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> error5 = calcDisparityCostForPlotting(images, cameras,
	//			id, x + testSGMarea.x-1, y + 1 + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> error6 = calcDisparityCostForPlotting(images, cameras,
	//			id, x + testSGMarea.x+1, y + 1 + testSGMarea.y, testSGMminD, testSGMminD + testSGMnumD);
	//		std::vector<float> sumError;
	//		for (int i = 0; i < error.size(); i++)
	//		{
	//			sumError.push_back(error[i] + error2[i] + error3[i]+ error4[i] + error5[i] + error6[i]);
	//		}
	//		plt::plot(sumError, { {"label", "error cam" + std::to_string(id)}, {"linewidth", "0.5"} });
	//		if (!testSGMgroundTruth.empty())
	//		{
	//			plt::axvline(testSGMgroundTruth.at<short>(y, x) / 16 - testSGMminD, 0, 1, { {"linewidth", "0.3"} });
	//		}
	//		calcPixelArrayIntensity(images[id], cameras[id], cameras[12], testSGMminD, 
	//			testSGMminD + testSGMnumD, Point2i(x + testSGMarea.x, y + testSGMarea.y));
	//	}
	//	plt::legend();
	//	plt::save("temp\\Costs.jpg");
	//	//plt::show();
	//	//Mat intImg = imread("temp\\Intensities.jpg");
	//	Mat costImg = imread("temp\\Costs.jpg");
	//	//Mat concat;
	//	//vconcat(intImg, costImg, concat);
	//	showImage("Intensities and costs", costImg, 1, false, 0.8);
	//	//plt::show();
	}
}

void testSGM(Mat disparity, std::vector<int> camIDs, int minD, int numD, Rect area, Mat groundTruth)
{
	testSGMCamIDs = camIDs;
	testSGMminD = minD;
	testSGMnumD = numD;
	testSGMarea = area;
	testSGMgroundTruth = groundTruth;
	cv::namedWindow("Disparity Click", cv::WindowFlags::WINDOW_AUTOSIZE);
	Mat resizedDisparity;
	resize(disparity, resizedDisparity, Size{}, 0.5 / scale, 0.5 / scale);
	if (resizedDisparity.type() == CV_8U)
	{
		cv::imshow("Disparity Click", resizedDisparity);
	}
	else 
	{
		cv::imshow("Disparity Click", (resizedDisparity - 4240 * scale) * 30/scale);
	}
	cv::setMouseCallback("Disparity Click", pixelSGM, &images);
	cv::waitKey(0);
}
//
//Point3d calibrateSingleCameraOrthogonal(int originCam, int calibrateCam, float step, StereoArraySGBM& sgbm, cv::Rect area)
//{
//	float minAvg = FLT_MAX;
//	Point3d bestPos;
//	std::vector<Mat> subImages;
//	std::vector<Camera> subCameras;
//	std::vector<Mat> empty{};
//	std::vector<float> avgs;
//	std::vector<double> is;
//	subImgsAndCams({ originCam, calibrateCam }, subImages, subCameras);
//	Point3d camDiff = subCameras[1].pos3D - subCameras[0].pos3D;
//	Point3d stepDirection = Point3d{ camDiff.y, -camDiff.x, 0 };
//	stepDirection = (Point3d)normalize((Vec3d)stepDirection);
//	Mat disparity;
//	int count = 1;
//	for (int i = 0; i <= 16; i++) {
//		subCameras[1].pos3D = cameras[calibrateCam].pos3D + (i - 8) * step * stepDirection;
//		is.push_back(norm((i - 8) * step * stepDirection)*((i-8>0)*2-1));
//		sgbm.computeMinCost(subImages, empty, subCameras, 12, area, disparity, 0.05);
//		//showImage("minCost", disparity, 1, true);
//		float avg = cv::mean(disparity)[0];
//		std::cout << avg << ", " << subCameras[1].pos3D << std::endl;
//		avgs.push_back(avg);
//		if (avg < minAvg) 
//		{
//			minAvg = avg;
//			bestPos = subCameras[1].pos3D;
//		}
//		else if (avg == minAvg)
//		{
//			count += 1;
//			bestPos = bestPos * float(count - 1) / (float)count + subCameras[1].pos3D * (1. / (float)count);
//		}
//	}
//	std::cout << bestPos << std::endl;
//	plt::figure(1);
//	plt::clf();
//	plt::plot(is, avgs);
//	plt::save("temp\\OrthCost.jpg");
//	Mat intImg = imread("temp\\OrthCost.jpg");
//	showImage("orthogonal cost", intImg, 1, false, 1);
//	std::cout << "Best position for camera " << calibrateCam << ": " << bestPos << std::endl;
//	return bestPos;
//}
//
//Point3d calibrateSingleCameraParallel(int originCam, int calibrateCam, float step, StereoArraySGBM& sgbm, cv::Rect area, Mat referenceDisparity)
//{
//	float minAvg = FLT_MAX;
//	Point3d bestPos;
//	std::vector<Mat> subImages;
//	std::vector<Camera> subCameras;
//	std::vector<Mat> empty{};
//	std::vector<float> avgs;
//	std::vector<double> is;
//	subImgsAndCams({ originCam, calibrateCam }, subImages, subCameras);
//	Point3d camDiff = subCameras[1].pos3D - subCameras[0].pos3D;
//	Point3d stepDirection = (Point3d)normalize((Vec3d)camDiff);
//	Mat disparity;
//	int count = 1;
//	for (int i = 0; i <= 16; i++) {
//		subCameras[1].pos3D = cameras[calibrateCam].pos3D + (i - 8) * step * stepDirection;
//		is.push_back(norm((i - 8) * step * stepDirection) * ((i - 8 > 0) * 2 - 1));
//		sgbm.compute(subImages, empty, subCameras, 12, area, disparity);
//		float avg = cv::mean(abs(disparity-referenceDisparity))[0];
//		std::cout << avg << ", " << subCameras[1].pos3D << std::endl;
//		avgs.push_back(avg);
//		if (avg < minAvg)
//		{
//			minAvg = avg;
//			bestPos = subCameras[1].pos3D;
//		}
//		else if (avg == minAvg)
//		{
//			count += 1;
//			bestPos = bestPos * float(count - 1) / (float)count + subCameras[1].pos3D * (1. / (float)count);
//		}
//	}
//	plt::figure(1);
//	plt::clf();
//	std::cout << bestPos << std::endl;
//	plt::plot(is, avgs);
//	std::cout << "Best position for camera " << calibrateCam << ": " << bestPos << std::endl;
//	plt::save("temp\\ParCost.jpg");
//	Mat intImg = imread("temp\\ParCost.jpg");
//	showImage("parallel cost", intImg, 1, false, 1);
//	return bestPos;
//}
//
//void calibrateCamerasFromOrigin(cv::Rect area, StereoArraySGBM& sgbm)
//{
//	showImage("pattern", images[12](area));
//	float step = 0.75e-4;
//	for (int c = 0; c < cameras.size(); c++)
//	{
//		if (c == 12) continue;
//		std::cout << "Orthogonal calibration of camera " << c << std::endl;
//		cameras[c].pos3D = calibrateSingleCameraOrthogonal(12, c, step, sgbm, area);
//	}
//
//	std::vector<Point3d> camPositions;
//	for (auto& c : cameras)
//		camPositions.push_back(c.pos3D);
//	FileStorage camPosFile(imageFolder + "\\cameraPositionEstimateDerivative.xyz", FileStorage::WRITE);
//	camPosFile << "camera_positions" << camPositions;
//	camPosFile << "camera_focal_length" << cameras[0].f;
//	camPosFile << "camera_pixel_size" << cameras[0].pixelSize;
//	camPosFile.release();
//	std::vector<Mat> empty{};
//	Mat disparity;
//	sgbm.compute(images, empty, cameras, 12, area, disparity);
//	showImage("Disparity", disparity-4300, 70, false, 1);
//	for (int c = 0; c < cameras.size(); c++)
//	{
//		if (c == 12) continue;
//		std::cout << "Parallel calibration of camera " << c << std::endl;
//		cameras[c].pos3D = calibrateSingleCameraParallel(12, c, step, sgbm, area, disparity);
//	}
//
//	camPositions.clear();
//	for (auto& c : cameras)
//		camPositions.push_back(c.pos3D);
//	FileStorage camPosFile2(imageFolder + "\\cameraPositionEstimateDerivative2.xyz", FileStorage::WRITE);
//	camPosFile2 << "camera_positions" << camPositions;
//	camPosFile2 << "camera_focal_length" << cameras[0].f;
//	camPosFile2 << "camera_pixel_size" << cameras[0].pixelSize;
//}

cv::Vec3f getBallScreenParams(cv::Mat image, int minSize, int maxSize)
{
	std::vector<cv::Vec3f> circles;
	//image.convertTo(image, CV_8U, 1. / 256.);
	HoughCircles(image, circles, HOUGH_GRADIENT, 3, 99999, 400, 100, minSize, maxSize);
	Mat img = image.clone();
	cvtColor(img, img, COLOR_GRAY2BGR);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(img, center, radius, Scalar(0, 255, 255), 1, 8, 0);
		std::cout << radius << std::endl;
	}
	if (circles.size() != 1)
		throw(std::to_string(circles.size()) + " circles found, while 1 circle was expected.");
	Rect roi = Rect{ Point2f(circles[0][0] - circles[0][2] * 1.1f, circles[0][1] - circles[0][2] * 1.1f),
				Point2f(circles[0][0] + circles[0][2] * 1.1f, circles[0][1] + circles[0][2] * 1.1f) };
	showImage("CirclePlot", img(roi), 1, false, 0.7f);
	return circles[0];
}

cv::Mat generateBallDepthMap(cv::Size imageSize, cv::Vec3f ballScreenParameters, Camera camParameters, double ballRadius)
{

	Mat invK;
	cv::invert(K, invK);
	double mX = invK.at<double>(0, 0);
	double mY = invK.at<double>(1, 1);
	double pX = invK.at<double>(0, 2);
	double pY = invK.at<double>(1, 2);

	Mat_<float> ballDepth{imageSize};
	ballDepth = FLT_MAX/2;
	double Sp = camParameters.pixelSize;
	//float f =  (float) camParameters.f;
	double f = (K.at<double>(0, 0) + K.at<double>(1, 1))/2;
	double a = (ballScreenParameters[2]);
	double depth = ballRadius * sqrt(f * f / (a * a) + 1); // Distance between camera plane and ball, corrected for apparent size.
	std::cout << "ball depth: " << depth << std::endl;
	std::cout << "ball screen params: " << ballScreenParameters << std::endl;
	double angleX = atan2((ballScreenParameters[0] - ballScreenParameters[2]) * mX + pX, 1)/2 + atan2((ballScreenParameters[0] + ballScreenParameters[2]) * mX + pX, 1)/2;
	double angleY = atan2((ballScreenParameters[1] - ballScreenParameters[2]) * mY + pY, 1)/2 + atan2((ballScreenParameters[1] + ballScreenParameters[2]) * mY + pY, 1)/2;
	double ballScreenXcenter = tan(angleX);
	double ballScreenYcenter = tan(angleY);
	Vec3d ballWorldParameters{
		ballScreenXcenter,
		ballScreenYcenter,
		1
	};
	ballWorldParameters = normalize(ballWorldParameters) * depth;
	Vec3d bwp = ballWorldParameters;

	for (int r = 0; r < ballDepth.rows; r++)
	{
		for (int c = 0; c < ballDepth.cols; c++)
		{
			Vec3d unitCamRay = normalize(Vec3d{ double(c)*mX + pX,double(r)*mY + pY, 1.});
			Vec3d pointClosestToBall = bwp.dot(unitCamRay)/unitCamRay.dot(unitCamRay) * unitCamRay;
			double distance = cv::norm(pointClosestToBall - bwp);
			if (distance > ballRadius) continue;
			//std::cout << unitCamRay << std::endl;
			//std::cout << pointClosestToBall << std::endl;
			//std::cout << "dist " << distance << std::endl;
			double ballInterceptDistance = sqrt(ballRadius * ballRadius - distance * distance);
			Vec3d ballInterceptPoint = pointClosestToBall - unitCamRay * ballInterceptDistance;
			//std::cout << r << ", " << c << ": " << ballInterceptPoint << std::endl;
			ballDepth(r, c) = (float)ballInterceptPoint[2];
		}
	}
	return ballDepth;
}

std::vector<float> movingAverage(std::vector<float>& inputVector, int windowSize)
{
	std::vector<float> mvAverage(inputVector.size(), 0);
	mvAverage[0] = inputVector[0];
	for (int i = 1; i < windowSize; i++)
	{
		mvAverage[i] = mvAverage[i - 1] * (i - 1) / i + inputVector[i] / i;
	}
	for (int i = windowSize; i < mvAverage.size(); i++)
	{
		mvAverage[i] = mvAverage[i - 1] - inputVector[i - windowSize] / windowSize + inputVector[i] / windowSize;
	}
	return mvAverage;
}

std::vector<float> centeredMovingAverage(std::vector<float>& inputVector, int windowSize)
{
	int vectorSize = (int)inputVector.size();
	std::vector<float> mvAverage(inputVector.size(), 0);
	int halfWindow = windowSize / 2;
	int i = 0;
	for (i = 0; i < halfWindow; i++)
	{
		mvAverage[0] += inputVector[i] / halfWindow;
	}
	for (i = 1; i < halfWindow; i++)
	{
		mvAverage[i] = (mvAverage[i - 1] * (halfWindow + i - 1) + inputVector[i+halfWindow-1]) / (halfWindow + i);
	}
	for (; i < vectorSize - halfWindow; i++)
	{
		mvAverage[i] = mvAverage[i - 1] - inputVector[i - halfWindow] / windowSize + inputVector[i + halfWindow - 1] / windowSize;
	}
	for (; i < vectorSize; i++)
	{
		mvAverage[i] = (mvAverage[i - 1] * (vectorSize - i + halfWindow) - inputVector[i - halfWindow]) 
			/ (vectorSize - i + halfWindow - 1);
	}
	return mvAverage;
}

std::vector<float> centeredMovingAverageAbsoluteDeviation(std::vector<float>& inputVector, int windowSize)
{
	std::vector<float> avg = centeredMovingAverage(inputVector, windowSize);
	std::vector<float> diff(inputVector.size(), 0);
	for (int i = 0; i < inputVector.size(); i++)
	{
		diff[i] = abs(inputVector[i] - avg[i]);
	}
	return centeredMovingAverage(diff, windowSize);
}

std::vector<std::pair<float, float>> getSortedOrthogonalityDifference(cv::Mat depth, cv::Mat groundTruth, cv::Mat_<float> orthogonality, cv::Mat_<uchar> mask)
{
	Mat_<float> depthDiff = (depth - groundTruth);
	std::vector< std::pair <float, float> > pairs;
	for (int r = 1; r < depth.rows; r++)
	{
		for (int c = 1; c < depth.cols; c++)
		{
			if (mask(r, c) == 0)
				continue;
			pairs.push_back(std::pair(orthogonality(r, c), depthDiff(r, c)));
		}
	}
	sort(pairs.begin(), pairs.end());
	return pairs;
}
