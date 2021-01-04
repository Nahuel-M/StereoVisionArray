
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imageHandling.h"

cv::Mat getIdealRef() {
	cv::Mat R;
	cv::FileStorage file;
	file.open("idealRef.yml", cv::FileStorage::READ);
	file["R"] >> R;
	return R;
}

void saveImage(std::string filename, cv::Mat image)
{
	cv::FileStorage file(filename, cv::FileStorage::WRITE);
	// Write to file!
	file << "image" << image;

}

cv::Mat loadImage(std::string filename)
{
	cv::Mat R;
	cv::FileStorage file;
	file.open(filename, cv::FileStorage::READ);
	file["image"] >> R;
	return R;
}

static void CallBackFuncs(int event, int x, int y, int flags, void* param)
{
	cv::Mat& ptrImage = *((cv::Mat*)param);
	if (event == cv::MouseEventTypes::EVENT_LBUTTONDOWN)
	{
		std::cout << "Type: " << ptrImage.type() << ", ";
		//std::cout << ptrImage.type() << std::endl;
		if (ptrImage.type() == 0) {
			std::cout << "at position (" << x << ", " << y << ")" << ": " << (int)ptrImage.at<unsigned char>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 1)
		{
			std::cout << "at position (" << x << ", " << y << ")" << ": " << (int)ptrImage.at<schar>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 2)
		{
			std::cout << "at position (" << x << ", " << y << ")" << ": " << (int)ptrImage.at<unsigned short>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 3)
		{
			std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage.at<signed short>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 5)
		{
			std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage.at<float>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 21)
		{
			std::cout << "Vec3f at position (" << x << ", " << y << ")" << ": " << ptrImage.at<cv::Vec3f>(cv::Point(x, y)) << std::endl;
		}
		else
		{
			std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage.at<double>(cv::Point(x, y)) << std::endl;
		}
	}


}

void showImage(std::string name, cv::Mat image) {
	cv::namedWindow(name, cv::WindowFlags::WINDOW_NORMAL);
	cv::Mat imHolder;
	cv::resizeWindow(name, 710, 540);
	cv::resize(image, imHolder, cv::Size{ 710,540 }, 0, 0, 0);
	cv::imshow(name, imHolder);
	cv::setMouseCallback(name, CallBackFuncs, &imHolder);
	cv::waitKey(0);
}
