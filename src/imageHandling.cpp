
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
		else if (ptrImage.type() == 18)
		{
			std::cout << "Vec3US at position (" << x << ", " << y << ")" << ": " << ptrImage.at<cv::Vec<ushort,3>>(cv::Point(x, y)) << std::endl;
		}
		else if (ptrImage.type() == 19)
		{
			std::cout << "Vec3SS at position (" << x << ", " << y << ")" << ": " << ptrImage.at<cv::Vec<short, 3>>(cv::Point(x, y)) << std::endl;
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

void showImage(std::string name, cv::Mat image, double multiplier, bool hold) {
	cv::resize(image, image, cv::Size(), 0.4, 0.4);
	cv::namedWindow(name, cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow(name, image * multiplier);
	if (hold)
	{
		cv::setMouseCallback(name, CallBackFuncs, &image);
		cv::waitKey(0);
	}
}

void showDifference(std::string name, cv::Mat image1, cv::Mat image2, double multiplier)
{
	cv::Mat diff;
	cv::subtract(image1, image2, diff, cv::noArray(), CV_16S);

	cv::Mat pDiff = diff.clone();
	cv::Mat nDiff = -diff.clone();
	pDiff.setTo(0, pDiff < 0);
	nDiff.setTo(0, nDiff < 0);

	std::vector<cv::Mat> channels;
	cv::Mat z =cv::Mat::zeros(cv::Size(nDiff.cols, nDiff.rows), nDiff.type());
	channels.push_back(nDiff);
	channels.push_back(z);
	channels.push_back(pDiff);

	cv::Mat merged;
	cv::merge(channels, merged);

	cv::namedWindow(name, cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow(name, 710, 540);
	cv::resize(merged, merged, cv::Size{ 710,540 }, 0, 0, 0);
	cv::resize(diff, diff, cv::Size{ 710,540 });
	cv::imshow(name, merged * multiplier);
	cv::setMouseCallback(name, CallBackFuncs, &diff);
	cv::waitKey(0);
}
