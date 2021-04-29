
#include <iostream>

#include <filesystem>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "imageHandling.h"

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
		if(p.path().extension()==".jpg" || p.path().extension() == ".png")
			filePaths.push_back(p.path().u8string());
		//std::cout << p.path().u8string() << std::endl;
	}
	std::sort(filePaths.begin(), filePaths.end(), compareNat);
	return filePaths;
}


cv::Mat getIdealRef() {
	cv::Mat R;
	cv::FileStorage file;
	try
	{
		file.open("idealRef_0_84.yml", cv::FileStorage::READ);
	}
	catch (cv::Exception & e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	file["R"] >> R;
	cv::flip(R, R, 0);
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
		std::cout << "Type: " << ptrImage.type() << " (" << cv::typeToString(ptrImage.type()) << "), ";
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
		else if (ptrImage.type() == 16)
		{
			std::cout << "Vec3UCHAR at position (" << x << ", " << y << ")" << ": " << ptrImage.at<cv::Vec<uchar, 3>>(cv::Point(x, y)) << std::endl;
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

void showImage(std::string name, cv::Mat image, double multiplier, bool hold, float scale) {
	if (scale > 1) {
		cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_NEAREST);
	}
	else if (scale < 1)
	{
		cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_CUBIC);
	}
	cv::namedWindow(name, cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow(name, image * multiplier);
	if (hold)
	{
		cv::setMouseCallback(name, CallBackFuncs, &image);
		cv::waitKey(0);
	}
	else 
	{
		cv::waitKey(1);
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
	//cv::Mat z =cv::Mat::zeros(cv::Size(nDiff.cols, nDiff.rows), nDiff.type());
	channels.push_back(nDiff);
	channels.push_back(pDiff);
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
