#pragma once
#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <string>
#include <opencv2/core.hpp>
#pragma warning (pop)


std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

void showImage(std::string name, cv::Mat image, double multiplier = 1, bool hold = true, float scale = 0.3);

void showDifference(std::string name, cv::InputArray image1, cv::InputArray image2, double multiplier = 1, cv::Mat mask = cv::Mat{});

cv::Mat getIdealRef();

void saveImage(std::string filename, cv::Mat image);

template<typename T>
void saveVector(std::string filename, std::vector<T>& vec, int reductionFactor)
{
	std::ofstream file;
	file.open(filename);
	int count = 0;
	for (T m : vec)
	{
		count++;
		if (count%reductionFactor==0)
			file << m << std::endl;
	}
	file.close();
}

template <typename T>
void operator+=(std::vector<T>& v, const T add) {
	for (T& val : v)
		val += add;
}

template <typename T1, typename T2>
void splitPairs(std::vector<std::pair<T1, T2>>& pairs, std::vector<T1>& v1, std::vector<T2>& v2)
{
	for (auto it = std::make_move_iterator(pairs.begin()),
		end = std::make_move_iterator(pairs.end()); it != end; ++it)
	{
		v1.push_back(std::move(it->first));
		v2.push_back(std::move(it->second));
	}
}

cv::Mat loadImage(std::string filename);
