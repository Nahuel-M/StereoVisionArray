#pragma once

#pragma warning (push, 0)	/// Disabling warnings for external libraries
#include <vector>
#include <opencv2/core.hpp>
#include <string>
#include "matplotlibcpp.h"
#pragma warning (pop)

void plotNoseBridge(cv::Mat image, std::string name = "");

void showPlot();