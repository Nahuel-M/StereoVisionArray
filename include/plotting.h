#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <string>
#include "matplotlibcpp.h"

//void plot(std::vector<double> x, std::vector<double> y);

void plotNoseBridge(cv::Mat image, std::string name = "");

void showPlot();