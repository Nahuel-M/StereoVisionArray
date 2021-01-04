#include "plotting.h"

namespace plt = matplotlibcpp;

void plot(std::vector<double> x, std::vector<double> y)
{
	plt::figure_size(1200, 780);
	plt::plot(x, y);
}