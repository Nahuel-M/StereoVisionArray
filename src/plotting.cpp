#include "plotting.h"

namespace plt = matplotlibcpp;
//
//void plot(std::vector<double> x, std::vector<double> y)
//{
//	plt::figure_size(1200, 780);
//	plt::plot(x, y);
//}

void plotNoseBridge(cv::Mat image, std::string name)
{
	cv::Mat noseLineMat = image(cv::Rect(325*2, 303*2, 65*2, 1));
	cv::Mat noseLineMat2;
	noseLineMat.copyTo(noseLineMat2);

	std::vector<ushort> noseLineVec;
	noseLineVec.assign((ushort*)noseLineMat2.datastart, (ushort*)noseLineMat2.dataend);

	if (name == "") {
		std::cout << "Oei";
		plt::plot(noseLineVec);
	}
	else {
		plt::plot(noseLineVec, { {"label", name} });

	}


	//plt::show();
}

void showPlot()
{
	plt::legend();
	plt::show();
}
