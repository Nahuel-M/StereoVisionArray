#include "bresenham.h"
#include <opencv2/core.hpp>
#include <iostream>
using namespace cv;

std::vector<Point2i> plotLineLow(int x0, int y0, int x1, int y1) {
	std::vector<Point2i> points;
	int dx = x1 - x0;
	int dy = y1 - y0;
	int yi = 1;
	if (dy < 0) {
		yi = -1;
		dy = -dy;
	}
	int D = 2 * dy - dx;
	int y = y0;

	for (int x = x0; x <= x1; x++) {
		points.push_back(Point2i(x, y));
		if (D > 0) {
			y = y + yi;
			D = D - 2 * dx;
		}
		D = D + 2 * dy;
	}
	return points;
}

std::vector<Point2i> plotLineHigh(int x0, int y0, int x1, int y1) {
	std::vector<Point2i> points;
	int dx = x1 - x0;
	int dy = y1 - y0;
	int xi = 1;
	if (dx < 0) {
		xi = -1;
		dx = -dx;
	}
	int D = 2 * dx - dy;
	int x = x0;

	for (int y = y0; y <= y1; y++) {
		points.push_back(Point2i(x, y));
		if (D > 0) {
			x = x + xi;
			D = D - 2 * dy;
		}
		D = D + 2 * dx;
	}
	return points;
}

std::vector<Point2i> bresenham(Point2i point1, Point2i point2)
{
	std::vector<Point2i> points;
	if (abs(point2.y - point1.y) < abs(point2.x - point1.x)) {
		if (point1.x > point2.x)
		{
			points = plotLineLow(point2.x, point2.y, point1.x, point1.y);
			std::reverse(points.begin(), points.end());
			return points;
		}
		else
		{
			return plotLineLow(point1.x, point1.y, point2.x, point2.y);
		}
	}
	else
	{
		if (point1.y > point2.y) {
			points = plotLineHigh(point2.x, point2.y, point1.x, point1.y);
			std::reverse(points.begin(), points.end());
			return points;
		}
		else
		{
			return plotLineHigh(point1.x, point1.y, point2.x, point2.y);
		}
	}
}

std::vector<cv::Point2i> bresenhamDxDy(cv::Point2i point1, cv::Point2f delta, cv::Size imageSize)
{
	float dx = delta.x;
	float dy = delta.y;
	float x0, y0;
	(dx < 0) ?  x0 = point1.x / -dx : x0 = (imageSize.width-1-point1.x + FLT_MIN)/dx;	// Check how much delta is needed to hit image edges for x,y
	(dy < 0) ?  y0 = point1.y / -dy : y0 = (imageSize.height-1-point1.y+ FLT_MIN)/dy;
	float min0 = min(x0, y0);									// Get the minimum delta to hit an edge
	//min0 = min0 * !isnan(min0) + max(x0, y0) * isnan(min0);		// Check to remove 0/0 results
	Point2i point2 = point1 + Point2i(delta * min0);	// Calculate the point at this edge
	//std::cout << dx << ", " << dy << ", " << x0 << ", " << y0 << ", " << point1 << ", " << point2 << std::endl;
	return bresenham(point1, point2);
}