#include "bresenham.h"
#include <opencv2/core.hpp>

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

std::vector<Point2i> bresenham(Point2i point2, Point2i point1)
{
	if (abs(point2.y - point1.y) < abs(point2.x - point1.x)) {
		if (point1.x > point2.x)
		{
			return plotLineLow(point2.x, point2.y, point1.x, point1.y);
		}
		else
		{
			return plotLineLow(point1.x, point1.y, point2.x, point2.y);
		}
	}
	else
	{
		if (point1.y > point2.y) {
			return plotLineHigh(point2.x, point2.y, point1.x, point1.y);
		}
		else
		{
			return plotLineHigh(point1.x, point1.y, point2.x, point2.y);
		}
	}
}