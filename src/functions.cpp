#include <fstream>
#include <iostream>
#include <filesystem>

#include "functions.h"
#include "Camera.h"
#include "dlibFaceSelect.h"

using namespace cv;

std::vector<std::array<int, 2>> getCameraPairs(std::vector<Camera>& cameras, pairType pair) {
    std::vector<std::array<int, 2>> pairs;
    if (pair == TO_CENTER) {
        for (int i = 0; i < cameras.size(); i++) {
            if (i == 12) continue;
            pairs.push_back({ 12, i });
        }
    }
    else if (pair == TO_CENTER_SMALL) {
            pairs.push_back({ 12, 6 });
            pairs.push_back({ 12, 7 });
            pairs.push_back({ 12, 8 });
            pairs.push_back({ 12, 11 });
            pairs.push_back({ 12, 13 });
            pairs.push_back({ 12, 16 });
            pairs.push_back({ 12, 17 });
            pairs.push_back({ 12, 18 });
    }
    else if (pair == MID_LEFT) {
        pairs.push_back({ 12, 11 });
    }
    else if (pair == MID_TOP) {
        pairs.push_back({ 12, 7 });
    }
    else if (pair == LINE_HORIZONTAL) {
        for (int i = 10; i < 15; i++) {
            if (i == 12) continue;
            pairs.push_back({ 12, i });
        }
    }
    else if (pair == LINE_VERTICAL) {
        for (int i = 2; i < 25; i+=5) {
            if (i == 12) continue;
            pairs.push_back({ 12, i });
        }
    }
    else if (pair == CROSS) {
            pairs.push_back({ 12, 11 });
            pairs.push_back({ 12, 13 });
            pairs.push_back({ 12, 7 });
            pairs.push_back({ 12, 17 });
    }
    else if (pair == JUMP_CROSS) {
        pairs.push_back({ 12, 10 });
        pairs.push_back({ 12, 14 });
        pairs.push_back({ 12, 2 });
        pairs.push_back({ 12, 24 });
    }
    return pairs;
}

double getAbsDiff(cv::Mat& mat1, cv::Mat& mat2)
{
    return sum(abs(mat1-mat2))[0];
}

void showImage(std::string name, Mat image) {
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, 710, 540);
    imshow(name, image);
}

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath)
{
    namespace fs = std::filesystem;
    std::vector<std::string> filePaths;
    for (auto& p : fs::directory_iterator(folderPath))
    {
        filePaths.push_back(p.path().u8string());
        //std::cout << p.path().u8string() << std::endl;
    }
    return filePaths;
}

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

cv::Mat getIdealRef() {
    Mat R;
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
    Mat R;
    cv::FileStorage file;
    file.open(filename, cv::FileStorage::READ);
    file["image"] >> R;
    return R;
}

double calculateAverageError(cv::Mat &image)
{
    std::string folder = "Images";
    std::vector<std::string> files = getImagesPathsFromFolder(folder);
    Mat mask = getFaceMask();
    return cv::mean(image, mask)[0];
}