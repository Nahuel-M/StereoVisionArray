#include <fstream>
#include <iostream>
#include <filesystem>

#include "functions.h"
#include "Camera.h"
#include "dlibFaceSelect.h"

using namespace cv;


cv::Mat shiftPerspective(Camera inputCam, Camera outputCam, cv::Mat depthMap, cv::Mat mask)
{
    Mat shiftedDepthMap = Mat{ depthMap.size() , depthMap.type() };
    double preMultX = (inputCam.pos3D.x - outputCam.pos3D.x) * inputCam.f / inputCam.pixel_size;
    double preMultY = (inputCam.pos3D.y - outputCam.pos3D.y) * inputCam.f / inputCam.pixel_size;
    Point2i halfRes = depthMap.size() / 2;
    for (int x = 0; x < shiftedDepthMap.cols; x++) {
        for (int y = 0; y < shiftedDepthMap.rows; y++) {
            
            Point3d vec = outputCam.inv_project(Point2i{ x, y }-halfRes);
            Point3d p1 = outputCam.pos3D + (vec * 0.5);
            Point3d p2 = outputCam.pos3D + vec;
            Point2i pixel1 = inputCam.project(p1) + halfRes;
            Point2i pixel2 = inputCam.project(p2) + halfRes;

            std::vector<Point2i> pixels = bresenham(pixel1, pixel2);
            std::vector<float> error;

            for (auto p : pixels) {
                Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
                Mat selection = images[pair[1]](selector);
                Mat result{ CV_32FC1 };
                error.push_back(getAbsDiff(selection, kernel));
            }

            int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));

            Point2i pixel = pixels[maxIndex];
            depth.at<double>(Point(x, y)) = depth.at<double>(Point(x, y)) + preMult / (norm(pixel - Point2i{ x, y }));

        }
    }
}

cv::Mat shiftPerspective2(Camera inputCam, Camera outputCam, cv::Mat depthMap)
{
    Mat shiftedDepthMap = Mat{ depthMap.size() , depthMap.type() };
    double preMultX = (inputCam.pos3D.x - outputCam.pos3D.x) * inputCam.f / inputCam.pixel_size;
    double preMultY = (inputCam.pos3D.y - outputCam.pos3D.y) * inputCam.f / inputCam.pixel_size;
    int direction[] = { outputCam.pos3D.x - inputCam.pos3D.x, outputCam.pos3D.y - inputCam.pos3D.y };
    //std::cout << "direction0 " << direction[0] << std::endl;
    for (int x = 0; x < depthMap.cols; x++) {
        for (int y = 0; y < depthMap.rows; y++) {
            double depth = depthMap.at<double>(y,x);
            if (depth < 0.5) 
                continue;
            int shiftedX = int(preMultX / depth) + x;
            int shiftedY = int(preMultY / depth) + y;
            if (shiftedY >= depthMap.rows || shiftedY < 0 || shiftedX >= depthMap.cols || shiftedX < 0) 
                continue;
            shiftedDepthMap.at<double>(Point(shiftedX, shiftedY)) = depth;
        }
    }

    //cv::imshow("Map", shiftedDepthMap);
    //cv::imshow("Original", depthMap);
    //cv::waitKey(0);
    return shiftedDepthMap;
}

std::vector< std::vector<std::array<int, 2>> > getGroups(std::vector<Camera> &cameras, std::string groupType)
{
    std::vector< std::vector<std::array<int, 2>> > groups;
    if (groupType == "CHESS") {
        for (int i = 0; i < 25; i += 2) {
            groups.push_back(getCameraPairs(cameras, CROSS, i));
        }
    }
    return groups;
}

cv::Mat Points3DToDepthMap(std::vector<Point3d>& points, Camera camera, cv::Size resolution)
{
    Mat depthMap = Mat{ resolution, CV_64FC1 };
    std::cout << depthMap.size() << std::endl;
    Point2i halfRes = resolution / 2;
    for (auto p : points) 
    {
        Point2i pixel = camera.project(p) + halfRes;
        if (pixel.x >= 0 && pixel.x < resolution.width && pixel.y >= 0 && pixel.y < resolution.height) {
            //std::cout << p << ", " << pixel << std::endl;
            depthMap.at<double>(pixel) = p.z - camera.pos3D.z;
        }
    }
    return depthMap;
}

std::vector<Point3d> DepthMapToPoints3D(cv::Mat& depthMap, Camera camera, cv::Size resolution)
{
    Point2i halfRes = resolution / 2;
    std::vector<Point3d> Points;
    for (int u = 0; u < depthMap.cols; u++) {
        for (int v = 0; v < depthMap.rows; v++) {
            double depth = depthMap.at<double>(Point(u, v));
            if (depth > 0.1)
                Points.push_back( camera.pos3D + camera.inv_project(Point(u,v)-halfRes) * depth );
        }
    }
    return Points;
}

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair) {
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

std::vector<std::array<int, 2>> getCameraPairs(const std::vector<Camera>& cameras, const pairType pair, const int cameraNum) {
    std::vector<std::array<int, 2>> pairs;
    if (pair == CROSS) {
        if(cameraNum-5>0)
            pairs.push_back({ cameraNum, cameraNum - 5 });
        if(cameraNum+5<25)
            pairs.push_back({ cameraNum, + 5 });
        if(cameraNum%5>0)
            pairs.push_back({ cameraNum, cameraNum - 1 });
        if (cameraNum % 5 < 4)
            pairs.push_back({ cameraNum, cameraNum + 1 });
        
    }
    return pairs;
}

double getAbsDiff(cv::Mat& mat1, cv::Mat& mat2)
{
    return sum(abs(mat1-mat2))[0];
}

void CallBackFuncs(int event, int x, int y, int flags, void* param)
{
    Mat* ptrImage = (Mat*)param;
    if (event == EVENT_LBUTTONDOWN)
    {
        std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage->at<unsigned char>(Point(x, y)) << std::endl;
    }


}

void showImage(std::string name, Mat &image) {
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, 710, 540);
    imshow(name, image);
    setMouseCallback(name, CallBackFuncs, (void*)&image);
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