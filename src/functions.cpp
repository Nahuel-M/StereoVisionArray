#include <fstream>
#include <iostream>
#include <filesystem>

#include "functions.h"
#include "Camera.h"
#include "dlibFaceSelect.h"

using namespace cv;

cv::Mat improveWithDisparity(cv::Mat& disparity, cv::Mat centerImage, std::vector<cv::Mat> &images, std::vector<std::array<Camera, 2>> &cameras, int windowSize)
{
    Mat mask = getFaceMask(centerImage);
    Size resolution = disparity.size();
    Mat improvedDisparity{ resolution, disparity.type() };
    Mat newDisparity{ disparity.size(), disparity.type() };
    int kernelSize = (windowSize - 1) / 2;
    for (int c = 0; c < cameras.size(); c++) {
        std::array<Camera,2> cam = cameras[c];
        imshow("Disparity", disparity);
        imshow("Before", images[c]);
        Mat shifted = shiftPerspectiveWithDisparity(cam[0], cam[1], disparity, images[c]);
        Point2d distance = Point2d{ cam[0].pos3D.x - cam[1].pos3D.x, cam[0].pos3D.y - cam[1].pos3D.y };
        distance.x = distance.x / norm(distance.x) && (distance.x>0.001);
        distance.y = distance.y / norm(distance.y) && (distance.y > 0.001);
        std::cout << distance << std::endl;
        for (int y = 0; y < centerImage.rows; y++) {
            for (int x = 0; x < centerImage.cols; x++) {
                if (mask.at<uint8_t>(Point(x, y)) == 0) continue;
                Mat window = centerImage(Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
                std::vector<double> error;
                for (int p = 0; p <= 10; p++) {
                    Point2i newP = Point2i{ x, y } + (Point2i) distance * (p - 5);
                    Mat compWindow = shifted(Rect{ newP - Point2i{kernelSize, kernelSize}, newP + Point2i{kernelSize, kernelSize} });
                    error.push_back(getAbsDiff(compWindow, window));
                }
                int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
                newDisparity.at<unsigned char>(y, x) = disparity.at<unsigned char>(y, x) + (maxIndex - 5)*(distance.x+distance.y);
            }
        }

        imshow("Shifted", shifted);
        waitKey(0);
    }
    double camDistance = norm(cameras[0][0].pos3D - cameras[0][1].pos3D);
    Mat pixSizeDisp;
    //multiply(newDisparity, cameras[0][0].pixel_size, pixSizeDisp, 1, 6);
    //improvedDisparity = camDistance * cameras[0][0].f / (pixSizeDisp);
    //std::cout << "Test" << std::endl;

    return newDisparity;
}


cv::Mat shiftPerspectiveWithDisparity(Camera& inputCam, Camera& outputCam, cv::Mat& disparity, cv::Mat& image)
{
    Mat shiftedImage = Mat{ image.size() , image.type() };
    double camDistance = norm(inputCam.pos3D - outputCam.pos3D);
    //improvedDepth = camDistance * outputCam.f / (pixSizeDisp);

    double preMultX = (inputCam.pos3D.x - outputCam.pos3D.x) / norm(inputCam.pos3D - outputCam.pos3D);
    double preMultY = (inputCam.pos3D.y - outputCam.pos3D.y) / norm(inputCam.pos3D - outputCam.pos3D);
    for (int y = 0; y < shiftedImage.rows; y++) {
        for (int x = 0; x < shiftedImage.cols; x++) {
            double disp = disparity.at<unsigned char>(y, x);
            if (disp == 0) {
                continue;
            }
            int shiftedX = disp * preMultX + x;
            int shiftedY = disp * preMultY + y;
            if (shiftedY >= disparity.rows || shiftedY < 0 || shiftedX >= disparity.cols || shiftedX < 0)
                continue;
            shiftedImage.at<unsigned char>(y, x) = image.at<unsigned char>(shiftedY, shiftedX);
        }
    }
    return shiftedImage;
}

cv::Mat shiftPerspective2(Camera inputCam, Camera outputCam, cv::Mat &depthMap)
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
        if(ptrImage->type() == 0)
            std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage->at<unsigned char>(Point(x, y)) << std::endl;
        else
            std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage->at<double>(Point(x, y)) << std::endl;
    }


}

void showImage(std::string name, Mat image) {
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
    Mat mask = getFaceMask(image);
    return cv::mean(image, mask)[0];
}