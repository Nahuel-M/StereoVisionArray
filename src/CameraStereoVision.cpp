#include <iostream>
#include <iterator>
#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"
#include "matplotlibcpp.h"

using namespace cv;

struct package 
{
    std::vector<Mat>& images; 
    std::vector<Camera>& cameras;
};

void showCalculation(std::vector<Camera> &cameras, std::vector<Mat> &images, int x, int y)
{
    Mat ref = getIdealRef();
    std::vector<std::array<int,2>> pairs = getCameraPairs(cameras, CROSS);
    int kernelSize = 10;

    double f = 0.05;
    double sensor_size = 0.036;
    Size resolution = Size{ images[12].cols, images[12].rows };
    std::cout << "Image Resolution: " << resolution << std::endl;
    Point2i halfRes = resolution / 2;
    Point2i centerPixel = resolution / 2;
    double pixelSize = sensor_size / resolution.width;
    for (auto pair : pairs) {
        double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
        Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
        Mat im1cop = images[pair[0]].clone();
        im1cop.at<uint8_t>(Point(x,y)) = 255;
        Mat im2cop = images[pair[1]].clone();

        Point3d vec = cameras[pair[0]].inv_project(Point2i{ x, y }-halfRes);
        Point3d p1 = cameras[pair[0]].pos3D + (vec * 0.5);
        Point3d p2 = cameras[pair[0]].pos3D + vec;
        Point2i pixel1 = cameras[pair[1]].project(p1) + halfRes;
        Point2i pixel2 = cameras[pair[1]].project(p2) + halfRes;

        if (pixel1.x<kernelSize || pixel1.y < kernelSize || pixel1.x > resolution.width - kernelSize || pixel1.y > resolution.height - kernelSize) {
            continue;
        }
        if (pixel2.x<kernelSize || pixel2.y < kernelSize || pixel2.x > resolution.width - kernelSize || pixel2.y > resolution.height - kernelSize) {
            continue;
        }

        std::vector<Point2i> pixels = bresenham(pixel1, pixel2);
        std::vector<float> error;
        std::vector<float> depths;

        for (auto p : pixels) {
            Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
            Mat selection = images[pair[1]](selector);
            Mat result{ CV_32FC1 };
            matchTemplate(selection, kernel, result, TM_CCORR_NORMED);
            error.push_back(result.at<float>(0, 0));
            depths.push_back(camDistance * f / (pixelSize * norm(p - Point2i{ x, y })));
            im2cop.at<uint8_t>(p) = 0;
        }

        std::cout << std::endl;
        std::cout << "  Plotting" << std::endl;
        matplotlibcpp::figure(1);
        int max = std::distance(error.begin(), std::max_element(error.begin(), error.end()));
        matplotlibcpp::axvline(depths[max]);
        matplotlibcpp::plot(depths, error);
        std::cout << "  Plotted" << std::endl;
        int maxIndex = std::distance(error.begin(), std::max_element(error.begin(), error.end()));
        Point2i pixel = pixels[maxIndex];
        //depth.at<double>(Point(x, y)) = camDistance * f / (pixelSize * norm(pixel - Point2i{ x, y }));
        im2cop.at<uint8_t>(pixel) = 255;
        //imshow("im2", im2cop);
    }
    std::cout << ref.at<double>(Point(x, y)) << std::endl;
    double truth = ref.at<double>(Point(x, y));
    //matplotlibcpp::ylim(0.9, 1.05);
    matplotlibcpp::xlim(0.5, 1.0);
    matplotlibcpp::xlabel("Depth (m)");
    matplotlibcpp::ylabel("Normalized Cross Correlation");

    matplotlibcpp::axvline(truth, 0.0, 1.0, { {"color", "red"}, {"linestyle", "--"} });
    matplotlibcpp::show();

}


void CallBackFunc(int event, int x, int y, int flags, void* param)
{
    package* pck = (package*)param;
    //package pk = *pck;
    if (event == EVENT_LBUTTONDOWN)
    {
        showCalculation(pck->cameras, pck->images, x, y);
        //std::cout << "at position (" << x << ", " << y << ")" << ": " << ptrImage->at<double>(Point(x, y)) << std::endl;
    }


}

int main()
{
    // Images
    std::string folder = "Images";
    std::vector<std::string> files = getImagesPathsFromFolder(folder);
    std::vector<Mat> images;
    for (int i = 0; i < files.size(); i++) {
        images.push_back(imread(files[i], IMREAD_GRAYSCALE));
        resize(images.back(), images.back(), Size(), 0.25, 0.25);
    }

    Mat mask = getFaceMask();

    // Camera parameters
    double f = 0.05;
    double sensor_size = 0.036;
    Size resolution = Size{ images[12].cols, images[12].rows };
    std::cout << "Image Resolution: " << resolution << std::endl;
    Point2i halfRes = resolution / 2;
    Point2i centerPixel = resolution / 2;
    double pixelSize = sensor_size / resolution.width;
    std::cout << "Pixel Size: " << pixelSize << std::endl;

    std::vector<Camera> cameras;
    for (int y = 0; y < 5; y++) {

        for (int x = 0; x < 5; x++) {
            cameras.push_back(Camera(f, Point3d{ -0.1 + x * 0.05, -0.1 + y * 0.05, -0.75 }, pixelSize));
        }
    }

    // Pairs
    std::vector<std::array<int,2>> pairs = getCameraPairs(cameras, CROSS);

    namedWindow("Face", WINDOW_NORMAL);
    resizeWindow("Face", 710, 540);
    imshow("Face", images[pairs[0][0]]);
    package pck{ images, cameras};
    setMouseCallback("Face", CallBackFunc, (void*)&pck);
    waitKey(0);

    //int kernelSize = 10;
    //Mat depth = Mat{ images[12].size(), CV_64FC1};

    //for (int y = resolution.height/2 + 70; y < (resolution.height - kernelSize);  y++) {
    //    for (int x = resolution.width / 2; x < resolution.width - kernelSize; x++) {
    //        if (mask.at<uint8_t>(Point(x, y)) == 0) continue;

    //        for (auto pair : pairs) {
    //            double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
    //            Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
    //            Mat im1cop = images[pair[0]].clone();
    //            im1cop.at<uint8_t>(Point(x,y)) = 255;
    //            Mat im2cop = images[pair[1]].clone();

    //            Point3d vec = cameras[pair[0]].inv_project(Point2i{ x, y }-halfRes);
    //            Point3d p1 = cameras[pair[0]].pos3D + (vec * 0.5);
    //            Point3d p2 = cameras[pair[0]].pos3D + vec;
    //            Point2i pixel1 = cameras[pair[1]].project(p1) + halfRes;
    //            Point2i pixel2 = cameras[pair[1]].project(p2) + halfRes;

    //            if (pixel1.x<kernelSize || pixel1.y < kernelSize || pixel1.x > resolution.width - kernelSize || pixel1.y > resolution.height - kernelSize) {
    //                continue;
    //            }
    //            if (pixel2.x<kernelSize || pixel2.y < kernelSize || pixel2.x > resolution.width - kernelSize || pixel2.y > resolution.height - kernelSize) {
    //                continue;
    //            }

    //            std::vector<Point2i> pixels = bresenham(pixel1, pixel2);
    //            std::vector<float> error;
    //            std::vector<float> depths;

    //            for (auto p : pixels) {
    //                Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
    //                Mat selection = images[pair[1]](selector);
    //                Mat result{ CV_32FC1 };
    //                matchTemplate(selection, kernel, result, TM_CCORR_NORMED);
    //                error.push_back(result.at<float>(0, 0));
    //                depths.push_back(camDistance * f / (pixelSize * norm(p - Point2i{ x, y })));
    //                im2cop.at<uint8_t>(p) = 0;
    //            }
    //            matplotlibcpp::plot(depths, error);
    //            int maxIndex = std::distance(error.begin(), std::max_element(error.begin(), error.end()));
    //            Point2i pixel = pixels[maxIndex];
    //            depth.at<double>(Point(x, y)) = camDistance * f / (pixelSize * norm(pixel - Point2i{ x, y }));
    //            im2cop.at<uint8_t>(pixel) = 255;
    //            imshow("im2", im2cop);
    //            waitKey();
    //        }
    //        matplotlibcpp::show();

    //    }
    //}
    //showImage("Depth", depth);
    //waitKey(0);
}

