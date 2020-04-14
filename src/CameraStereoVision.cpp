#include <iostream>
#include <iterator>
#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"
#include "matplotlibcpp.h"

using namespace cv;

int main()
{
    // Images
    std::string folder = "Images";
    std::vector<std::string> files = getImagesPathsFromFolder(folder);
    std::vector<Mat> images;
    for (int i = 0; i < files.size(); i++) {
        images.push_back(imread(files[i], IMREAD_GRAYSCALE));
        //resize(images.back(), images.back(), Size(), 0.5, 0.5);
    }

    Mat mask = getFaceMask(images[12]);

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
    std::vector<std::array<int,2>> pairs = getCameraPairs(cameras, MID_LEFT);

    int kernelSize = 20;
    Mat depth = Mat{ images[12].size(), CV_64FC1};

    for (int x = kernelSize; x < resolution.width - kernelSize; x++) {
    //for (int x = resolution.width/2; x < resolution.width - kernelSize; x++) {
        std::cout << x << std::endl;
        for (int y = kernelSize; y < (resolution.height - kernelSize);  y++) {
        //for (int y = resolution.height/2; y < (resolution.height - kernelSize);  y++) {
            if (mask.at<uint8_t>(Point(x, y)) == 0) continue;

            for (auto pair : pairs) {
                double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
                Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });
                //imshow("Kernel", kernel);
                //Mat im1cop = images[pair[0]].clone();
                //im1cop.at<uint8_t>(Point(x,y)) = 255;
                //Mat im2cop = images[pair[1]].clone();
                //imshow("im1", im1cop);

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

                for (auto p : pixels) {
                    Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
                    Mat selection = images[pair[1]](selector);
                    Mat result{ CV_32FC1 };
                    //matchTemplate(selection, kernel, result, TM_CCORR_NORMED);
                    //error.push_back(result.at<float>(0, 0));
                    error.push_back(getAbsDiff(selection, kernel));
                    //depthS.push_back(camDistance * f / (pixelSize * norm(p - Point2i{ x, y })));

                    //im2cop(selector) = 30;
                    //im2cop.at<uint8_t>(p) = 0;
                    //std::cout << result.at<float>(0, 0) << std::endl;
                    //imshow("Selection", selection);
                    //waitKey(0);
                }
                //imshow("im2", im2);
                //waitKey(0);
                //matplotlibcpp::plot(depthS, error);
                int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
                //std::cout << "MaxValue: " << error[maxIndex] << " at " << maxIndex << std::endl;
                //float err = *std::max_element(error.begin(), error.end());
                //std::cout << "Max value: " << err << " at: " << maxIndex << std::endl;
                Point2i pixel = pixels[maxIndex];
                depth.at<double>(Point(x, y)) = depth.at<double>(Point(x, y)) + camDistance * f / (pixelSize * norm(pixel - Point2i{ x, y }));
                //std::cout << "Depth: " << camDistance * f / (pixelSize * norm(pixel - Point2i{ x, y })) << std::endl;
                //imshow("Depth", depth);
                //im2cop.at<uint8_t>(pixel) = 255;
                //Mat kernel2 = im2(Rect{ Point2i{pixel.x - kernelSize, pixel.y - kernelSize}, Point2i{pixel.x + kernelSize, pixel.y + kernelSize} });
                //imshow("Ker1", kernel);
                //imshow("Ker2", kernel2);
                //imshow("im2", im2cop);
                //waitKey();
            }
            //matplotlibcpp::show();

        }
        //std::cout << x << std::endl;
    }
    namedWindow("Depth", 1);
    //setMouseCallback("My Window", CallBackFunc, NULL);
    imshow("Depth", depth);
    waitKey(0);
}

