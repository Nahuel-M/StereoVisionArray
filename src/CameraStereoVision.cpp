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
    std::string folder = "Renders2";
    std::vector<std::string> files = getImagesPathsFromFolder(folder);
    std::vector<Mat> images;
    for (int i = 0; i < files.size(); i++) {
        images.push_back(imread(files[i], IMREAD_GRAYSCALE));
        resize(images.back(), images.back(), Size(), 0.5, 0.5);
    }

    Mat mask = getFaceMask(images.back());

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
    Mat disparity = Mat{ images[12].size(), CV_8UC1};
    double camDistance = norm(cameras[pairs[0][0]].pos3D - cameras[pairs[0][1]].pos3D);

    for (int y = kernelSize; y < (resolution.height - kernelSize); y++) {
        //for (int y = resolution.height/2; y < (resolution.height - kernelSize);  y++) {
        for (int x = kernelSize; x < resolution.width - kernelSize; x++) {
            //for (int x = resolution.width/2; x < resolution.width - kernelSize; x++) {
            if (mask.at<uint8_t>(Point(x, y)) == 0) continue;

            for (auto pair : pairs) {
                
                Mat kernel = images[pair[0]](Rect{ Point2i{x - kernelSize, y - kernelSize}, Point2i{x + kernelSize, y + kernelSize} });


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
                std::vector<double> error;

                for (auto p : pixels) {
                    Rect selector = Rect{ p - Point(kernelSize, kernelSize), p + Point(kernelSize, kernelSize) };
                    Mat selection = images[pair[1]](selector);
                    Mat result{ CV_32FC1 };

                    error.push_back(getAbsDiff(selection, kernel));

                }

                int maxIndex = std::distance(error.begin(), std::min_element(error.begin(), error.end()));

                Point2i pixel = pixels[maxIndex];
                //std::cout << norm(pixel - Point2i{ x, y }) << std::endl;
                disparity.at<unsigned char>(Point(x, y)) = (int) norm(pixel - Point2i{ x, y });

                
            }

        }
    }
    //imshow("Disp", disparity);

    Mat pixSizeDisp;
    multiply(disparity, pixelSize, pixSizeDisp, 1, 6);
    depth =  camDistance * f / (pixSizeDisp);
    std::cout << "Test" << std::endl;

    namedWindow("Depth", 1);
    //setMouseCallback("My Window", CallBackFunc, NULL);
    showImage("Depth", depth);
    //waitKey(0);
    Mat ref = getIdealRef();
    Mat depth2;
    resize(depth, depth2, ref.size());
    Mat error = (depth2 - ref) * 50;
    showImage("Error", error);
    std::vector<Mat> inpImages = { images[pairs[0][1]] };
    std::vector < std::array<Camera, 2> > inpCameras = { {cameras[pairs[0][0]], cameras[pairs[0][1]]} };
    Mat improvedDepth = improveWithDisparity(disparity, images[pairs[0][0]], inpImages, inpCameras, 21);
    namedWindow("Depth2", 1);
    showImage("Depth2", improvedDepth);
    waitKey(0);
    resize(improvedDepth, depth2, ref.size());
    error = (depth2 - ref) * 50;
    showImage("Error2", error);

    waitKey(0);
}

