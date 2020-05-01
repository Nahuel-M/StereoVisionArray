#include <iostream>
#include <iterator>
#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"
#include "matplotlibcpp.h"
#include "generateIdealReference.h"
#include <string>

using namespace cv;

double f = 0.05;
double sensor_size = 0.036;

Mat generateDepthFromImages(std::vector<Mat> &images, const std::vector<std::array<int, 2>> &pairs, std::vector<Camera> &cameras);

void CallBackFunc(int event, int x, int y, int flags, void* param)
{
    Mat* ptrImage = (Mat*)param;
    if (event == EVENT_LBUTTONDOWN)
    {
        std::cout << "at position (" << x << ", " << y << ")" << ": "<< ptrImage->at<double>(Point(x,y)) << std::endl;
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
    std::vector< std::vector<std::array<int, 2>> > groups = getGroups(cameras, "CHESS");
    std::vector<Mat> perspectives;
    std::vector<Mat> projected;
    Mat example = loadImage("perspectivesR025K20\\cam0");
    Mat combined = Mat{ example.size(), example .type() };
    Mat divides = Mat{ example.size(), CV_8U, Scalar(0)};
    int i = 0;
    for (auto group : groups) {
        perspectives.push_back(generateDepthFromImages(images, group, cameras));
        //perspectives.push_back( loadImage("perspectivesR025K20\\cam" + std::to_string(group.back()[0])) );
        projected.push_back( shiftPerspective2(cameras[group.back()[0]], cameras[12], perspectives[i]) );
        combined += projected.back();
        divides = divides + (projected.back() > 0.1)/255;
        i++;
        showImage(std::to_string(group.back()[0]), perspectives.back());
        waitKey(0);
        //saveImage("cam" + std::to_string(group.back()[0]), perspectives.back());
    }
    Mat plot; 
    normalize(combined, plot);
    imshow("CombinedPre", plot*255);
    waitKey(0);

    //std::cout << combined << std::endl;
    cv::divide(combined, divides, combined, 1, CV_64F) ; 
    //std::cout << combined.size() << ", " << divides.size() << std::endl;

    bitwise_and(combined, 0, combined, (divides == 0));
    //combined = combined & (divides > 0);

    //std::cout << combined << std::endl;
    normalize(combined, plot);
    imshow("CombinedPost", plot * 255);
    waitKey(0);


    //Mat leftGroup = generateDepthFromImages(images, getCameraPairs(cameras, CROSS, 10), cameras);
    //saveImage("LeftGroupR025K10", leftGroup);


    Mat ref = getIdealRef();
    namedWindow("Rec", WINDOW_NORMAL);
    resizeWindow("Rec", 710, 540);
    Mat refC;
    showImage("ref", ref);
    //showImage("im", im);
    Mat diff = abs(ref - combined);
    Mat diffStore;
    diff.copyTo(diffStore, (ref < 1));
    //std::cout << calculateAverageError(diff) << std::endl;
    //saveImage("DifferenceCrossR100K20", diff);
    namedWindow("Diff", WINDOW_NORMAL);
    resizeWindow("Diff", 710, 540);
    imshow("Diff", diffStore*150);
    //setMouseCallback("Diff", CallBackFunc, NULL);
    waitKey(0);
    return 0;
    
}

Mat generateDepthFromImages(std::vector<Mat>& images, const std::vector<std::array<int, 2>>& pairs, std::vector<Camera>& cameras) {

    //Mat mask = getFaceCircle(images[12]);
    Size resolution = Size{ images[12].cols, images[12].rows };
    Point2i halfRes = resolution / 2;
    double pixelSize = sensor_size / resolution.width;

    int kernelSize = 10;
    Mat depth = Mat{ images[12].size(), CV_64FC1 };
    int pairCount = pairs.size();
    for (int x = kernelSize; x < resolution.width - kernelSize; x++) {
        std::cout << x << std::endl;
        for (int y = kernelSize; y < (resolution.height - kernelSize); y++) {
            //if (mask.at<uint8_t>(Point(x, y)) == 0) continue;

            for (auto pair : pairs) {
                double camDistance = norm(cameras[pair[0]].pos3D - cameras[pair[1]].pos3D);
                double preMult = camDistance * f / pixelSize / pairCount;
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
    //showImage("Depth", (depth - 0.5) * 3.333);
    ////saveImage("CrossR050K10", depth);
    //waitKey(0);
    return depth;
}