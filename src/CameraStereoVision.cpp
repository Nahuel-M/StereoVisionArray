#include <iostream>
#include <iterator>
#include "Camera.h"
#include "functions.h"
#include "dlibFaceSelect.h"
#include "plotting.h"
namespace plt = matplotlibcpp;
using namespace cv;

int main()
{
    std::vector<double> heat;
    std::vector<double> err;

    /// Images
    std::vector<cv::Mat> images = getImages("Renders2", 0.5);

    /// Cameras
    std::vector<Camera> cameras = getCameras(images.back().size());

    /// Mask
    std::vector<cv::Point2i> maskPoints = getFaceMaskPoints(images.back());
    Mat centerMask = drawMask(images.back(), maskPoints);
    Mat centerDisparity = getDisparityFromPairSGM(images[12], images[13], centerMask, cameras[12], cameras[13]);
    Mat centerDepth = disparity2Depth(centerDisparity, cameras[12], cameras[13]);
    std::vector<cv::Point3d> maskPoints3D = Points2DtoPoints3D(centerDepth, maskPoints, cameras[12]);

    /// Pairs
    std::vector<std::array<int,2>> pairs = getCameraPairs(cameras, MID_BOTTOM);

    Mat reference = getIdealRef();
    Mat disparityRef = depth2Disparity(reference, cameras[12], cameras[11]);
    Mat disparityResize;
    Mat combinedDisparity{ images[0].size(), CV_16U, Scalar(0) };
    Mat combinedHeatmap{ images[0].size(), CV_32F, Scalar(0) };

    for (int c = 0; c < 25; c += 2)
    {
        /// Create mask from current camera perspective
        std::vector<cv::Point2i> maskPoints = Points3DtoPoints2D(maskPoints3D, cameras[c]);
        Mat mask = drawMask(images.back(), maskPoints);

        std::vector<std::array<int, 2>> pairs = getCameraPairs(cameras, CROSS, c);
        Mat interimDisparity{ images[0].size(), CV_32F, Scalar(0) };
        Mat weight{ images[0].size(), CV_32F, Scalar(0) };
        std::vector<Mat> subDisparities;
        std::vector<Mat> weightedSubDisparities;
        std::vector<Mat> weights;


        Mat deltaH = getBlurredSlope(images[c], 0);
        Mat deltaV = getBlurredSlope(images[c], 1);
        Mat nDH = deltaH / (deltaH + deltaV);
        Mat nDV = deltaV / (deltaH + deltaV);
        //Mat nDV{deltaH.size(), deltaH.type(), Scalar(1) };
        //Mat nDH{deltaH.size(), deltaH.type(), Scalar(1) };
        for (auto p : pairs) 
        {
            /// Get disparity
            subDisparities.push_back(getDisparityFromPairSGM(images[p[0]], images[p[1]], mask, cameras[p[0]], cameras[p[1]]));

            /// Get weights for disparity based on blurred x,y slope
            Mat validWeight;
            if ((cameras[p[0]].pos3D - cameras[p[1]].pos3D).y != 0) 
            {
                multiply(nDV / 255, subDisparities.back() != 0 & subDisparities.back() < 3000, validWeight, 1, nDV.type());
            }
            else
            {
                multiply(nDH / 255, subDisparities.back() != 0 & subDisparities.back() < 3000, validWeight, 1, nDH.type());
            }
            weights.push_back(validWeight);
            
            /// Multiply the disparities with their weights
            weightedSubDisparities.push_back(Mat{});
            multiply(subDisparities.back(), validWeight, weightedSubDisparities.back(), 1, CV_32F);
            //showImage("subDisparities.back()", subDisparities.back());
            
            /// Add weighted disparity to interim disparity sum
            add(interimDisparity, weightedSubDisparities.back(), interimDisparity, mask, interimDisparity.type());
            
            //showImage("weightedSubDisparities.back()", weightedSubDisparities.back()/1000);

            /// Add weight to weight sum
            add(weight, validWeight, weight, mask, weight.type());
            
        }
        divide(interimDisparity, weight, interimDisparity, 1, CV_32F);
        weight = Scalar(0);
        Mat stdDev{ images[0].size(), CV_32F, Scalar(0) };
        std::vector<Mat> deviations;
        for (auto d : subDisparities)
        {
            Mat deviation;
            subtract(interimDisparity, d, deviation, mask, interimDisparity.type());
            deviations.push_back(abs(deviation));
            stdDev += deviations.back();
        }
        stdDev /= deviations.size();

        Mat disparity{ images[0].size(), CV_16U, Scalar(0) };
        for (int i = 0; i < pairs.size(); i++)
        {
            Mat valid = deviations[i] <= stdDev+50;    ///VARIABLE
            Mat validWeights;
            multiply(valid / 255, weights[i], validWeights, 1, weights[i].type());
            add(weight, weights[i], weight, valid, CV_32F);
            Mat validDisps;
            weightedSubDisparities[i].copyTo(validDisps, valid);
            add(disparity, validDisps, disparity, noArray(), CV_16U);
        }

        divide(disparity, weight, disparity, 1, CV_16U);
        //showImage("Disparity", disparity*10);
        Mat shiftedDisparity = shiftDisparityPerspective(cameras[c], cameras[12], disparity);
        fillHoles(shiftedDisparity, 11);
        //showImage("shiftedDisparity", shiftedDisparity * 16);

        resize(shiftedDisparity, disparityResize, disparityRef.size());
        Mat error = abs(disparityResize - disparityRef);
        imshow("error", error*1000);
        //showImage("Error", error);

        Mat heatmap = getOrthogonalityFromCamera(centerDisparity, centerMask, cameras[12], cameras[13], cameras[c]);
        showImage("heatmap", heatmap);

        multiply(heatmap/255, shiftedDisparity != 0, heatmap, 1, heatmap.type());
        //std::cout << heatmap.type() << ", " << mask.type() << ", " << error.type() << std::endl;
        //for (int v = 0; v < heatmap.rows; v++)
        //{
        //    for (int u = 0; u < heatmap.cols; u++)
        //    {
        //        if (mask.at<uchar>(v, u) != 0) 
        //        {
        //            heat.push_back(heatmap.at<float>(v, u));
        //            err.push_back(error.at<ushort>(int(v/2), int(u/2)));
        //        }
        //    }
        //}
        //std::cout << heat.size() << std::endl;

        //showImage("Bin", heatmap);
        combinedHeatmap += heatmap;
        Mat weightedDisparity;
        multiply(heatmap, shiftedDisparity, weightedDisparity, 1, CV_16U);
        combinedDisparity += weightedDisparity;

    }
    //plt::figure_size(1200, 780);
    //plt::plot(heat, err, "r.");
    //plt::show();
    divide(combinedDisparity, combinedHeatmap, combinedDisparity, 1, CV_16U);
    showImage("Combined", combinedDisparity*22);

    resize(combinedDisparity, disparityResize, disparityRef.size());
    Mat error = abs(disparityResize - disparityRef) * 800;
    showImage("Error", error);
    resize(centerDisparity, disparityResize, disparityRef.size());
    Mat errorC = abs(disparityResize - disparityRef) * 800;
    showImage("ErrorC", errorC);

    showImage("Diff", error - errorC);
    showImage("Diff2", errorC - error);

    /// Disparity
    std::vector<cv::Point2i> maskPoints0 = Points3DtoPoints2D(maskPoints3D, cameras[0]);
    Mat mask0 = drawMask(images.back(), maskPoints0);
    showImage("Mask0", mask0);
    Mat disparity = getDisparityFromPairSGM(images[0], images[1], mask0, cameras[0], cameras[1]);
    showImage("disparity", disparity * 22);
    Mat shiftedDisparity = shiftDisparityPerspective(cameras[0], cameras[12], disparity);

    showImage("shiftedDisparity", shiftedDisparity * 22);
    fillHoles(shiftedDisparity, 11);
    showImage("shift", shiftedDisparity*22);
    Mat disparity2;
    copyTo(shiftedDisparity, disparity2, centerMask);
    disparity = disparity2;


    //Mat directionality2 = getOrthogonalityFromCamera(disparity, mask, cameras[12], cameras[13], cameras[0]);
    //showImage("Diff", directionality1 - directionality2);

    showImage("DispAft", disparity*16);

    /// Depth
    Mat depth = disparity2Depth(disparity, cameras[12], cameras[13]);

    //Mat reference = getIdealRef();
    //Mat resDepth;
    //resize(depth, resDepth, reference.size());
    //showImage("Diff", abs(reference - resDepth)*200);

    //Mat disparityRef = depth2Disparity(reference, cameras[12], cameras[11]);
    //resize(depth, depth, reference.size());
    //Mat disparityResize;

    //resize(disparity, disparityResize, disparityRef.size());

    //Mat error = abs(disparityResize - disparityRef) * 800;
    //showImage("Error", error);

}

