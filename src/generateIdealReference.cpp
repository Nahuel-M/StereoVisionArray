#include "generateIdealReference.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



std::vector<std::string> splitString(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    size_t pos = 0;
    size_t prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

void saveReference(std::string inputPath) {
    std::cout << "Start" << std::endl;
    std::ifstream ifs(inputPath);
    std::string data((std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));

    std::vector<std::string> lines = splitString(data, "\n");
    std::vector<std::vector<std::string>> points;
    for (auto& line : lines) {
        if (line == " " || line == "") break;
        points.push_back(splitString(line, " "));
    }

    cv::Point3d camPos = { 0, 0, -0.75 };
    cv::Size resolution = { 710, 540 };
    cv::Point2i halfres = resolution / 2;
    double f = 50e-3;
    double sensorSize = 36e-3;
    double pixel_size = sensorSize / resolution.width;
    cv::Mat screen = cv::Mat::ones(resolution, CV_64FC1);
    std::cout << screen.at<double>(10, 10) << std::endl;
    double fPerPos = f / pixel_size;
    for (auto i : points) {
        //std::cout << std::stof(i[0]) << ", " << std::stof(i[1]) << ", " << std::stof(i[2]) << ", " << std::endl;
        double z = -std::stof(i[1]) + 0.75;
        cv::Point2i pos2D = { (int)(std::stof(i[2]) * fPerPos / z), -(int)(std::stof(i[0]) * fPerPos / z) };
        //std::cout << pos2D << ", " << pos2D + halfres << std::endl;
        if (screen.at<double>(pos2D + halfres) > z)
            screen.at<double>(pos2D + halfres) = z;
        //std::cout << z << std::endl;
    }
    cv::imwrite("idealRef.png", screen);
    cv::FileStorage file("idealRef.yml", cv::FileStorage::WRITE);
    // Write to file!
    file << "R" << screen;

    imshow("ID", screen);
    cv::waitKey(0);

}


//
//clear screen
//camPos = [0, 0, -0.75];
//resolution = [1080, 1420] / 2;
//halfres = resolution / 2;
//f = 50e-3;
//sensorSize = 36e-3;
//pixel_size = sensorSize / resolution(2);
//screen = ones(resolution(1), resolution(2));
//Model = HeadModel;
//fPerPos = f / pixel_size
//
//for i = 1:size(Model, 1)
//z = Model(i, 3) - camPos(3);
//Pos2d = [
//		  round(Model(i, 1) * fPerPos / z), ...
//        round(Model(i, 2) * fPerPos / z)];
//if (screen(Pos2d(1) + halfres(1), Pos2d(2) + halfres(2)) > z)
//screen(Pos2d(1) + halfres(1), Pos2d(2) + halfres(2)) = z;
//end
//end
//imshow(screen, [0.5, 0.8]);
//set(gca, 'YDir', 'normal')