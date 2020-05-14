#include "dlibFaceSelect.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include "functions.h"

#include <opencv2/imgproc.hpp>

cv::Mat getFaceMask()
{
    std::string folder = "Images";
    std::vector<std::string> files = getImagesPathsFromFolder(folder);
    std::vector<cv::Mat> images;
    cv::Mat image = cv::imread(files[12], cv::IMREAD_GRAYSCALE);
    resize(image, image, cv::Size(), 0.25, 0.25);

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    IplImage iplImage = cvIplImage(image);
    dlib::cv_image<uchar> img(iplImage);
    std::vector<dlib::rectangle> faces = detector(img);
    dlib::full_object_detection shape = sp(img, faces[0]);

    cv::Mat mask{ image.size(), CV_8U, cv::Scalar(0) };

    std::vector<cv::Point2i> PointIndices;
    for (int i = 0; i < 17; i++)
        PointIndices.push_back(cv::Point2i(shape.part(i).x(), shape.part(i).y()));
    PointIndices.push_back(cv::Point2i(shape.part(24).x(), shape.part(24).y()));
    PointIndices.push_back(cv::Point2i(shape.part(19).x(), shape.part(19).y()));

    fillConvexPoly(mask, PointIndices, cv::Scalar(255), 16, 0);
    return mask;
}


cv::Mat getFaceCircle(cv::Mat& image)
{
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    IplImage iplImage = cvIplImage(image);
    dlib::cv_image<uchar> img(iplImage);
    std::vector<dlib::rectangle> faces = detector(img);
    dlib::full_object_detection shape = sp(img, faces[0]);

    cv::Mat mask{ image.size(), image.type(), cv::Scalar(0) };
    cv::RotatedRect selection{
        cv::Point2f(
            (shape.part(0).x() + shape.part(16).x()) / 2,
            (shape.part(0).y() + shape.part(16).y()) / 2 * 0.75 + shape.part(8).y() * 0.25
        ),
        cv::Size2f(
            shape.part(16).x() - shape.part(0).x(),
            (shape.part(8).y() - (shape.part(0).y() + shape.part(16).y()) / 2) * 1.7
        ),
        0.f
    };
    cv::ellipse(mask, selection, cv::Scalar{ 255 }, -1);
    return mask;
}