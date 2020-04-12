#include "dlibFaceSelect.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/imgproc.hpp>

cv::Mat getFaceMask(cv::Mat& image)
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