

#include <iostream>
#include "opencv2/opencv.hpp"
#include "apriltag/public/apriltag.h"
#include "apriltag/public/detector.h"

int main(int argc, char const *argv[])
{
    // cv::VideoCapture cap(0);

    // if (!cap.isOpened()) return -1; 

    cv::Mat image; 
 
    apriltag::detector detector(apriltag::AprilTagFamily::tag16h5);

    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("orig", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Gx", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Gy", cv::WINDOW_AUTOSIZE);
    // while (true)
    // {
    //     if (!cap.read(image)) continue;
    //     cv::imshow("orig", image);
        image = cv::imread("E:/Projects/cone_detection/mosaic.jpg", cv::IMREAD_UNCHANGED); 
        detector.detect(image);

        cv::waitKey(); 
        // if (cv::waitKey(30) >= 0)
        //     break;
    // }
     



    return 0;
}
