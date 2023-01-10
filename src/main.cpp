
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// config
std::vector<int> HSV_UPPER_BOUND = {33, 255, 255};
std::vector<int> HSV_LOWER_BOUND = {17, 106, 103};

int main(int, char **)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::Mat frame, hslMasked;
    cv::namedWindow("Original Camera", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("HSV Masked", cv::WINDOW_AUTOSIZE);
    while (true)
    {
        cap >> frame;
        cv::imshow("Original Camera", frame);

        cv::cvtColor(frame, hslMasked, cv::COLOR_BGR2HSV); 
        // yellow HSL mask 

        cv::inRange(hslMasked, HSV_LOWER_BOUND, HSV_UPPER_BOUND, hslMasked);

        cv::imshow("HSV Masked", hslMasked);

        // cvtColor(frame, edges, COLOR_BGR2GRAY);
        // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        // Canny(edges, edges, 0, 30, 3);

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}