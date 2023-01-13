
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

// config
namespace HSV_BOUNDS {
    std::vector<int> CONE_UPPER_BOUND = {33, 255, 255};
    std::vector<int> CONE_LOWER_BOUND = {17, 106, 103};

    std::vector<int> CUBE_UPPER_BOUND = {143, 255, 255};
    std::vector<int> CUBE_LOWER_BOUND = {111, 62, 30};
}; 


int main(int, char **)
{
    std::cout << "cuda enabled device count: \n"; 
    std::cout << cv::cuda::getCudaEnabledDeviceCount()<< std::endl; 

    

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::Mat frame, hsv, coneMasked, cubeMasked, hsvMasked;
    cv::namedWindow("Original Camera", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("HSV Masked", cv::WINDOW_AUTOSIZE);
    while (true)
    {
        cap >> frame;
        cv::imshow("Original Camera", frame);

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); 
        // yellow HSL mask 

        cv::inRange(hsv, HSV_BOUNDS::CONE_LOWER_BOUND, HSV_BOUNDS::CONE_UPPER_BOUND, coneMasked);
        cv::inRange(hsv, HSV_BOUNDS::CUBE_LOWER_BOUND, HSV_BOUNDS::CUBE_UPPER_BOUND, cubeMasked);

        cv::add(coneMasked, cubeMasked, hsvMasked);  

        cv::imshow("HSV Masked", hsvMasked);

        // cvtColor(frame, edges, COLOR_BGR2GRAY);
        // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        // Canny(edges, edges, 0, 30, 3);

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}
