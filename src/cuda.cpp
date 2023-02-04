
#include "opencv4/opencv2/opencv.hpp"
using namespace cv; 
namespace HSV_BOUNDS {
    std::vector<int> CONE_UPPER_BOUND = {40, 255, 255};
    std::vector<int> CONE_LOWER_BOUND = {17, 71, 75};

    std::vector<int> CUBE_UPPER_BOUND = {143, 255, 255};
    std::vector<int> CUBE_LOWER_BOUND = {111, 62, 30};
}; 

const String window_detection_name = "HSV Masked";
int low_H = HSV_BOUNDS::CONE_LOWER_BOUND[0], low_S = HSV_BOUNDS::CONE_LOWER_BOUND[1], low_V = HSV_BOUNDS::CONE_LOWER_BOUND[2];
int high_H = HSV_BOUNDS::CONE_UPPER_BOUND[0], high_S = HSV_BOUNDS::CONE_UPPER_BOUND[1], high_V = HSV_BOUNDS::CONE_UPPER_BOUND[2];
static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

int main(int, char**) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::namedWindow("Original Camera", cv::WINDOW_FREERATIO);
    cv::resizeWindow("Original Camera", cv::Size(500, 500));
    cv::namedWindow("Edges", cv::WINDOW_FREERATIO); 
    cv::resizeWindow("Edges", cv::Size(500, 500));
    cv::namedWindow("Final", cv::WINDOW_FREERATIO); 
    cv::resizeWindow("Final", cv::Size(500, 500));
    cv::namedWindow(window_detection_name, cv::WINDOW_FREERATIO);
    createTrackbar("Low H", window_detection_name, &low_H, 255, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, 255, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, 255, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, 255, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, 255, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, 255, on_high_V_thresh_trackbar);
    cv::resizeWindow(window_detection_name, cv::Size(500, 500)); 


    cv::cuda::GpuMat frame; 
    while (true) {
        cap.read(frame);

        cv::imshow("Original Camera", frame);
    }


}