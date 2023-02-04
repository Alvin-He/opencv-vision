
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
using namespace cv; 
// config
namespace HSV_BOUNDS {
    std::vector<int> CONE_UPPER_BOUND = {40, 255, 255};
    std::vector<int> CONE_LOWER_BOUND = {17, 71, 100};

    std::vector<int> CUBE_UPPER_BOUND = {143, 255, 255};
    std::vector<int> CUBE_LOWER_BOUND = {111, 62, 30};
}; 

namespace CONE_CONST {
    double MAX_WIDTH_OVER_HEIGHT = 0.8; 
    double WIDTH_OVER_HEIGHT = 0.63; 
}

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

int main(int, char **)
{
    std::cout << "cuda enabled device count: \n"; 
    std::cout << cv::cuda::getCudaEnabledDeviceCount()<< std::endl; 

    

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::Mat frame, hsv, coneMasked, cubeMasked, hsvMasked;
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
    while (true)
    {
        cap >> frame;
        // frame = cv::imread("/home/alh/opencv-vision/78A09405-9EEE-42E1-9AA7-680E614515CD.jpg", cv::IMREAD_UNCHANGED);
        frame.convertTo(frame, -1, 0.5); 
        cv::imshow("Original Camera", frame);
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); 
        
        // yellow HSL mask 

        // cv::add(coneMasked, cubeMasked, hsvMasked);

        // noise reduction 
        cv::Mat blurred;
        cv::GaussianBlur(hsv, blurred, cv::Size(3, 3), 0.8);

        // hsv masking 
        // cv::inRange(blurred, HSV_BOUNDS::CONE_LOWER_BOUND, HSV_BOUNDS::CONE_UPPER_BOUND, coneMasked);
        inRange(blurred, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), coneMasked);
        // cv::inRange(blurred, HSV_BOUNDS::CUBE_LOWER_BOUND, HSV_BOUNDS::CUBE_UPPER_BOUND, cubeMasked);

        // erode to reduce hsv masking errors
        int erodeSize = 5; 
        cv::Mat erodeKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erodeSize, erodeSize)); 
        cv::erode(coneMasked, coneMasked, erodeKernal);
        
        // dilate for sharpening edges
        // int dilateSize = 1; 
        // cv::Mat dilateKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize)); 
        // cv::dilate(coneMasked, coneMasked, dilateKernal); 
        

        // sobel image derivative and canny edge detection 
        cv::Mat Gx, Gy;
        cv::Sobel(coneMasked, Gx, CV_16S, 1, 0, 3);
        cv::Sobel(coneMasked, Gy, CV_16S, 0, 1, 3); 

        cv::Mat edges; 
        cv::Canny(Gx, Gy, edges, 50, 160);

        // find contours 
        std::vector<std::vector<cv::Point>> contours; 
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); 

        // approximate shapes and connections 
        std::vector<std::vector<cv::Point>> contourApproximations; 
        for (auto i = contours.begin(); i != contours.end(); i++) {
            std::vector<cv::Point> approxCurve;
            cv::approxPolyDP(*i, approxCurve, 10, true); 
            contourApproximations.push_back(approxCurve); 
        }
        cv::Mat contoursImage = cv::Mat::zeros(edges.size(), edges.type());
        // outlining 
        std::vector<std::vector<cv::Point>> convexHulls; 
        for (auto i = contourApproximations.begin(); i != contourApproximations.end(); i++) {
            std::vector<cv::Point> hull;
            cv::convexHull(*i, hull); 

            // filter out noise and other smaller detections
            // still need a maximum number of points
            if (3 > hull.size() || hull.size() > 13) continue;
            cv::Rect boundingBox = cv::boundingRect(*i); 
            if (boundingBox.area() <= 4000) continue; 
            double longerSideLength = boundingBox.height, shorterSideLength = boundingBox.width;
            if ( shorterSideLength > longerSideLength) {
                longerSideLength = boundingBox.width; 
                shorterSideLength = boundingBox.height;
            }
            if ((longerSideLength / shorterSideLength) > 2) continue; 
            std::cout << (hull.size())  << std::endl; 

            // if ((boundingBox.width / boundingBox.height) > 0.8) continue;
            // cones have a very distinctive feature: 
            // if u draw a line through the tip to the bottom rectangle in a 2d representation of a cone
            // the left and right side is always, no matter the aspect ratio/distance of the camera,
            // an reflection of each other

            // now, how tf do I determine what's the bottom base

            // find the vertical center and split the points into above and below arrays
            // int verticalCenter = boundingBox.y + boundingBox.height / 2;
            // std::vector<cv::Point> pointsAboveCenter, pointsBelowCenter; 
            // for (auto i = hull.begin(); i != hull.end(); i++) {
            //     if (i->y < verticalCenter) pointsAboveCenter.push_back(*i); 
            //     else pointsBelowCenter.push_back(*i); 
            // }

            // // finding the cords for base center that's used as one of the points for mid line  
            // int baseY = boundingBox.y + boundingBox.height;
            // int leftX = INT_MAX, rightX = 0; 
            // for (auto i = pointsBelowCenter.begin(); i != pointsBelowCenter.end(); i++) {
            //     if (i->y > baseY + 100 || i->y < baseY - 100) continue; // +- 100 buffering / error correction range
            //     if (i->x > rightX) rightX = i->x; 
            //     if (i->x < leftX) leftX = i->x; 
            // }
            // int baseX = (rightX - leftX) / 2 + leftX;

            // // find the cords for the top/tip point
            // int tipY = boundingBox.y;
            // leftX = INT_MAX; rightX = 0;
            // for (auto i = pointsAboveCenter.begin(); i != pointsAboveCenter.end(); i++){
            //     if (i->y > tipY + 100 || i->y < tipY - 100) continue; // +- 100 buffering / error correction range
            //     if (i->x > rightX) rightX = i->x;
            //     if (i->x < leftX) leftX = i->x;
            // }
            // int tipX = (rightX - leftX) / 2 + leftX;

            // std::cout << rightX << ";" << leftX << "\n"
            //           << hull << std::endl;
            convexHulls.push_back(hull);

            
            // cv::line(contoursImage, cv::Point(tipX, tipY), cv::Point(baseX, baseY), cv::Scalar(255,255,255), 2); 

            // cvtColor(frame, edges, COLOR_BGR2GRAY);
            // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
            // Canny(edges, edges, 0, 30, 3);

            // cv::waitKey();
        }

        // cv::drawContours(contoursImage, convexHulls, -1, cv::Scalar(255, 255, 255), 2);
        
        cv::Mat output; 
        frame.copyTo(output, coneMasked);
        cv::imshow("HSV Masked", output);


        // cv::Mat contoursImage = cv::Mat::zeros(edges.size(), edges.type());
        cv::drawContours(contoursImage, convexHulls, -1, cv::Scalar(255, 255, 255), 2);
        for (auto i = convexHulls.begin(); i != convexHulls.end(); i++) {
            cv::rectangle(frame, cv::boundingRect(*i), cv::Scalar(30,144,255), 2); 
        }
        cv::imshow("Final", frame); 
        cv::imshow("Edges", contoursImage);

        // // cvtColor(frame, edges, COLOR_BGR2GRAY);
        // // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        // // Canny(edges, edges, 0, 30, 3);

        // cv::waitKey(); 
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}
