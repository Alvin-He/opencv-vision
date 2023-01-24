
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

// config
namespace HSV_BOUNDS {
    std::vector<int> CONE_UPPER_BOUND = {33, 255, 255};
    std::vector<int> CONE_LOWER_BOUND = {20, 50, 30};

    std::vector<int> CUBE_UPPER_BOUND = {143, 255, 255};
    std::vector<int> CUBE_LOWER_BOUND = {111, 62, 30};
}; 

namespace CONE_CONST {
    double WIDTH_OVER_HEIGHT = 0.63; 
}

int main(int, char **)
{
    std::cout << "cuda enabled device count: \n"; 
    std::cout << cv::cuda::getCudaEnabledDeviceCount()<< std::endl; 

    

    // cv::VideoCapture cap(0);
    // if (!cap.isOpened()) return -1;
    cv::Mat frame, hsv, coneMasked, cubeMasked, hsvMasked;
    cv::namedWindow("Original Camera", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("HSV Masked", cv::WINDOW_KEEPRATIO);
    while (true)
    {
        // cap >> frame;
        frame = cv::imread("E:/Projects/cone_detection/78A09405-9EEE-42E1-9AA7-680E614515CD.jpg", cv::IMREAD_UNCHANGED);
        cv::imshow("Original Camera", frame);

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); 
        // yellow HSL mask 

        // cv::add(coneMasked, cubeMasked, hsvMasked);

        // noise reduction 
        cv::Mat blurred;
        cv::GaussianBlur(hsv, blurred, cv::Size(3, 3), 0.8);

        // hsv masking 
        cv::inRange(blurred, HSV_BOUNDS::CONE_LOWER_BOUND, HSV_BOUNDS::CONE_UPPER_BOUND, coneMasked);
        // cv::inRange(blurred, HSV_BOUNDS::CUBE_LOWER_BOUND, HSV_BOUNDS::CUBE_UPPER_BOUND, cubeMasked);

        // erode to reduce hsv masking errors
        int erodeSize = 17; 
        cv::Mat erodeKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erodeSize, erodeSize)); 
        cv::erode(coneMasked, coneMasked, erodeKernal);
        
        // dilate for sharpening edges
        int dilateSize = 10; 
        cv::Mat dilateKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize)); 
        cv::dilate(coneMasked, coneMasked, dilateKernal); 

        // sobel image derivative and canny edge detection 
        cv::Mat Gx, Gy;
        cv::Sobel(coneMasked, Gx, CV_32F, 1, 0, 3);
        cv::Sobel(coneMasked, Gy, CV_32F, 0, 1, 3);

        Gx.convertTo(Gx, CV_16S); 
        Gy.convertTo(Gy, CV_16S); 

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

        // outlining 
        std::vector<std::vector<cv::Point>> convexHulls; 
        for (auto i = contourApproximations.begin(); i != contourApproximations.end(); i++) {
            std::vector<cv::Point> hull;
            cv::convexHull(*i, hull); 

            // filter out noise and other smaller detections
            // still need a maximum number of points
            if (5 >= hull.size() || hull.size() >= 20) continue;

            cv::Rect boundingBox = cv::boundingRect(*i); 
            
            // cones have a very distinctive feature: 
            // if u draw a line through the tip to the bottom rectangle in a 2d representation of a cone
            // the left and right side is always, no matter the aspect ratio/distance of the camera,
            // an reflection of each other

            // now, how tf do I determine what's the bottom base

            // find the vertical center and split the points into above and below arrays
            int verticalCenter = boundingBox.y + boundingBox.height / 2;
            std::vector<cv::Point> pointsAboveCenter, pointsBelowCenter; 
            for (auto i = hull.begin(); i != hull.end(); i++) {
                if (i->y < verticalCenter) pointsAboveCenter.push_back(*i); 
                else pointsBelowCenter.push_back(*i); 
            }

            // finding the cords for base center that's used as one of the points for mid line  
            int baseY = boundingBox.y + boundingBox.height;
            int leftX = INT_MAX, rightX = 0; 
            for (auto i = pointsBelowCenter.begin(); i != pointsBelowCenter.end(); i++) {
                if (i->y > baseY + 100 || i->y < baseY - 100) continue; // +- 100 buffering / error correction range
                if (i->x > rightX) rightX = i->x; 
                if (i->x < leftX) leftX = i->x; 
            }
            int baseX = (rightX - leftX) / 2 + leftX;

            // find the cords for the top/tip point
            int tipY = boundingBox.y;
            leftX = INT_MAX; rightX = 0;
            for (auto i = pointsAboveCenter.begin(); i != pointsAboveCenter.end(); i++){
                if (i->y > tipY + 100 || i->y < tipY - 100) continue; // +- 100 buffering / error correction range
                if (i->x > rightX) rightX = i->x;
                if (i->x < leftX) leftX = i->x;
            }
            int tipX = (rightX - leftX) / 2 + leftX;


            // math on reflections
            int pointX, pointY; 

            double LOR_m = (baseY - tipY) / (baseX - tipX); 
            double LOR_b = baseY - (LOR_m * baseX);

            // equation of the line that connects the original to the reflected point
            double PLR_m = - 1 / LOR_m; 
            double PLR_b = pointY - (PLR_m * pointX); 

            // point when the 2 lines intercept  LOR = PLR so x(MID): LOR = PLR
            double MID_x = (1 / (LOR_m * PLR_b)) - (1 / (LOR_m * LOR_b));
            double MID_y = PLR_m * MID_x + PLR_b; //either equ is fine

            // add the negate delta x, y of MID and ORG to MID will get us the final ideal reflected point
            double DLT_x = MID_x - pointX; 
            double DLT_y = MID_y - pointY; 

            int RST_x = MID_x - DLT_x; 
            int RST_y = MID_y - DLT_y; 

            std::cout << rightX << ";" << leftX << "\n"
                      << hull << std::endl;
            convexHulls.push_back(hull);

            cv::Mat contoursImage = cv::Mat::zeros(edges.size(), edges.type());
            cv::drawContours(contoursImage, convexHulls, -1, cv::Scalar(255, 255, 255), 2);
            cv::line(contoursImage, cv::Point(tipX, tipY), cv::Point(baseX, baseY), cv::Scalar(255,255,255), 2); 
            cv::imshow("HSV Masked", contoursImage);

            // cvtColor(frame, edges, COLOR_BGR2GRAY);
            // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
            // Canny(edges, edges, 0, 30, 3);

            cv::waitKey();
        }



        // cv::Mat contoursImage = cv::Mat::zeros(edges.size(), edges.type());
        // cv::drawContours(contoursImage, convexHulls, -1, cv::Scalar(255, 255, 255), 2);

        // cv::imshow("HSV Masked", contoursImage);

        // // cvtColor(frame, edges, COLOR_BGR2GRAY);
        // // GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        // // Canny(edges, edges, 0, 30, 3);

        // cv::waitKey(); 
        // if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}
