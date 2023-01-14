

#include <iostream>
#include "opencv2/opencv.hpp"


namespace apriltag {
    //-----------------------------------------------------------------------------//
    // WARNING: Manual Memory Management for every thing in the apriltag namespace //
    //-----------------------------------------------------------------------------//
    #include <apriltag/apriltag.h>
    #include <apriltag/tag16h5.h>
}


int main(int argc, char const *argv[])
{
    cv::VideoCapture capture(0); 
    if (!capture.isOpened()) return -1; 

    cv::namedWindow("Origional"); 
    cv::namedWindow("Processed"); 

    apriltag::apriltag_detector_t *detector = apriltag::apriltag_detector_create(); 
    apriltag::apriltag_family_t *family = apriltag::tag16h5_create(); 
    apriltag::apriltag_detector_add_family(detector, family); 
    // detector->quad_decimate = double(0.0); // 2.0
    // detector->quad_sigma = double(0.0); 
    // detector->debug = true; 
    // detector->nthreads = 1; 
    // detector->refine_edges = true; 

    cv::Mat origional, processed; 

    while (true)
    {
        if (! capture.read(origional)) std::printf("failed to read frame"); 
        
        // cv::imshow("Origional", origional); 

        cv::cvtColor(origional, processed, cv::COLOR_BGR2GRAY);

        apriltag::image_u8_t frameHeader {
            .width = processed.cols,
            .height = processed.rows,
            .stride = processed.cols,
            .buf = processed.data
        };

        apriltag::zarray_t *detections = apriltag::apriltag_detector_detect(detector, &frameHeader); 

        std::cout << apriltag::zarray_size(detections) << std::endl; 

        for (int i = 0; i < apriltag::zarray_size(detections); i++) {
            apriltag::apriltag_detection_t *detection; 
            apriltag::zarray_get(detections, i, detection);


            

            // draw bounding boxes
            { cv::line(processed, cv::Point(detection->p[0][0], detection->p[0][1]),
                        cv::Point(detection->p[1][0], detection->p[1][1]),
                        cv::Scalar(0, 0xff, 0), 2);
            cv::line(processed, cv::Point(detection->p[0][0], detection->p[0][1]),
                        cv::Point(detection->p[3][0], detection->p[3][1]),
                        cv::Scalar(0, 0, 0xff), 2);
            cv::line(processed, cv::Point(detection->p[1][0], detection->p[1][1]),
                        cv::Point(detection->p[2][0], detection->p[2][1]),
                        cv::Scalar(0xff, 0, 0), 2);
            cv::line(processed, cv::Point(detection->p[2][0], detection->p[2][1]),
                        cv::Point(detection->p[3][0], detection->p[3][1]),
                        cv::Scalar(0xff, 0, 0), 2);}
            
            // apriltag::apriltag_detection_destroy(detection); 
        }
        // memory management
        // apriltag::image_u8_destroy(&frameHeader);
        apriltag::apriltag_detections_destroy(detections);

        cv::imshow("Processed", processed);
        if (cv::waitKey(30) >= 0)
            break;
    }

    apriltag::apriltag_detector_destroy(detector);
    apriltag::tag16h5_destroy(family); 


    return 0;
}
